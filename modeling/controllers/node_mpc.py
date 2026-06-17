"""Neural ODE Model Predictive Controller for Cleo closed-loop experiments.

Implements a LatencyIOProcessor that uses a trained causal LatentControlAffineODE
for model-predictive control (MPC) of neural firing rates.

Usage:
    from modeling.controllers.node_mpc import NeuralODEMPC
    ctrl = NeuralODEMPC(
        checkpoint_path="/path/to/model.pt",
        target_rate=target,  # (n_channels,) target firing rates
        sample_period_ms=1.0,
        mpc_horizon=20,
        compute_delay_ms=5.0,
    )
    sim.set_io_processor(ctrl)
    sim.run(duration)
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import brian2.only as b2
from brian2 import ms, mwatt, mm

from cleo.ioproc import LatencyIOProcessor, exp_firing_rate_estimate


class NeuralODEMPC(LatencyIOProcessor):
    """MPC controller using a trained causal latent CA-NODE.

    At each control step:
      1. Record observation (spike counts → exponentially smoothed rates)
      2. Encode past 200ms of rates → z₀ via causal GRU encoder
      3. Solve H-step MPC: min_{u} Σ ||decoder(z_k) - target||² + λ||u_k||²
         subject to z_{k+1} = z_k + dt*(f(z_k) + g(z_k)·u_k), 0 ≤ u ≤ u_max
      4. Apply first control action u₀

    Parameters
    ----------
    checkpoint_path : str
        Path to trained LatentControlAffineODE checkpoint (.pt file)
    target_rate : np.ndarray, shape (n_channels,)
        Target firing rates in Hz for rate clamping
    sample_period_ms : float
        How often process() is called (default 1.0 = 1kHz)
    mpc_horizon : int
        MPC planning horizon in steps (default 20)
    mpc_iters : int
        Number of Adam optimization steps per MPC solve (default 30)
    mpc_lr : float
        Learning rate for the MPC optimizer (default 0.5)
    mpc_batch_size : int
        Number of parallel candidate trajectories for multi-start
        optimization. B=1 is the original single-trajectory behavior.
        B>1 initializes multiple candidates with Gaussian perturbations
        and picks the best after optimization. (default 1)
    mpc_solver : str
        ODE integration method for MPC rollout. 'euler' for fixed-step
        Euler (fast, original behavior). 'dopri5' for adaptive Dormand-
        Prince (matches training solver, slower but more accurate).
        (default 'euler')
    lambda_u : float
        Control effort penalty weight (default 0.01)
    u_max : float
        Maximum irradiance in mW/mm² (default 50.0)
    compute_delay_ms : float
        Simulated processing latency in ms (default 5.0)
    tau_rate_ms : float
        Time constant for exponential firing rate estimation (default 20.0)
    past_window : int
        Number of past steps for the encoder (default 200)
    warmup_steps : int
        Steps before control starts (fill rate buffer) (default 200)
    device : str
        Torch device ('cpu' or 'cuda')
    light_name : str
        Name of the Light stimulator in Cleo
    probe_name : str
        Name of the Probe recorder
    mua_name : str
        Name of the MUA signal
    """

    def __init__(
        self,
        checkpoint_path: str,
        target_rate: np.ndarray,
        sample_period_ms: float = 1.0,
        mpc_horizon: int = 20,
        mpc_iters: int = 30,
        mpc_lr: float = 0.5,
        mpc_batch_size: int = 1,
        mpc_solver: str = "euler",
        lambda_u: float = 0.01,
        u_max: float = 50.0,
        compute_delay_ms: float = 5.0,
        tau_rate_ms: float = 20.0,
        past_window: int = 200,
        warmup_steps: int = 200,
        device: str = "cpu",
        light_name: str = "fibers",
        probe_name: str = "probe",
        mua_name: str = "mua",
        bidirectional: bool = False,
        light_name_exc: str = "fiber_exc",
        light_name_inh: str = "fiber_inh",
    ):
        super().__init__(sample_period=sample_period_ms * ms)

        self.mpc_horizon = mpc_horizon
        self.mpc_iters = mpc_iters
        self.mpc_lr = mpc_lr
        self.mpc_batch_size = mpc_batch_size
        self.mpc_solver = mpc_solver
        self.lambda_u = lambda_u
        self.u_max = u_max
        self.compute_delay = compute_delay_ms * ms
        self.tau_rate = tau_rate_ms * ms
        self.past_window = past_window
        self.warmup_steps = warmup_steps
        self.device = device
        self.light_name = light_name
        self.probe_name = probe_name
        self.mua_name = mua_name
        self.bidirectional = bidirectional
        self.light_name_exc = light_name_exc
        self.light_name_inh = light_name_inh

        # Load model
        self._load_model(checkpoint_path)

        # Target in normalized space
        self.target_raw = np.array(target_rate, dtype=np.float32)
        target_norm = (self.target_raw - self.x_mean) / self.x_std
        self.target_tensor = torch.tensor(
            target_norm, dtype=torch.float32, device=device
        )

        # State tracking
        self._rates = None  # current firing rate estimate (Brian units)
        self._rate_buffer = deque(maxlen=past_window)  # raw Hz values
        self._u_buffer = deque(maxlen=past_window)  # raw mW/mm² values
        self._step = 0
        self._u_plan = None  # warm-start MPC plan

        # Logging
        self.rate_log: list[np.ndarray] = []
        self.u_log: list[np.ndarray] = []
        self.z0_log: list[np.ndarray] = []
        self.cost_log: list[float] = []

    def _load_model(self, path: str):
        """Load trained causal model and extract normalization stats."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Normalization stats
        self.x_mean = np.asarray(ckpt["x_mean"]).flatten()
        self.x_std = np.asarray(ckpt["x_std"]).flatten()
        self.u_mean = np.asarray(ckpt["u_mean"]).flatten()
        self.u_std = np.asarray(ckpt["u_std"]).flatten()

        # Build model
        from modeling.models.latent_canode import LatentControlAffineODE

        n_x = len(self.x_mean)
        n_u = len(self.u_mean)
        z_dim = ckpt.get("z_dim", 64)
        hidden = ckpt.get("hidden", ckpt.get("hidden_dim", 128))
        n_layers = ckpt.get("n_layers", 2)

        self.model = LatentControlAffineODE(
            n_x=n_x,
            n_u=n_u,
            z_dim=z_dim,
            hidden_dim=hidden,
            n_layers=n_layers,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # Freeze all model weights (we only optimize u in MPC)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.n_x = n_x
        self.n_u = n_u
        self.z_dim = z_dim

        print(f"[NeuralODEMPC] Loaded model: n_x={n_x}, n_u={n_u}, z_dim={z_dim}")

    def _normalize_x(self, x_raw: np.ndarray) -> np.ndarray:
        """Normalize firing rates."""
        return (x_raw - self.x_mean) / self.x_std

    def _normalize_u(self, u_raw: np.ndarray) -> np.ndarray:
        """Normalize inputs."""
        return (u_raw - self.u_mean) / self.u_std

    def _denormalize_u(self, u_norm: np.ndarray) -> np.ndarray:
        """Denormalize inputs back to mW/mm²."""
        return u_norm * self.u_std + self.u_mean


    def _build_output_dict(self, u_raw):
        """Build the Cleo output dict, handling bidirectional sign-splitting.

        In bidirectional mode, the model already produces 2D control
        [u_exc, u_inh] since n_u=2. We route each component to its
        respective fiber device instead of sending both to a single device.

        The v1 plant sent the same scalar to both exc and inh fibers,
        which is self-defeating. Sign-splitting routes positive error
        (rate too low) to excitation and negative error (rate too high)
        to inhibition.
        """
        if self.bidirectional:
            return {
                self.light_name_exc: u_raw[0:1] * mwatt / mm**2,
                self.light_name_inh: u_raw[1:2] * mwatt / mm**2,
            }
        else:
            return {self.light_name: u_raw * mwatt / mm**2}

    def process(self, state_dict: dict, t_samp) -> Tuple[dict, float]:
        """Process one observation and compute control action.

        Parameters
        ----------
        state_dict : dict
            {recorder_name: state} from CLSimulator
        t_samp : Quantity
            Current simulation time

        Returns
        -------
        Tuple[dict, Quantity]
            {stimulator_name: control_signal}, delivery_time
        """
        # 1. Extract spike counts
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)

        # 2. Initialize firing rate estimate
        if self._rates is None:
            from brian2 import Hz
            self._rates = np.zeros(len(counts)) * Hz

        # 3. Exponential smoothing → firing rates
        self._rates = exp_firing_rate_estimate(
            counts, self.sample_period, self._rates, self.tau_rate
        )
        rate_hz = np.array(self._rates, dtype=np.float64)
        self._rate_buffer.append(rate_hz.copy())

        # 4. Store last applied u (or zeros if warmup)
        if len(self.u_log) > 0:
            self._u_buffer.append(self.u_log[-1].copy())
        else:
            self._u_buffer.append(np.zeros(self.n_u))

        self._step += 1
        self.rate_log.append(rate_hz.copy())

        # 5. During warmup: no control, just collect data
        if self._step < self.warmup_steps:
            u_out = np.zeros(self.n_u)
            self.u_log.append(u_out)
            return self._build_output_dict(u_out), t_samp

        # 6. Encode z₀ from past window
        z0 = self._encode_z0()
        self.z0_log.append(z0.detach().cpu().numpy().flatten())

        # 7. Solve MPC
        u_optimal, cost = self._solve_mpc(z0)
        self.cost_log.append(cost)

        # 8. Denormalize and clip
        u_raw = self._denormalize_u(u_optimal)
        u_raw = np.clip(u_raw, 0, self.u_max)
        self.u_log.append(u_raw.copy())

        # 9. Return with processing delay
        return (
            self._build_output_dict(u_raw),
            t_samp + self.compute_delay,
        )

    @torch.no_grad()
    def _encode_z0(self) -> torch.Tensor:
        """Encode z₀ from the past window using the causal GRU encoder."""
        # Build normalized tensors from buffers
        x_past = np.array(list(self._rate_buffer))  # (past_window, n_x)
        u_past = np.array(list(self._u_buffer))  # (past_window, n_u)

        x_norm = self._normalize_x(x_past)
        u_norm = self._normalize_u(u_past)

        x_t = torch.tensor(x_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_t = torch.tensor(u_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

        mu, logvar = self.model.encoder(x_t, u_t)
        return mu.squeeze(0)  # (z_dim,)

    def _solve_mpc(self, z0: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Solve the finite-horizon MPC problem with batched multi-start.

        When mpc_batch_size=1 (default), this is identical to the original
        single-trajectory optimization.

        When mpc_batch_size=B>1, initializes B candidate control sequences
        (one from warm-start, rest with Gaussian perturbations) and optimizes
        all B trajectories simultaneously. Picks the best candidate.

        min_{u₀...u_{H-1}} Σ_{k=0}^{H-1} ||decoder(z_k) - target||² + λ||u_k||²
        s.t. z_{k+1} = z_k + dt * (f(z_k) + g(z_k) · u_k)
             u_k ∈ [u_norm_lb, u_norm_ub]

        Returns
        -------
        u_first : np.ndarray, shape (n_u,)
            First control action (normalized)
        cost : float
            Final cost value of the best candidate
        """
        H = self.mpc_horizon
        B = self.mpc_batch_size
        use_dopri5 = (self.mpc_solver == "dopri5")
        dt = 1e-3  # 1ms timestep

        # Normalized bounds for u
        u_lb_norm = (0.0 - self.u_mean) / self.u_std
        u_ub_norm = (self.u_max - self.u_mean) / self.u_std
        u_lb_t = torch.tensor(u_lb_norm, dtype=torch.float32, device=self.device)
        u_ub_t = torch.tensor(u_ub_norm, dtype=torch.float32, device=self.device)

        # Initialize control sequence (warm start or zeros)
        if self._u_plan is not None:
            # Shift plan forward by one step
            u_base = torch.cat([self._u_plan[1:], self._u_plan[-1:]], dim=0)
        else:
            # Start at current mean (normalized zero)
            u_base = torch.zeros(H, self.n_u, device=self.device)

        if B == 1:
            # Original single-trajectory path (no batching overhead)
            u_var = u_base.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([u_var], lr=self.mpc_lr)

            best_cost = float("inf")
            best_u = u_base.clone()

            for iteration in range(self.mpc_iters):
                optimizer.zero_grad()

                # Rollout through learned ODE (Euler integration)
                z = z0.clone()
                cost = torch.tensor(0.0, device=self.device)

                if use_dopri5:
                    # Use adaptive ODE solver (matches training)
                    from torchdiffeq import odeint as _odeint
                    from modeling.models.canode import BatchLinearInterpolation
                    t_span = torch.linspace(0, H * dt, H + 1, device=self.device)
                    u_interp = BatchLinearInterpolation(
                        t_span, u_var.unsqueeze(0)  # (1, H, n_u) -> interpolate
                    )
                    def _ode_rhs(t, z_):
                        u_t = u_interp(t).squeeze(0)  # (n_u,)
                        d = self.model.f(z_.unsqueeze(0)).squeeze(0)
                        g_ = self.model.g(z_.unsqueeze(0)).squeeze(0)
                        return d + g_ @ u_t
                    z_traj = _odeint(_ode_rhs, z, t_span, method='dopri5',
                                     rtol=1e-4, atol=1e-5)  # (H+1, z_dim)
                    for k in range(H):
                        x_pred = self.model.decoder(z_traj[k])
                        tracking_err = ((x_pred - self.target_tensor) ** 2).sum()
                        effort = (u_var[k] ** 2).sum()
                        cost = cost + tracking_err + self.lambda_u * effort
                else:
                    for k in range(H):
                        # Decode and compute tracking error
                        x_pred = self.model.decoder(z)  # (n_x,)
                        tracking_err = ((x_pred - self.target_tensor) ** 2).sum()

                        # Control effort penalty
                        effort = (u_var[k] ** 2).sum()
                        cost = cost + tracking_err + self.lambda_u * effort

                        # Euler step through CA-ODE
                        drift = self.model.f(z.unsqueeze(0)).squeeze(0)  # (z_dim,)
                        g_z = self.model.g(z.unsqueeze(0)).squeeze(0)  # (z_dim, n_u)
                        control = g_z @ u_var[k]  # (z_dim,)
                        z = z + dt * (drift + control)

                # Backward + update
                cost.backward()
                optimizer.step()

                # Project u onto feasible set
                with torch.no_grad():
                    u_var.data.clamp_(u_lb_t, u_ub_t)

                # Track best
                if cost.item() < best_cost:
                    best_cost = cost.item()
                    best_u = u_var.detach().clone()

            # Save warm start for next step
            self._u_plan = best_u.detach()
            u_first = best_u[0].cpu().numpy()
            return u_first, best_cost

        else:
            # Batched multi-start optimization
            # Initialize B candidates: first from warm-start, rest perturbed
            u_init = u_base.unsqueeze(0).expand(B, -1, -1).clone()  # (B, H, n_u)
            if B > 1:
                # Add Gaussian noise to candidates 1..B-1
                noise = torch.randn(B - 1, H, self.n_u, device=self.device) * 0.1
                u_init[1:] = u_init[1:] + noise
                # Clamp the perturbed candidates to feasible set
                u_init[1:].clamp_(u_lb_t, u_ub_t)

            u_var = u_init.detach().requires_grad_(True)  # (B, H, n_u)
            optimizer = torch.optim.Adam([u_var], lr=self.mpc_lr)

            # Expand target for batched comparison
            target_batch = self.target_tensor.unsqueeze(0).expand(B, -1)  # (B, n_x)

            best_costs = torch.full((B,), float("inf"), device=self.device)
            best_u = u_init.clone()

            for iteration in range(self.mpc_iters):
                optimizer.zero_grad()

                # Batched rollout through learned ODE
                z = z0.unsqueeze(0).expand(B, -1).clone()  # (B, z_dim)
                costs = torch.zeros(B, device=self.device)

                for k in range(H):
                    # Decode: (B, z_dim) -> (B, n_x)
                    x_pred = self.model.decoder(z)
                    tracking_err = ((x_pred - target_batch) ** 2).sum(dim=1)  # (B,)

                    # Control effort: (B,)
                    effort = (u_var[:, k, :] ** 2).sum(dim=1)
                    costs = costs + tracking_err + self.lambda_u * effort

                    # Euler step: batched f and g
                    drift = self.model.f(z)  # (B, z_dim)
                    g_z = self.model.g(z)  # (B, z_dim, n_u)
                    control = torch.bmm(g_z, u_var[:, k, :].unsqueeze(-1)).squeeze(-1)  # (B, z_dim)
                    z = z + dt * (drift + control)

                # Sum of costs for backward (optimize all candidates jointly)
                total_cost = costs.sum()
                total_cost.backward()
                optimizer.step()

                # Project onto feasible set
                with torch.no_grad():
                    u_var.data.clamp_(u_lb_t, u_ub_t)

                # Track per-candidate best
                with torch.no_grad():
                    improved = costs < best_costs
                    best_costs[improved] = costs[improved]
                    best_u[improved] = u_var.detach()[improved]

            # Select the candidate with lowest cost
            best_idx = best_costs.argmin().item()
            winner = best_u[best_idx]  # (H, n_u)

            # Save warm start for next step (from the best candidate)
            self._u_plan = winner.detach()

            u_first = winner[0].cpu().numpy()
            return u_first, best_costs[best_idx].item()

    def get_results(self) -> dict:
        """Return logged data for analysis."""
        return {
            "rates": np.array(self.rate_log) if self.rate_log else np.array([]),
            "u": np.array(self.u_log) if self.u_log else np.array([]),
            "z0": np.array(self.z0_log) if self.z0_log else np.array([]),
            "costs": np.array(self.cost_log) if self.cost_log else np.array([]),
        }

    def _base_reset(self):
        """Reset state for a new simulation run."""
        super()._base_reset()
        self._rates = None
        self._rate_buffer.clear()
        self._u_buffer.clear()
        self._step = 0
        self._u_plan = None
        self.rate_log.clear()
        self.u_log.clear()
        self.z0_log.clear()
        self.cost_log.clear()


class PeriodicNeuralODEMPC(NeuralODEMPC):
    def __init__(self, *args, reencode_period=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.reencode_period = reencode_period
        self._z_current = None

    def _base_reset(self):
        super()._base_reset()
        self._z_current = None

    def process(self, state_dict: dict, t_samp):
        # 1. Extract spike counts
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)

        # 2. Initialize firing rate estimate
        if self._rates is None:
            from brian2 import Hz
            self._rates = np.zeros(len(counts)) * Hz

        # 3. Exponential smoothing
        self._rates = exp_firing_rate_estimate(
            counts, self.sample_period, self._rates, self.tau_rate
        )
        rate_hz = np.array(self._rates, dtype=np.float64)
        self._rate_buffer.append(rate_hz.copy())

        # 4. Store last applied u (or zeros if warmup)
        if len(self.u_log) > 0:
            self._u_buffer.append(self.u_log[-1].copy())
        else:
            self._u_buffer.append(np.zeros(self.n_u))

        self._step += 1
        self.rate_log.append(rate_hz.copy())

        from brian2 import mwatt, mm
        
        # 5. During warmup: no control
        if self._step < self.warmup_steps:
            u_out = np.zeros(self.n_u)
            self.u_log.append(u_out)
            return self._build_output_dict(u_out), t_samp

        # 6. Periodic Re-encoding
        if (self._step - self.warmup_steps) % self.reencode_period == 0 or self._z_current is None:
            self._z_current = self._encode_z0()
        else:
            # Forward simulate 1 step using last applied u
            u_prev = self._u_buffer[-1]
            u_norm = self._normalize_u(u_prev)
            u_t = torch.tensor(u_norm, dtype=torch.float32, device=self.device)
            
            dt = 1e-3
            z = self._z_current.unsqueeze(0)
            with torch.no_grad():
                dz = self.model.f(z) + (self.model.g(z) @ u_t.unsqueeze(-1)).squeeze(-1)
                self._z_current = (z + dt * dz).squeeze(0)

        z0 = self._z_current
        self.z0_log.append(z0.detach().cpu().numpy().flatten())

        # 7. Solve MPC
        u_optimal, cost = self._solve_mpc(z0)
        self.cost_log.append(cost)

        # 8. Denormalize and clip
        u_raw = self._denormalize_u(u_optimal)
        u_raw = np.clip(u_raw, 0, self.u_max)
        self.u_log.append(u_raw.copy())

        # 9. Return with processing delay
        return (
            self._build_output_dict(u_raw),
            t_samp + self.compute_delay,
        )


class AdaptivePeriodicNeuralODEMPC(PeriodicNeuralODEMPC):
    """MPC with online model adaptation.

    At each control step, before solving MPC:
      1. Forward-predict what the rate should be given last step's z and u
      2. Compare to actual observed rate
      3. Update selected model weights via gradient descent on prediction error
      4. Then solve MPC with updated model

    Parameters
    ----------
    adapt_lr : float
        Learning rate for online adaptation (default 1e-5)
    adapt_params : str
        Which parameters to fine-tune: 'decoder', 'g+decoder', or
        'f+g+decoder' (default 'g+decoder')
    adapt_grad_clip : float
        Maximum gradient norm for stability (default 1.0)
    """

    def __init__(self, *args, adapt_lr: float = 1e-5,
                 adapt_params: str = "g+decoder",
                 adapt_grad_clip: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapt_lr = adapt_lr
        self.adapt_grad_clip = adapt_grad_clip
        self.adapt_params_name = adapt_params

        # Select parameters to adapt
        if adapt_params == "decoder":
            params = list(self.model.decoder.parameters())
        elif adapt_params == "g+decoder":
            params = (list(self.model.g.parameters())
                      + list(self.model.decoder.parameters()))
        elif adapt_params == "f+g+decoder":
            params = (list(self.model.f.parameters())
                      + list(self.model.g.parameters())
                      + list(self.model.decoder.parameters()))
        else:
            raise ValueError(f"Unknown adapt_params: {adapt_params}")

        # Unfreeze selected parameters
        for p in params:
            p.requires_grad_(True)

        self._adapt_params = params
        self.adapt_optimizer = torch.optim.Adam(params, lr=adapt_lr)

        # State for prediction error computation
        self._last_z = None
        self._last_u_norm = None

        # Logging
        self.adapt_loss_log: list[float] = []

    def _base_reset(self):
        super()._base_reset()
        self._last_z = None
        self._last_u_norm = None
        self.adapt_loss_log.clear()

    def _adapt_step(self, observed_rate_hz: np.ndarray):
        """Online adaptation: update model based on prediction error.

        Compares the model's 1-step prediction (from last step's z and u)
        against the actual observed rate. Updates g(z) + decoder weights.
        """
        if self._last_z is None or self._last_u_norm is None:
            return  # no prediction to compare yet

        with torch.enable_grad():
            # Forward sim one step from last z with last u
            z = self._last_z.detach().clone()
            u = self._last_u_norm.detach().clone()
            dt = 1e-3

            z_in = z.unsqueeze(0)
            drift = self.model.f(z_in).squeeze(0)
            g_z = self.model.g(z_in).squeeze(0)
            z_pred = z + dt * (drift + g_z @ u)

            # Decode predicted rate
            x_pred = self.model.decoder(z_pred)

            # Compare to actual observed rate (normalized)
            x_obs_norm = self._normalize_x(observed_rate_hz)
            x_obs_t = torch.tensor(
                x_obs_norm, dtype=torch.float32, device=self.device
            )

            adapt_loss = ((x_pred - x_obs_t) ** 2).mean()

            self.adapt_optimizer.zero_grad()
            adapt_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._adapt_params, self.adapt_grad_clip
            )
            self.adapt_optimizer.step()

        self.adapt_loss_log.append(adapt_loss.item())

    def process(self, state_dict: dict, t_samp):
        # 1. Extract spike counts
        i_chan, t_spikes, counts = state_dict[self.probe_name][self.mua_name]
        counts = np.asarray(counts, dtype=float)

        # 2. Initialize firing rate estimate
        if self._rates is None:
            from brian2 import Hz
            self._rates = np.zeros(len(counts)) * Hz

        # 3. Exponential smoothing
        self._rates = exp_firing_rate_estimate(
            counts, self.sample_period, self._rates, self.tau_rate
        )
        rate_hz = np.array(self._rates, dtype=np.float64)
        self._rate_buffer.append(rate_hz.copy())

        # 4. Store last applied u (or zeros if warmup)
        if len(self.u_log) > 0:
            self._u_buffer.append(self.u_log[-1].copy())
        else:
            self._u_buffer.append(np.zeros(self.n_u))

        self._step += 1
        self.rate_log.append(rate_hz.copy())

        from brian2 import mwatt, mm

        # 5. During warmup: no control
        if self._step < self.warmup_steps:
            u_out = np.zeros(self.n_u)
            self.u_log.append(u_out)
            return self._build_output_dict(u_out), t_samp

        # 5b. Online adaptation step (before MPC solve)
        self._adapt_step(rate_hz)

        # 6. Periodic Re-encoding
        if ((self._step - self.warmup_steps) % self.reencode_period == 0
                or self._z_current is None):
            self._z_current = self._encode_z0()
        else:
            # Forward simulate 1 step using last applied u
            u_prev = self._u_buffer[-1]
            u_norm = self._normalize_u(u_prev)
            u_t = torch.tensor(u_norm, dtype=torch.float32, device=self.device)

            dt = 1e-3
            z = self._z_current.unsqueeze(0)
            with torch.no_grad():
                dz = (self.model.f(z)
                      + (self.model.g(z) @ u_t.unsqueeze(-1)).squeeze(-1))
                self._z_current = (z + dt * dz).squeeze(0)

        z0 = self._z_current
        self.z0_log.append(z0.detach().cpu().numpy().flatten())

        # Save state for next step's adaptation
        self._last_z = z0.detach().clone()
        if len(self.u_log) > 0:
            u_norm = self._normalize_u(self.u_log[-1])
            self._last_u_norm = torch.tensor(
                u_norm, dtype=torch.float32, device=self.device
            )

        # 7. Solve MPC
        u_optimal, cost = self._solve_mpc(z0)
        self.cost_log.append(cost)

        # 8. Denormalize and clip
        u_raw = self._denormalize_u(u_optimal)
        u_raw = np.clip(u_raw, 0, self.u_max)
        self.u_log.append(u_raw.copy())

        # 9. Return with processing delay
        return (
            self._build_output_dict(u_raw),
            t_samp + self.compute_delay,
        )

    def get_results(self) -> dict:
        """Return logged data including adaptation loss."""
        base = super().get_results()
        base["adapt_loss"] = (np.array(self.adapt_loss_log)
                              if self.adapt_loss_log else np.array([]))
        return base
