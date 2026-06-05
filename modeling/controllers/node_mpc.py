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
    ):
        super().__init__(sample_period=sample_period_ms * ms)

        self.mpc_horizon = mpc_horizon
        self.mpc_iters = mpc_iters
        self.mpc_lr = mpc_lr
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
        self.x_mean = ckpt["x_mean"].numpy().flatten()
        self.x_std = ckpt["x_std"].numpy().flatten()
        self.u_mean = ckpt["u_mean"].numpy().flatten()
        self.u_std = ckpt["u_std"].numpy().flatten()

        # Build model
        from modeling.models.latent_canode import LatentControlAffineODE

        n_x = len(self.x_mean)
        n_u = len(self.u_mean)
        z_dim = ckpt.get("z_dim", 64)
        hidden = ckpt.get("hidden_dim", 128)
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
            return {self.light_name: u_out * mwatt / mm**2}, t_samp

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
            {self.light_name: u_raw * mwatt / mm**2},
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
        """Solve the finite-horizon MPC problem.

        min_{u₀...u_{H-1}} Σ_{k=0}^{H-1} ||decoder(z_k) - target||² + λ||u_k||²
        s.t. z_{k+1} = z_k + dt * (f(z_k) + g(z_k) · u_k)
             u_k ∈ [u_norm_lb, u_norm_ub]

        Returns
        -------
        u_first : np.ndarray, shape (n_u,)
            First control action (normalized)
        cost : float
            Final cost value
        """
        H = self.mpc_horizon
        dt = 1e-3  # 1ms timestep

        # Normalized bounds for u
        u_lb_norm = (0.0 - self.u_mean) / self.u_std
        u_ub_norm = (self.u_max - self.u_mean) / self.u_std
        u_lb_t = torch.tensor(u_lb_norm, dtype=torch.float32, device=self.device)
        u_ub_t = torch.tensor(u_ub_norm, dtype=torch.float32, device=self.device)

        # Initialize control sequence (warm start or zeros)
        if self._u_plan is not None:
            # Shift plan forward by one step
            u_init = torch.cat([self._u_plan[1:], self._u_plan[-1:]], dim=0)
        else:
            # Start at current mean (normalized zero)
            u_init = torch.zeros(H, self.n_u, device=self.device)

        u_var = u_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([u_var], lr=self.mpc_lr)

        best_cost = float("inf")
        best_u = u_init.clone()

        for iteration in range(self.mpc_iters):
            optimizer.zero_grad()

            # Rollout through learned ODE (Euler integration)
            z = z0.clone()
            cost = torch.tensor(0.0, device=self.device)

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

        # Return first action (normalized)
        u_first = best_u[0].cpu().numpy()
        return u_first, best_cost

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
