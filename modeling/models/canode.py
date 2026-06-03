"""Control-Affine Neural ODE for system identification.

Models the dynamics as:
    dx/dt = f_theta(x) + g_phi(x) @ u(t)

where f_theta (drift) and g_phi (control susceptibility) are MLPs.
Integrated using torchdiffeq.odeint with MSE loss against recorded rates.

Supports batched training: multiple windows solved simultaneously on GPU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint


class DriftNet(nn.Module):
    """f_theta(x): Autonomous dynamics MLP."""

    def __init__(self, n_x: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(n_x, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, n_x))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ControlNet(nn.Module):
    """g_phi(x): Control susceptibility MLP.
    Maps state x to a (n_x, n_u) matrix."""

    def __init__(self, n_x: int, n_u: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        layers = [nn.Linear(n_x, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, n_x * n_u))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.net(x)
        if x.dim() == 1:
            return flat.view(self.n_x, self.n_u)
        return flat.view(-1, self.n_x, self.n_u)


class BatchLinearInterpolation:
    """Linearly interpolate batched input signals u(t).

    Parameters
    ----------
    t : Tensor, shape (T,)
    u : Tensor, shape (batch, T, n_u)
    """

    def __init__(self, t: torch.Tensor, u: torch.Tensor):
        self.t = t
        self.u = u  # (B, T, n_u)
        self.B = u.shape[0]

    def __call__(self, t_query: torch.Tensor) -> torch.Tensor:
        t = self.t
        t_q = torch.clamp(t_query, t[0], t[-1])
        idx = torch.searchsorted(t, t_q) - 1
        idx = torch.clamp(idx, 0, len(t) - 2)
        dt = t[idx + 1] - t[idx]
        w = ((t_q - t[idx]) / (dt + 1e-8)).unsqueeze(0).unsqueeze(-1)  # (1, 1, 1) for broadcast
        # u[:, idx] -> (B, n_u), u[:, idx+1] -> (B, n_u)
        return (1 - w) * self.u[:, idx] + w * self.u[:, idx + 1]  # (B, n_u)


class ControlAffineODE(nn.Module):
    """dx/dt = f_theta(x) + g_phi(x) @ u(t), with optional skip connections.

    The ODE integration captures smooth/slow dynamics. Skip connections add
    direct (non-integrated) pathways for high-frequency components:

        x_hat(t) = ODE(t) + skip_state(x_ode(t)) + skip_input(u(t))

    where skip_state is a learned linear residual on the ODE state and
    skip_input is a small MLP mapping inputs directly to outputs.

    Parameters
    ----------
    n_x : int
        State dimension
    n_u : int
        Input dimension
    hidden : int
        Hidden layer size for drift/control nets
    n_layers : int
        Number of hidden layers for drift/control nets
    use_skip : bool
        Whether to add skip connections (default: True)
    skip_hidden : int
        Hidden size for skip_input MLP (default: 64)
    """

    def __init__(self, n_x: int, n_u: int, hidden: int = 128, n_layers: int = 2,
                 use_skip: bool = True, skip_hidden: int = 64, skip_type: str = "mlp"):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.use_skip = use_skip
        self.skip_type = skip_type
        self.f = DriftNet(n_x, hidden, n_layers)
        self.g = ControlNet(n_x, n_u, hidden, n_layers)
        self._u_interp: BatchLinearInterpolation | None = None

        if use_skip:
            if skip_type == "linear":
                # State skip: purely linear residual without bias
                self.skip_state = nn.Linear(n_x, n_x, bias=False)
                nn.init.zeros_(self.skip_state.weight)

                # Input skip: purely linear map from u(t) -> x(t) without bias
                self.skip_input = nn.Linear(n_u, n_x, bias=False)
                nn.init.zeros_(self.skip_input.weight)
            else:
                # State skip: learned linear residual on ODE state
                # Initialized near-zero so ODE output dominates initially
                self.skip_state = nn.Linear(n_x, n_x)
                nn.init.zeros_(self.skip_state.weight)
                nn.init.zeros_(self.skip_state.bias)

                # Input skip: MLP mapping u(t) directly to output corrections
                # Bypasses integration entirely for fast input-driven responses
                self.skip_input = nn.Sequential(
                    nn.Linear(n_u, skip_hidden),
                    nn.Tanh(),
                    nn.Linear(skip_hidden, skip_hidden),
                    nn.Tanh(),
                    nn.Linear(skip_hidden, n_x),
                )
                # Initialize final layer near zero
                nn.init.zeros_(self.skip_input[-1].weight)
                nn.init.zeros_(self.skip_input[-1].bias)

    def set_input(self, t: torch.Tensor, u: torch.Tensor):
        """Set input trajectory for ODE integration.

        Parameters
        ----------
        t : Tensor, shape (T,)
        u : Tensor, shape (B, T, n_u) or (T, n_u)
        """
        if u.dim() == 2:
            u = u.unsqueeze(0)  # (1, T, n_u)
        self._u_interp = BatchLinearInterpolation(t, u)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side: dx/dt = f(x) + g(x) @ u(t).

        This is called by torchdiffeq and should NOT include skip connections.
        x shape: (B, n_x) for batched, (n_x,) for single
        """
        assert self._u_interp is not None, "Call set_input() before integration"

        u_t = self._u_interp(t)  # (B, n_u)
        drift = self.f(x)  # (B, n_x)
        g_x = self.g(x)  # (B, n_x, n_u)

        if x.dim() == 1:
            u_t = u_t.squeeze(0)  # (n_u,)
            control = g_x @ u_t  # (n_x,)
        else:
            control = torch.bmm(g_x, u_t.unsqueeze(-1)).squeeze(-1)  # (B, n_x)

        return drift + control

    def integrate(self, x0: torch.Tensor, t: torch.Tensor, method: str = "dopri5") -> torch.Tensor:
        """Integrate ODE. Input must be set via set_input() first.

        Parameters
        ----------
        x0 : (B, n_x) or (n_x,)
        t : (T,)

        Returns
        -------
        (T, B, n_x) or (T, n_x)
        """
        return odeint(self, x0, t, method=method, rtol=1e-4, atol=1e-5)

    def predict(self, x0: torch.Tensor, t: torch.Tensor, u: torch.Tensor,
                method: str = "dopri5") -> torch.Tensor:
        """Integrate ODE and apply skip connections.

        Parameters
        ----------
        x0 : (B, n_x)
        t : (T,)
        u : (B, T, n_u)
        method : str

        Returns
        -------
        x_pred : (B, T, n_x)  — already permuted to batch-first
        """
        self.set_input(t, u)
        x_ode = self.integrate(x0, t, method=method)  # (T, B, n_x)
        x_ode = x_ode.permute(1, 0, 2)  # (B, T, n_x)

        if not self.use_skip:
            return x_ode

        # State skip: linear residual on ODE output
        x_skip_state = self.skip_state(x_ode)  # (B, T, n_x)

        # Input skip: direct u(t) -> output, bypasses integration
        x_skip_input = self.skip_input(u)  # (B, T, n_x)

        return x_ode + x_skip_state + x_skip_input


class MultiRateControlAffineODE(nn.Module):
    """Multi-Scale / Multi-Rate Control-Affine Neural ODE.

    Decomposes the predicted state x_hat(t) into slow component x_slow(t) and
    fast component x_fast(t).
    
    x_slow is integrated using standard macro-steps.
    x_fast is integrated using sub-stepped Euler steps.

    Parameters
    ----------
    n_x : int
        State dimension
    n_u : int
        Input dimension
    hidden : int
        Hidden layer size for drift/control nets
    n_layers : int
        Number of hidden layers
    sub_steps : int
        Number of Euler integration sub-steps per macro step (default: 10)
    init_type : str
        Initialization split method: "zero_fast" or "learnable" (default: "zero_fast")
    """

    def __init__(self, n_x: int, n_u: int, hidden: int = 128, n_layers: int = 2,
                 sub_steps: int = 10, init_type: str = "zero_fast"):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.sub_steps = sub_steps
        self.init_type = init_type

        # Slow dynamics networks (macro-rate)
        self.f_slow = DriftNet(n_x, hidden, n_layers)
        self.g_slow = ControlNet(n_x, n_u, hidden, n_layers)
        self._u_interp: BatchLinearInterpolation | None = None

        # Fast dynamics networks (micro-rate)
        # Receives both slow and fast states concatenated: (B, 2 * n_x)
        fast_drift_layers = [nn.Linear(2 * n_x, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            fast_drift_layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        fast_drift_layers.append(nn.Linear(hidden, n_x))
        self.f_fast = nn.Sequential(*fast_drift_layers)

        fast_control_layers = [nn.Linear(2 * n_x, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            fast_control_layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        fast_control_layers.append(nn.Linear(hidden, n_x * n_u))
        self.g_fast_net = nn.Sequential(*fast_control_layers)

        # Initialization split parameterization
        if init_type == "learnable":
            self.init_slow = nn.Linear(n_x, n_x)
            self.init_fast = nn.Linear(n_x, n_x)
            
            # Initialize as Identity (slow) and Zero (fast)
            nn.init.eye_(self.init_slow.weight)
            nn.init.zeros_(self.init_slow.bias)
            nn.init.zeros_(self.init_fast.weight)
            nn.init.zeros_(self.init_fast.bias)

    def get_initial_states(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.init_type == "learnable":
            x_s0 = self.init_slow(x0)
            x_f0 = self.init_fast(x0)
        else:
            x_s0 = x0
            x_f0 = torch.zeros_like(x0)
        return x_s0, x_f0

    def g_fast(self, inp: torch.Tensor) -> torch.Tensor:
        flat = self.g_fast_net(inp)
        if inp.dim() == 1:
            return flat.view(self.n_x, self.n_u)
        return flat.view(-1, self.n_x, self.n_u)

    def set_input(self, t: torch.Tensor, u: torch.Tensor):
        if u.dim() == 2:
            u = u.unsqueeze(0)
        self._u_interp = BatchLinearInterpolation(t, u)

    def forward(self, t: torch.Tensor, x_slow: torch.Tensor) -> torch.Tensor:
        """Slow dynamics derivative: dx_slow/dt."""
        assert self._u_interp is not None, "Call set_input() before integration"
        u_t = self._u_interp(t)  # (B, n_u)
        drift = self.f_slow(x_slow)  # (B, n_x)
        g_x = self.g_slow(x_slow)  # (B, n_x, n_u)

        if x_slow.dim() == 1:
            u_t = u_t.squeeze(0)  # (n_u,)
            control = g_x @ u_t
        else:
            control = torch.bmm(g_x, u_t.unsqueeze(-1)).squeeze(-1)

        return drift + control

    def integrate_slow(self, x0: torch.Tensor, t: torch.Tensor, method: str = "dopri5") -> torch.Tensor:
        return odeint(self, x0, t, method=method, rtol=1e-4, atol=1e-5)

    def predict(self, x0: torch.Tensor, t: torch.Tensor, u: torch.Tensor,
                method: str = "dopri5") -> torch.Tensor:
        self.set_input(t, u)
        
        # Split initial state
        x_s0, x_f0 = self.get_initial_states(x0)
        
        # Integrate slow component
        x_slow = self.integrate_slow(x_s0, t, method=method)  # (T, B, n_x)
        x_slow = x_slow.permute(1, 0, 2)  # (B, T, n_x)

        # Micro sub-stepping for fast component
        dt = t[1] - t[0]
        M = self.sub_steps
        ds = dt / M

        B, T, n_x = x_slow.shape
        x_fast_list = []
        x_f_curr = x_f0
        x_fast_list.append(x_f_curr)

        for k in range(T - 1):
            x_s_k = x_slow[:, k]
            x_s_k1 = x_slow[:, k+1]
            u_k = u[:, k]
            u_k1 = u[:, k+1]

            for j in range(M):
                w = j / M
                x_s_interp = (1 - w) * x_s_k + w * x_s_k1
                u_interp = (1 - w) * u_k + w * u_k1

                inp = torch.cat([x_s_interp, x_f_curr], dim=-1)  # (B, 2 * n_x)
                drift = self.f_fast(inp)  # (B, n_x)
                g_x = self.g_fast(inp)  # (B, n_x, n_u)
                control = torch.bmm(g_x, u_interp.unsqueeze(-1)).squeeze(-1)

                dx_f_dt = drift + control
                x_f_curr = x_f_curr + ds * dx_f_dt

            x_fast_list.append(x_f_curr)

        x_fast = torch.stack(x_fast_list, dim=1)  # (B, T, n_x)

        return x_slow + x_fast


# ============================================================================
# Dataset
# ============================================================================



class TrajectoryWindowDataset(Dataset):
    """Dataset of trajectory windows for batched training.

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_total_steps)
    u : ndarray, shape (n_inputs, n_total_steps)
    dt : float
    window_size : int
    stride : int
    """

    def __init__(self, x, u, dt, window_size=200, stride=100):
        n_total = x.shape[1]
        t_base = np.arange(window_size, dtype=np.float32) * dt

        # Pre-build all windows as contiguous arrays
        starts = list(range(0, n_total - window_size, stride))
        self.x_windows = torch.tensor(
            np.stack([x[:, s:s + window_size].T for s in starts]),  # (N, T, n_x)
            dtype=torch.float32,
        )
        self.u_windows = torch.tensor(
            np.stack([u[:, s:s + window_size].T for s in starts]),  # (N, T, n_u)
            dtype=torch.float32,
        )
        self.t = torch.tensor(t_base, dtype=torch.float32)

    def __len__(self):
        return self.x_windows.shape[0]

    def __getitem__(self, idx):
        return self.x_windows[idx], self.u_windows[idx], self.t


# ============================================================================
# Training
# ============================================================================


def train_canode(
    model: ControlAffineODE,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 64,
    window_size: int = 200,
    stride: int = 100,
    val_fraction: float = 0.1,
    device: str = "cuda",
    verbose: bool = True,
    method: str = "dopri5",
    weight_decay: float = 1e-5,
    skip_weight_decay: float | None = None,
    spectral_alpha: float = 0.0,
) -> dict:
    """Train the control-affine Neural ODE with batched GPU integration.

    Parameters
    ----------
    model : ControlAffineODE
    x : (n_channels, n_total_steps)
    u : (n_inputs, n_total_steps)
    dt : float
    n_epochs : int
    lr : float
    batch_size : int
        Windows per GPU batch
    window_size : int
    stride : int
    val_fraction : float
    device : str
    verbose : bool
    method : str
    weight_decay : float
        Weight decay for base model parameters.
    skip_weight_decay : float, optional
        Separate weight decay for skip parameters.
    spectral_alpha : float
        Coefficient for frequency-aware spectral loss.

    Returns
    -------
    history : dict with training and validation losses
    """
    model = model.to(device)

    dataset = TrajectoryWindowDataset(x, u, dt, window_size, stride)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=True)

    if hasattr(model, "use_skip") and model.use_skip and skip_weight_decay is not None:
        # Separate skip parameters to apply different weight decay
        skip_params = []
        if hasattr(model, "skip_state"):
            skip_params.extend(list(model.skip_state.parameters()))
        if hasattr(model, "skip_input"):
            skip_params.extend(list(model.skip_input.parameters()))
        
        skip_param_ids = {id(p) for p in skip_params}
        base_params = [p for p in model.parameters() if id(p) not in skip_param_ids]
        
        optimizer = torch.optim.AdamW([
            {"params": base_params, "weight_decay": weight_decay},
            {"params": skip_params, "weight_decay": skip_weight_decay}
        ], lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_mse": [],
        "train_loss_fft": [],
        "val_loss_mse": [],
        "val_loss_fft": [],
    }

    if verbose:
        print(f"  Dataset: {len(dataset)} windows, {n_train} train, {n_val} val")
        print(f"  Batch size: {batch_size}, batches/epoch: {len(train_loader)}")

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_losses = []
        train_mses = []
        train_ffts = []
        for x_batch, u_batch, t_batch in train_loader:
            # x_batch: (B, T, n_x), u_batch: (B, T, n_u), t_batch: (B, T) [all same]
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            t_vec = t_batch[0].to(device)  # (T,) — same for all windows

            x0 = x_batch[:, 0]  # (B, n_x)
            x_pred = model.predict(x0, t_vec, u_batch, method=method)  # (B, T, n_x)

            loss_mse = F.mse_loss(x_pred, x_batch)
            
            # Compute real FFT over time dimension (dim=1)
            fft_pred = torch.fft.rfft(x_pred, dim=1)
            fft_true = torch.fft.rfft(x_batch, dim=1)
            mag_pred = torch.abs(fft_pred)
            mag_true = torch.abs(fft_true)
            loss_fft = F.mse_loss(mag_pred, mag_true)

            loss = loss_mse + spectral_alpha * loss_fft

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_mses.append(loss_mse.item())
            train_ffts.append(loss_fft.item())

        avg_train = np.mean(train_losses)
        avg_train_mse = np.mean(train_mses)
        avg_train_fft = np.mean(train_ffts)
        history["train_loss"].append(avg_train)
        history["train_loss_mse"].append(avg_train_mse)
        history["train_loss_fft"].append(avg_train_fft)

        # --- Validation ---
        model.eval()
        val_losses = []
        val_mses = []
        val_ffts = []
        with torch.no_grad():
            for x_batch, u_batch, t_batch in val_loader:
                x_batch = x_batch.to(device)
                u_batch = u_batch.to(device)
                t_vec = t_batch[0].to(device)

                x_pred = model.predict(x_batch[:, 0], t_vec, u_batch, method=method)
                
                loss_mse = F.mse_loss(x_pred, x_batch)
                
                # Compute real FFT over time dimension (dim=1)
                fft_pred = torch.fft.rfft(x_pred, dim=1)
                fft_true = torch.fft.rfft(x_batch, dim=1)
                mag_pred = torch.abs(fft_pred)
                mag_true = torch.abs(fft_true)
                loss_fft = F.mse_loss(mag_pred, mag_true)

                loss = loss_mse + spectral_alpha * loss_fft

                val_losses.append(loss.item())
                val_mses.append(loss_mse.item())
                val_ffts.append(loss_fft.item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        avg_val_mse = np.mean(val_mses) if val_mses else float("nan")
        avg_val_fft = np.mean(val_ffts) if val_ffts else float("nan")
        history["val_loss"].append(avg_val)
        history["val_loss_mse"].append(avg_val_mse)
        history["val_loss_fft"].append(avg_val_fft)
        
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            if spectral_alpha > 0.0:
                print(f"  Epoch {epoch:4d}/{n_epochs} | "
                      f"train: {avg_train:.6f} (mse: {avg_train_mse:.6f}, fft: {avg_train_fft:.6f}) | "
                      f"val: {avg_val:.6f} (mse: {avg_val_mse:.6f}, fft: {avg_val_fft:.6f}) | "
                      f"lr: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"  Epoch {epoch:4d}/{n_epochs} | "
                      f"train: {avg_train:.6f} | val: {avg_val:.6f} | "
                      f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    return history
