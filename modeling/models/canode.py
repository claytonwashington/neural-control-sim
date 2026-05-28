"""Control-Affine Neural ODE for system identification.

Models the dynamics as:
    dx/dt = f_theta(x) + g_phi(x) @ u(t)

where f_theta (drift) and g_phi (control susceptibility) are MLPs.
Integrated using torchdiffeq.odeint with MSE loss against recorded rates.

Strictly control-affine by construction, enabling downstream QP-based control.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint


class DriftNet(nn.Module):
    """f_theta(x): Autonomous dynamics MLP.

    Maps state x to its time derivative contribution from
    intrinsic (uncontrolled) dynamics.
    """

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

    Maps state x to a (n_x, n_u) matrix. The control contribution
    is then g(x) @ u, which is linear in u (control-affine).
    """

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
        """Returns (batch, n_x, n_u) or (n_x, n_u) matrix."""
        flat = self.net(x)
        if x.dim() == 1:
            return flat.view(self.n_x, self.n_u)
        return flat.view(-1, self.n_x, self.n_u)


class LinearInterpolation:
    """Linearly interpolate a discrete input signal u(t) for ODE solver."""

    def __init__(self, t: torch.Tensor, u: torch.Tensor):
        """
        Parameters
        ----------
        t : Tensor, shape (T,)
            Time points
        u : Tensor, shape (T, n_u) or (batch, T, n_u)
            Input values at each time point
        """
        self.t = t
        self.u = u

    def __call__(self, t_query: torch.Tensor) -> torch.Tensor:
        """Interpolate u at time t_query."""
        t = self.t
        # Clamp to valid range
        t_q = torch.clamp(t_query, t[0], t[-1])

        # Find interval
        idx = torch.searchsorted(t, t_q) - 1
        idx = torch.clamp(idx, 0, len(t) - 2)

        # Linear interpolation weight
        dt = t[idx + 1] - t[idx]
        w = (t_q - t[idx]) / (dt + 1e-8)

        if self.u.dim() == 2:
            return (1 - w) * self.u[idx] + w * self.u[idx + 1]
        else:  # batched
            w = w.unsqueeze(-1)
            return (1 - w) * self.u[:, idx] + w * self.u[:, idx + 1]


class ControlAffineODE(nn.Module):
    """dx/dt = f_theta(x) + g_phi(x) @ u(t)

    A Neural ODE with strictly control-affine structure.

    Parameters
    ----------
    n_x : int
        State dimension (number of recording channels)
    n_u : int
        Input dimension (number of fibers)
    hidden : int
        Hidden layer size for both f and g networks
    n_layers : int
        Number of hidden layers
    """

    def __init__(self, n_x: int, n_u: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.f = DriftNet(n_x, hidden, n_layers)
        self.g = ControlNet(n_x, n_u, hidden, n_layers)
        self._u_interp: LinearInterpolation | None = None

    def set_input(self, t: torch.Tensor, u: torch.Tensor):
        """Set the input trajectory for the next ODE integration.

        Parameters
        ----------
        t : Tensor, shape (T,)
        u : Tensor, shape (T, n_u)
        """
        self._u_interp = LinearInterpolation(t, u)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side: dx/dt = f(x) + g(x) @ u(t)."""
        assert self._u_interp is not None, "Call set_input() before integration"

        u_t = self._u_interp(t)  # (n_u,) or (batch, n_u)
        drift = self.f(x)  # (n_x,) or (batch, n_x)
        g_x = self.g(x)  # (n_x, n_u) or (batch, n_x, n_u)

        if x.dim() == 1:
            control = g_x @ u_t  # (n_x,)
        else:
            control = torch.bmm(g_x, u_t.unsqueeze(-1)).squeeze(-1)  # (batch, n_x)

        return drift + control

    def integrate(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integrate the ODE from x0 over time points t.

        Input must be set via set_input() before calling.

        Parameters
        ----------
        x0 : Tensor, shape (n_x,) or (batch, n_x)
        t : Tensor, shape (T,)

        Returns
        -------
        x_traj : Tensor, shape (T, n_x) or (T, batch, n_x)
        """
        return odeint(self, x0, t, method="dopri5", rtol=1e-5, atol=1e-6)


# ============================================================================
# Dataset and training utilities
# ============================================================================


class TrajectoryWindowDataset(Dataset):
    """Dataset of (x_window, u_window, t_window) trajectory windows.

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_total_steps)
    u : ndarray, shape (n_inputs, n_total_steps)
    dt : float
        Time step in seconds
    window_size : int
        Number of time steps per window
    stride : int
        Stride between consecutive windows
    """

    def __init__(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
        window_size: int = 200,
        stride: int = 100,
    ):
        self.windows = []
        n_total = x.shape[1]
        t_base = np.arange(window_size) * dt

        for start in range(0, n_total - window_size, stride):
            x_win = x[:, start : start + window_size].T  # (T, n_x)
            u_win = u[:, start : start + window_size].T  # (T, n_u)
            self.windows.append((
                torch.tensor(x_win, dtype=torch.float32),
                torch.tensor(u_win, dtype=torch.float32),
                torch.tensor(t_base, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def train_canode(
    model: ControlAffineODE,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    n_epochs: int = 500,
    lr: float = 1e-3,
    window_size: int = 200,
    stride: int = 100,
    val_fraction: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> dict:
    """Train the control-affine Neural ODE.

    Parameters
    ----------
    model : ControlAffineODE
    x : ndarray, shape (n_channels, n_total_steps)
    u : ndarray, shape (n_inputs, n_total_steps)
    dt : float
        Time step in seconds
    n_epochs : int
    lr : float
        Learning rate
    window_size : int
        Steps per training window
    stride : int
        Stride between windows
    val_fraction : float
        Fraction of windows for validation
    device : str
    verbose : bool

    Returns
    -------
    history : dict with 'train_loss' and 'val_loss' lists
    """
    model = model.to(device)

    dataset = TrajectoryWindowDataset(x, u, dt, window_size, stride)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, verbose=verbose
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for x_win, u_win, t_win in train_set:
            x_win = x_win.to(device)
            u_win = u_win.to(device)
            t_win = t_win.to(device)

            model.set_input(t_win, u_win)
            x0 = x_win[0]  # initial state
            x_pred = model.integrate(x0, t_win)  # (T, n_x)

            loss = F.mse_loss(x_pred, x_win)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_win, u_win, t_win in val_set:
                x_win = x_win.to(device)
                u_win = u_win.to(device)
                t_win = t_win.to(device)

                model.set_input(t_win, u_win)
                x_pred = model.integrate(x_win[0], t_win)
                val_losses.append(F.mse_loss(x_pred, x_win).item())

        avg_val = np.mean(val_losses)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} | val: {avg_val:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    return history
