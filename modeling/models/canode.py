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
    """dx/dt = f_theta(x) + g_phi(x) @ u(t)

    Supports both single and batched integration.

    Parameters
    ----------
    n_x : int
        State dimension
    n_u : int
        Input dimension
    hidden : int
        Hidden layer size
    n_layers : int
        Number of hidden layers
    """

    def __init__(self, n_x: int, n_u: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.f = DriftNet(n_x, hidden, n_layers)
        self.g = ControlNet(n_x, n_u, hidden, n_layers)
        self._u_interp: BatchLinearInterpolation | None = None

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

    Returns
    -------
    history : dict with 'train_loss' and 'val_loss'
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": []}

    if verbose:
        print(f"  Dataset: {len(dataset)} windows, {n_train} train, {n_val} val")
        print(f"  Batch size: {batch_size}, batches/epoch: {len(train_loader)}")

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for x_batch, u_batch, t_batch in train_loader:
            # x_batch: (B, T, n_x), u_batch: (B, T, n_u), t_batch: (B, T) [all same]
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            t_vec = t_batch[0].to(device)  # (T,) — same for all windows

            model.set_input(t_vec, u_batch)  # (B, T, n_u)
            x0 = x_batch[:, 0]  # (B, n_x)
            x_pred = model.integrate(x0, t_vec, method=method)  # (T, B, n_x)
            x_pred = x_pred.permute(1, 0, 2)  # (B, T, n_x)

            loss = F.mse_loss(x_pred, x_batch)
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
            for x_batch, u_batch, t_batch in val_loader:
                x_batch = x_batch.to(device)
                u_batch = u_batch.to(device)
                t_vec = t_batch[0].to(device)

                model.set_input(t_vec, u_batch)
                x_pred = model.integrate(x_batch[:, 0], t_vec, method=method)
                x_pred = x_pred.permute(1, 0, 2)
                val_losses.append(F.mse_loss(x_pred, x_batch).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} | val: {avg_val:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    return history
