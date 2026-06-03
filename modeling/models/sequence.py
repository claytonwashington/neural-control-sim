"""Discrete-time sequence models (GRU/LSTM) for system identification.

Models the dynamics as a discrete-time recurrent update:
    h_t = RNNCell(u_t, h_{t-1})
    x_hat_t = MLP(h_t)

where the initial hidden state h_0 is mapped from x_0.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from modeling.config import DEFAULT_SEED


class GRUModel(nn.Module):
    """Discrete-time GRU model for neural state prediction.

    Parameters
    ----------
    n_x : int
        State dimension (e.g. 50 recording channels)
    n_u : int
        Input dimension (e.g. 2 stimulation channels)
    hidden : int
        Hidden dimension of the GRU cell
    num_layers : int
        Number of GRU layers
    """

    def __init__(self, n_x: int, n_u: int, hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.hidden = hidden
        self.num_layers = num_layers

        # Network to initialize GRU hidden state from x0
        self.init_net = nn.Sequential(
            nn.Linear(n_x, hidden),
            nn.Tanh(),
        )

        # GRU network
        self.gru = nn.GRU(
            input_size=n_u,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output decoder network
        self.out_net = nn.Linear(hidden, n_x)

    def predict(self, x0: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Predict the state trajectory.

        Parameters
        ----------
        x0 : Tensor, shape (B, n_x)
            Initial states
        t : Tensor, shape (T,)
            Time vectors (unused for GRU but kept for interface compatibility)
        u : Tensor, shape (B, T, n_u)
            Control inputs

        Returns
        -------
        x_pred : Tensor, shape (B, T, n_x)
            Predicted states
        """
        B, T, _ = u.shape

        # Initialize hidden state from x0
        h0 = self.init_net(x0)  # (B, hidden)
        h0 = h0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()  # (num_layers, B, hidden)

        # Run GRU over inputs
        out, _ = self.gru(u, h0)  # (B, T, hidden)

        # Decode hidden states to predictions
        x_pred = self.out_net(out)  # (B, T, n_x)

        # Force x_pred[:, 0] = x0 to match the exact initial condition of ODE integration
        x_pred = x_pred.clone()
        x_pred[:, 0] = x0

        return x_pred


from modeling.models.canode import DriftNet, ControlNet


class DiscreteControlAffineModel(nn.Module):
    """Discrete-time Control-Affine state-space model (Euler-discretized CA-NODE).

    dx/dt = f(x) + g(x)*u, discretized as:
    x_t = x_{t-1} + f(x_{t-1}) + g(x_{t-1}) * u_t
    """

    def __init__(self, n_x: int, n_u: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.f = DriftNet(n_x, hidden, n_layers)
        self.g = ControlNet(n_x, n_u, hidden, n_layers)

    def predict(self, x0: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Predict state trajectory using discrete-time Euler steps.

        Parameters
        ----------
        x0 : Tensor, shape (B, n_x)
        t : Tensor, shape (T,)
        u : Tensor, shape (B, T, n_u)
        """
        B, T, _ = u.shape
        x_pred = []
        x_curr = x0
        x_pred.append(x_curr)

        # Dynamic dt calculation from time vector (defaults to 0.001s / 1ms)
        dt = (t[1] - t[0]).item() if len(t) > 1 else 0.001

        for t_idx in range(1, T):
            u_t = u[:, t_idx - 1]  # (B, n_u)
            drift = self.f(x_curr)  # (B, n_x)
            g_x = self.g(x_curr)  # (B, n_x, n_u)
            control = torch.bmm(g_x, u_t.unsqueeze(-1)).squeeze(-1)  # (B, n_x)

            # Euler step: x_next = x_curr + (dx/dt) * dt
            x_next = x_curr + (drift + control) * dt
            x_pred.append(x_next)
            x_curr = x_next

        return torch.stack(x_pred, dim=1)  # (B, T, n_x)



# ============================================================================
# Training Function
# ============================================================================


def train_sequence_model(
    model: nn.Module,
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
    seed: int = DEFAULT_SEED,
) -> dict:
    """Train the sequence model with batched sequence learning.

    Parameters
    ----------
    model : nn.Module
    x : (n_channels, n_total_steps)
    u : (n_inputs, n_total_steps)
    dt : float
    n_epochs : int
    lr : float
    batch_size : int
    window_size : int
    stride : int
    val_fraction : float
    device : str
    verbose : bool
    seed : int
        Random seed for validation split.

    Returns
    -------
    history : dict
    """
    model = model.to(device)

    # Reuse the same TrajectoryWindowDataset structure
    from modeling.models.canode import TrajectoryWindowDataset
    dataset = TrajectoryWindowDataset(x, u, dt, window_size, stride)
    
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)

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
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            t_vec = t_batch[0].to(device)

            x0 = x_batch[:, 0]
            x_pred = model.predict(x0, t_vec, u_batch)  # (B, T, n_x)

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

                x_pred = model.predict(x_batch[:, 0], t_vec, u_batch)
                val_losses.append(F.mse_loss(x_pred, x_batch).item())

        avg_val = np.mean(val_losses) if val_losses else float("nan")
        history["val_loss"].append(avg_val)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train: {avg_train:.6f} | val: {avg_val:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    return history
