"""Latent Neural ODE for system identification.

Models the dynamics using an Encoder-Decoder architecture:
1. Encoder: Bidirectional GRU infers q(z0 | x_1:T, u_1:T)
2. ODE: dz/dt = f_theta(z, u)
3. Decoder: x_hat = W @ z + b

Trained by maximizing the Evidence Lower Bound (ELBO):
Loss = MSE(x, x_hat) + KL(q(z0) || p(z0))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint


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


class EncoderGRU(nn.Module):
    """Encodes trajectory (x_1:T, u_1:T) into initial state distribution q(z0)."""
    
    def __init__(self, n_x: int, n_u: int, hidden_dim: int = 128, z_dim: int = 32):
        super().__init__()
        # Bidirectional GRU effectively reads forwards and backwards
        self.gru = nn.GRU(n_x + n_u, hidden_dim, batch_first=True, bidirectional=True)
        # Bidirectional means the final hidden state is 2 * hidden_dim
        self.fc_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, z_dim)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, n_x), u: (B, T, n_u)
        seq = torch.cat([x, u], dim=-1)  # (B, T, n_x + n_u)
        _, h_n = self.gru(seq)  
        # h_n is (num_layers * num_directions, B, hidden_dim) -> (2, B, hidden_dim)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, hidden_dim * 2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ODEDynamicsMLP(nn.Module):
    """f_theta(z, u): Generative dynamics in the latent space."""
    
    def __init__(self, z_dim: int, n_u: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(z_dim + n_u, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.net = nn.Sequential(*layers)
        self._u_interp: BatchLinearInterpolation | None = None

    def set_input(self, t: torch.Tensor, u: torch.Tensor):
        self._u_interp = BatchLinearInterpolation(t, u)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side.
        z shape: (B, z_dim) for batched, (z_dim,) for single
        """
        assert self._u_interp is not None, "Call set_input() before integration"
        u_t = self._u_interp(t)  # (B, n_u)
        
        if z.dim() == 1:
            u_t = u_t.squeeze(0)
            
        zu = torch.cat([z, u_t], dim=-1)
        return self.net(zu)


class LatentNeuralODE(nn.Module):
    """Latent Neural ODE Model combining Encoder, ODE Dynamics, and Decoder.
    
    Parameters
    ----------
    n_x : int
        Observation dimension (e.g., number of neurons/channels)
    n_u : int
        Input dimension (e.g., stimulators)
    z_dim : int
        Latent state dimension
    hidden_dim : int
        Hidden dimension for GRU and MLP
    n_layers : int
        Number of hidden layers in ODE MLP
    """
    
    def __init__(self, n_x: int, n_u: int, z_dim: int = 32, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.z_dim = z_dim
        
        self.encoder = EncoderGRU(n_x, n_u, hidden_dim, z_dim)
        self.ode_func = ODEDynamicsMLP(z_dim, n_u, hidden_dim, n_layers)
        self.decoder = nn.Linear(z_dim, n_x)

    def sample_z0(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq: torch.Tensor, u_seq: torch.Tensor, t: torch.Tensor, 
                method: str = "dopri5", deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass to compute ELBO components.
        
        Parameters
        ----------
        x_seq : (B, T, n_x)
        u_seq : (B, T, n_u)
        t : (T,)
        method : str, ODE solver method
        deterministic : bool, if True uses mu as z0 directly instead of sampling
        
        Returns
        -------
        x_pred : (B, T, n_x)
        mu : (B, z_dim)
        logvar : (B, z_dim)
        """
        # 1. Encode into initial state distribution
        mu, logvar = self.encoder(x_seq, u_seq)
        
        # 2. Sample initial state z0
        if deterministic:
            z0 = mu
        else:
            z0 = self.sample_z0(mu, logvar)
            
        # 3. Solve ODE forward in time
        self.ode_func.set_input(t, u_seq)
        z_seq = odeint(self.ode_func, z0, t, method=method, rtol=1e-4, atol=1e-5)  # (T, B, z_dim)
        z_seq = z_seq.permute(1, 0, 2)  # (B, T, z_dim)
        
        # 4. Decode latent states to observations
        x_pred = self.decoder(z_seq)  # (B, T, n_x)
        
        return x_pred, mu, logvar

    def predict(self, x_seq: torch.Tensor, u_seq: torch.Tensor, t: torch.Tensor, 
                method: str = "dopri5") -> torch.Tensor:
        """Deterministic prediction using mean of z0."""
        x_pred, _, _ = self.forward(x_seq, u_seq, t, method=method, deterministic=True)
        return x_pred
