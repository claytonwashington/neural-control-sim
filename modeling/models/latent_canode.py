"""Latent Control-Affine Neural ODE for system identification.

Marries the denoising properties of a Latent ODE with the physical inductive
bias of a Control-Affine architecture:
    dz/dt = f_theta(z) + g_phi(z) @ u(t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint

from modeling.models.canode import DriftNet, ControlNet, BatchLinearInterpolation
from modeling.models.latent_node import EncoderGRU


class LatentControlAffineODE(nn.Module):
    """Latent CA-NODE Model.
    
    Parameters
    ----------
    n_x : int
        Observation dimension (e.g., number of neurons/channels)
    n_u : int
        Input dimension (e.g., stimulators)
    z_dim : int
        Latent state dimension
    hidden_dim : int
        Hidden dimension for GRU and MLPs
    n_layers : int
        Number of hidden layers in ODE MLPs
    """
    
    def __init__(self, n_x: int, n_u: int, z_dim: int = 32, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.z_dim = z_dim
        
        # Inference
        self.encoder = EncoderGRU(n_x, n_u, hidden_dim, z_dim)
        
        # Control-Affine Generative Dynamics in latent space
        self.f = DriftNet(z_dim, hidden_dim, n_layers)
        self.g = ControlNet(z_dim, n_u, hidden_dim, n_layers)
        self._u_interp: BatchLinearInterpolation | None = None
        
        # Readout
        self.decoder = nn.Linear(z_dim, n_x)

    def set_input(self, t: torch.Tensor, u: torch.Tensor):
        self._u_interp = BatchLinearInterpolation(t, u)

    def ode_func(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ODE right-hand side in latent space."""
        assert self._u_interp is not None, "Call set_input() before integration"
        
        u_t = self._u_interp(t)  # (B, n_u)
        drift = self.f(z)  # (B, z_dim)
        g_z = self.g(z)  # (B, z_dim, n_u)
        
        if z.dim() == 1:
            u_t = u_t.squeeze(0)
            control = g_z @ u_t
        else:
            control = torch.bmm(g_z, u_t.unsqueeze(-1)).squeeze(-1)
            
        return drift + control

    def sample_z0(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_past: torch.Tensor, u_past: torch.Tensor, u_future: torch.Tensor, t_future: torch.Tensor, 
                method: str = "dopri5", deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass to compute ELBO components for causal forecasting."""
        # 1. Encode past history into initial state distribution at t=0
        mu, logvar = self.encoder(x_past, u_past)
        
        # 2. Sample initial state z0
        if deterministic:
            z0 = mu
        else:
            z0 = self.sample_z0(mu, logvar)
            
        # 3. Solve CA-ODE forward in time using future stimulus
        self.set_input(t_future, u_future)
        z_seq = odeint(self.ode_func, z0, t_future, method=method, rtol=1e-4, atol=1e-5)  # (T, B, z_dim)
        z_seq = z_seq.permute(1, 0, 2)  # (B, T, z_dim)
        
        # 4. Decode latent states to observations
        x_pred_future = self.decoder(z_seq)  # (B, T, n_x)
        
        return x_pred_future, mu, logvar

    def predict(self, x_past: torch.Tensor, u_past: torch.Tensor, u_future: torch.Tensor, t_future: torch.Tensor, 
                method: str = "dopri5") -> torch.Tensor:
        """Deterministic prediction of future using mean of z0."""
        x_pred_future, _, _ = self.forward(x_past, u_past, u_future, t_future, method=method, deterministic=True)
        return x_pred_future
