"""Conditional Flow Matching for synthesis route generation.

Based on "Flow Matching for Generative Modeling" (Lipman et al. 2022)
and "Improving and Generalizing Flow-Based Generative Models" (ICML 2023).

The key idea:
1. Define a probability path from noise N(0,I) to data distribution
2. Learn a vector field v(x,t,condition) that generates this flow
3. Sample by integrating dx/dt = v(x,t,condition) from t=0 to t=1
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (like in diffusion models)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class MLPBlock(nn.Module):
    """MLP block with residual connection."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class VectorFieldNetwork(nn.Module):
    """Neural network that predicts the vector field v(x, t, condition).

    Architecture:
    - Input: x (route embedding), t (time), condition (properties), reaction (optional)
    - Output: v (vector field, same dimension as x)
    """

    def __init__(
        self,
        input_dim: int = 4096,  # Route embedding dimension (2 * 2048)
        cond_dim: int = 2,  # (activity, qed)
        hidden_dim: int = 1024,
        num_layers: int = 6,
        time_embed_dim: int = 128,
        dropout: float = 0.1,
        num_reactions: int = 124,  # Total number of reaction types
        reaction_embed_dim: int = 64,  # Dimension of reaction embedding
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_reactions = num_reactions
        self.reaction_embed_dim = reaction_embed_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        # Reaction embedding (for reaction-conditioned generation)
        self.reaction_embed = nn.Embedding(num_reactions, reaction_embed_dim)

        # Input projection (includes reaction embedding dimension)
        self.input_proj = nn.Linear(
            input_dim + time_embed_dim + hidden_dim // 4 + reaction_embed_dim,
            hidden_dim,
        )

        # Main network
        self.layers = nn.ModuleList([
            MLPBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, input_dim) - noisy route embedding
        t: torch.Tensor,  # (batch,) - time in [0, 1]
        cond: torch.Tensor,  # (batch, cond_dim) - target properties
        reaction_idx: Optional[torch.Tensor] = None,  # (batch,) - reaction indices
    ) -> torch.Tensor:
        """Predict vector field at (x, t) conditioned on properties and reaction.

        Args:
            x: Noisy route embedding
            t: Time step in [0, 1]
            cond: Target properties (activity, qed)
            reaction_idx: Optional reaction indices for conditioning

        Returns:
            Predicted vector field
        """
        # Embed time and condition
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(cond)

        # Embed reaction (use zeros if not provided)
        if reaction_idx is not None:
            r_emb = self.reaction_embed(reaction_idx)
        else:
            # Use learned average embedding for generation without reaction conditioning
            r_emb = torch.zeros(x.size(0), self.reaction_embed_dim, device=x.device)

        # Concatenate and project
        h = torch.cat([x, t_emb, r_emb, c_emb], dim=-1)
        h = self.input_proj(h)

        # Process through layers
        for layer in self.layers:
            h = layer(h)

        # Output
        v = self.output_proj(h)
        return v


class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching model for synthesis route generation.

    Learns to generate synthesis routes (building block pairs) conditioned
    on target molecular properties (activity, QED) and optionally reaction type.

    Training:
        1. Sample x_1 from data (route embeddings)
        2. Sample x_0 from N(0, sigma*I)
        3. Sample t ~ U(0, 1)
        4. Compute x_t = (1-t)*x_0 + t*x_1 (linear interpolation)
        5. Target vector field: u_t = x_1 - x_0
        6. Loss: ||v(x_t, t, cond, reaction) - u_t||^2

    Generation:
        1. Sample x_0 ~ N(0, sigma*I)
        2. Integrate dx/dt = v(x, t, cond, reaction) from t=0 to t=1
        3. Return x_1 as generated route embedding
    """

    def __init__(
        self,
        input_dim: int = 4096,
        cond_dim: int = 2,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        sigma: float = 0.001,  # Noise scale for source distribution
        num_reactions: int = 124,
        reaction_embed_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.sigma = sigma
        self.num_reactions = num_reactions
        self.reaction_embed_dim = reaction_embed_dim

        self.vector_field = VectorFieldNetwork(
            input_dim=input_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_reactions=num_reactions,
            reaction_embed_dim=reaction_embed_dim,
        )

    def forward(
        self,
        x_1: torch.Tensor,  # (batch, input_dim) - data samples
        cond: torch.Tensor,  # (batch, cond_dim) - conditions
        reaction_idx: Optional[torch.Tensor] = None,  # (batch,) - reaction indices
    ) -> tuple[torch.Tensor, dict]:
        """Compute flow matching loss.

        Args:
            x_1: Data samples (route embeddings)
            cond: Target properties (activity, qed)
            reaction_idx: Optional reaction indices for conditioning

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample source noise
        x_0 = torch.randn_like(x_1) * self.sigma

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Linear interpolation (optimal transport path)
        x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1

        # Target vector field (derivative of x_t w.r.t. t)
        u_t = x_1 - x_0

        # Predicted vector field (with optional reaction conditioning)
        v_t = self.vector_field(x_t, t, cond, reaction_idx=reaction_idx)

        # Flow matching loss (MSE)
        loss = ((v_t - u_t) ** 2).mean()

        metrics = {
            "loss": loss.item(),
            "v_norm": v_t.norm(dim=-1).mean().item(),
            "u_norm": u_t.norm(dim=-1).mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,  # (batch, cond_dim) or (cond_dim,)
        num_samples: int = 1,
        num_steps: int = 50,
        return_trajectory: bool = False,
        reaction_idx: Optional[torch.Tensor] = None,  # (batch,) or scalar
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate samples by integrating the learned vector field.

        Uses Euler method for ODE integration:
            x_{t+dt} = x_t + dt * v(x_t, t, cond, reaction)

        Args:
            cond: Target properties (activity, qed)
            num_samples: Number of samples per condition
            num_steps: Number of integration steps
            return_trajectory: Whether to return intermediate states
            reaction_idx: Optional reaction indices for conditioning

        Returns:
            Generated route embeddings of shape (batch * num_samples, input_dim)
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)

        batch_size = cond.shape[0]
        device = cond.device

        # Expand condition for multiple samples
        if num_samples > 1:
            cond = cond.repeat_interleave(num_samples, dim=0)

        # Expand reaction indices if provided
        if reaction_idx is not None:
            if reaction_idx.dim() == 0:
                reaction_idx = reaction_idx.unsqueeze(0)
            if num_samples > 1:
                reaction_idx = reaction_idx.repeat_interleave(num_samples, dim=0)

        total_samples = batch_size * num_samples

        # Start from noise
        x = torch.randn(total_samples, self.input_dim, device=device) * self.sigma

        trajectory = [x.clone()] if return_trajectory else None

        # Euler integration
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((total_samples,), step * dt, device=device)
            v = self.vector_field(x, t, cond, reaction_idx=reaction_idx)
            x = x + dt * v

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_adaptive(
        self,
        cond: torch.Tensor,
        num_samples: int = 1,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        reaction_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using adaptive ODE solver (RK45).

        More accurate but slower than fixed-step Euler.

        Args:
            cond: Target properties (activity, qed)
            num_samples: Number of samples per condition
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            reaction_idx: Optional reaction indices for conditioning

        Returns:
            Generated route embeddings
        """
        from scipy.integrate import solve_ivp

        if cond.dim() == 1:
            cond = cond.unsqueeze(0)

        batch_size = cond.shape[0]
        device = cond.device

        if num_samples > 1:
            cond = cond.repeat_interleave(num_samples, dim=0)

        # Expand reaction indices if provided
        if reaction_idx is not None:
            if reaction_idx.dim() == 0:
                reaction_idx = reaction_idx.unsqueeze(0)
            if num_samples > 1:
                reaction_idx = reaction_idx.repeat_interleave(num_samples, dim=0)

        total_samples = batch_size * num_samples

        # Start from noise
        x0 = torch.randn(total_samples, self.input_dim, device=device) * self.sigma
        x0_np = x0.cpu().numpy().flatten()

        # Define ODE function
        def ode_func(t, x_flat):
            x = torch.tensor(x_flat.reshape(total_samples, -1), device=device, dtype=torch.float32)
            t_tensor = torch.full((total_samples,), t, device=device)
            v = self.vector_field(x, t_tensor, cond, reaction_idx=reaction_idx)
            return v.cpu().numpy().flatten()

        # Solve ODE
        solution = solve_ivp(
            ode_func,
            t_span=(0, 1),
            y0=x0_np,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )

        x1 = torch.tensor(
            solution.y[:, -1].reshape(total_samples, -1),
            device=device,
            dtype=torch.float32,
        )
        return x1
