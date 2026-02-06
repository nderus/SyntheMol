"""Conditional Masked Autoregressive Flow for molecule fingerprint generation."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """Linear layer with autoregressive masking."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        cond_features: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cond_features = cond_features

        # Main weight matrix (masked)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer("mask", mask)

        # Conditioning weight (unmasked)
        if cond_features > 0:
            self.cond_weight = nn.Parameter(torch.empty(out_features, cond_features))
        else:
            self.register_parameter("cond_weight", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_features + self.cond_features
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        if self.cond_weight is not None:
            nn.init.kaiming_uniform_(self.cond_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = F.linear(x, self.weight * self.mask, self.bias)
        if cond is not None and self.cond_weight is not None:
            out = out + F.linear(cond, self.cond_weight)
        return out


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.

    Outputs parameters (shift, log_scale) for autoregressive transformation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        cond_dim: int = 0,
        num_masks: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.cond_dim = cond_dim

        # Create connectivity pattern for autoregressive property
        self._create_masks(num_masks)

        # Build layers
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(
                MaskedLinear(prev_dim, hidden_dim, self.masks[i], cond_dim)
            )
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer: 2 outputs per input dimension (shift and log_scale)
        layers.append(
            MaskedLinear(prev_dim, input_dim * 2, self.masks[-1], cond_dim)
        )

        self.layers = nn.ModuleList(layers)

    def _create_masks(self, num_masks: int):
        """Create autoregressive masks for all layers."""
        # Assign random degrees to each hidden unit
        rng = torch.Generator().manual_seed(42)

        # Input degrees: 0 to D-1 (each input depends only on previous inputs)
        input_degrees = torch.arange(self.input_dim)

        self.masks = []
        prev_degrees = input_degrees

        for hidden_dim in self.hidden_dims:
            # Hidden units can depend on inputs with degree < their degree
            hidden_degrees = torch.randint(
                0, self.input_dim - 1, (hidden_dim,), generator=rng
            )
            # Mask: hidden unit h can connect to input/prev hidden p if degree[p] <= degree[h]
            mask = (prev_degrees.unsqueeze(0) <= hidden_degrees.unsqueeze(1)).float()
            self.masks.append(mask)
            prev_degrees = hidden_degrees

        # Output layer mask: output i can connect to hidden h if degree[h] < i
        # For shift and log_scale, we have 2*D outputs
        output_degrees = torch.arange(self.input_dim).repeat_interleave(2)
        output_mask = (prev_degrees.unsqueeze(0) < output_degrees.unsqueeze(1)).float()
        self.masks.append(output_mask)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning shift and log_scale."""
        h = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MaskedLinear):
                h = layer(h, cond)
            else:
                h = layer(h)

        # Split into shift and log_scale
        shift, log_scale = h.chunk(2, dim=-1)
        # Clamp log_scale for numerical stability
        log_scale = torch.clamp(log_scale, min=-5, max=3)

        return shift, log_scale


class MAFLayer(nn.Module):
    """Single layer of Masked Autoregressive Flow."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        cond_dim: int = 0,
    ):
        super().__init__()
        self.made = MADE(input_dim, hidden_dims, cond_dim)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform x to z, return z and log_det_jacobian."""
        shift, log_scale = self.made(x, cond)
        z = (x - shift) * torch.exp(-log_scale)
        log_det = -log_scale.sum(dim=-1)
        return z, log_det

    def inverse(
        self, z: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform z to x (generation direction)."""
        x = torch.zeros_like(z)
        for i in range(z.shape[-1]):
            shift, log_scale = self.made(x, cond)
            x[..., i] = z[..., i] * torch.exp(log_scale[..., i]) + shift[..., i]
        return x


class ConditionalMAF(nn.Module):
    """Conditional Masked Autoregressive Flow for fingerprint generation.

    This flow models p(fingerprint | target_scores) where:
    - fingerprint: 2048-dim (or reduced) Morgan fingerprint
    - target_scores: 2-dim (activity, qed) target properties

    The flow learns to transform a simple base distribution (Gaussian)
    to match the distribution of fingerprints conditioned on target scores.
    """

    def __init__(
        self,
        input_dim: int = 128,  # Reduced fingerprint dimension
        cond_dim: int = 2,  # (activity, qed)
        hidden_dims: list[int] = None,
        num_layers: int = 8,
        cond_encoder_dims: list[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        if hidden_dims is None:
            hidden_dims = [256, 256]
        if cond_encoder_dims is None:
            cond_encoder_dims = [64, 64]

        # Condition encoder: transform (activity, qed) to latent representation
        encoder_layers = []
        prev_dim = cond_dim
        for dim in cond_encoder_dims:
            encoder_layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        self.cond_encoder = nn.Sequential(*encoder_layers)
        encoded_cond_dim = cond_encoder_dims[-1] if cond_encoder_dims else cond_dim

        # MAF layers with alternating dimension ordering
        self.layers = nn.ModuleList()
        self.permutations = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(MAFLayer(input_dim, hidden_dims, encoded_cond_dim))
            # Reverse permutation for every other layer
            if i < num_layers - 1:
                perm = torch.arange(input_dim - 1, -1, -1) if i % 2 == 0 else torch.arange(input_dim)
                self.permutations.append(Permutation(perm))

        # For log prob computation
        self.register_buffer("base_loc", torch.zeros(input_dim))
        self.register_buffer("base_scale", torch.ones(input_dim))

    def encode_condition(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode conditioning variables."""
        return self.cond_encoder(cond)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform x to z, return z and log probability.

        Args:
            x: Fingerprints of shape (batch, input_dim)
            cond: Conditions of shape (batch, cond_dim) - (activity, qed)

        Returns:
            z: Latent codes of shape (batch, input_dim)
            log_prob: Log probability of shape (batch,)
        """
        encoded_cond = self.encode_condition(cond)

        z = x
        sum_log_det = torch.zeros(x.shape[0], device=x.device)

        for i, layer in enumerate(self.layers):
            z, log_det = layer(z, encoded_cond)
            sum_log_det = sum_log_det + log_det
            if i < len(self.permutations):
                z = self.permutations[i](z)

        # Compute log probability
        log_prob_z = -0.5 * (
            self.input_dim * math.log(2 * math.pi)
            + (z ** 2).sum(dim=-1)
        )
        log_prob = log_prob_z + sum_log_det

        return z, log_prob

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute log probability of x given condition."""
        _, log_prob = self.forward(x, cond)
        return log_prob

    def sample(
        self,
        cond: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample fingerprints given conditions.

        Args:
            cond: Conditions of shape (batch, cond_dim) or (cond_dim,)
            num_samples: Number of samples per condition
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            Fingerprints of shape (batch * num_samples, input_dim)
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)

        batch_size = cond.shape[0]
        encoded_cond = self.encode_condition(cond)

        # Expand for multiple samples per condition
        if num_samples > 1:
            encoded_cond = encoded_cond.repeat_interleave(num_samples, dim=0)

        # Sample from base distribution
        z = torch.randn(
            batch_size * num_samples,
            self.input_dim,
            device=cond.device,
        ) * temperature

        # Inverse transform
        x = self.inverse(z, encoded_cond)

        return x

    def inverse(
        self, z: torch.Tensor, encoded_cond: torch.Tensor
    ) -> torch.Tensor:
        """Inverse transform: z -> x (generation direction)."""
        x = z

        # Apply layers in reverse order
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.permutations):
                x = self.permutations[i].inverse(x)
            x = self.layers[i].inverse(x, encoded_cond)

        return x


class Permutation(nn.Module):
    """Fixed permutation layer."""

    def __init__(self, perm: torch.Tensor):
        super().__init__()
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.inv_perm]
