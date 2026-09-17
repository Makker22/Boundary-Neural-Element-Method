

from __future__ import annotations

import torch
from torch import nn

class AffineComplementCholeskyNet(nn.Module):






    def __init__(
        self,
        *,
        feature_dim: int = 22,
        port_dofs: int = 16,
        complement_rank: int = 10,
        hidden_dim: int = 96,
        hidden_layers: int = 3,
        feature_mean: torch.Tensor | None = None,
        feature_scale: torch.Tensor | None = None,
        base_factor: torch.Tensor | None = None,
        correction_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or port_dofs <= 0 or complement_rank <= 0:
            raise ValueError("feature_dim, port_dofs, and complement_rank must be positive")
        if hidden_dim <= 0 or hidden_layers < 1:
            raise ValueError("hidden_dim must be positive and hidden_layers at least one")
        self.feature_dim = int(feature_dim)
        self.port_dofs = int(port_dofs)
        self.complement_rank = int(complement_rank)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.correction_scale = float(correction_scale)
        mean = torch.zeros(self.feature_dim) if feature_mean is None else feature_mean.detach().clone()
        scale = torch.ones(self.feature_dim) if feature_scale is None else feature_scale.detach().clone()
        base = (
            torch.zeros(self.port_dofs, self.complement_rank)
            if base_factor is None
            else base_factor.detach().clone()
        )
        if mean.shape != (self.feature_dim,) or scale.shape != (self.feature_dim,):
            raise ValueError("feature_mean and feature_scale must match feature_dim")
        if base.shape != (self.port_dofs, self.complement_rank):
            raise ValueError("base_factor has incompatible shape")
        if torch.any(scale <= 0):
            raise ValueError("feature_scale must be positive")
        self.register_buffer("feature_mean", mean.to(dtype=torch.float32))
        self.register_buffer("feature_scale", scale.to(dtype=torch.float32))
        self.register_buffer("base_factor", base.to(dtype=torch.float32))

        layers: list[nn.Module] = []
        width = self.feature_dim
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(width, self.hidden_dim), nn.SiLU()))
            width = self.hidden_dim
        output = nn.Linear(width, self.port_dofs * self.complement_rank)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def factor(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features last dimension must be {self.feature_dim}, got {features.shape}"
            )
        normalized = (features - self.feature_mean) / self.feature_scale
        correction = self.network(normalized).reshape(
            *features.shape[:-1], self.port_dofs, self.complement_rank
        )
        return self.base_factor + self.correction_scale * correction

    def forward(
        self,
        features: torch.Tensor,
        affine_operator: torch.Tensor,
        complement_projector: torch.Tensor,
    ) -> torch.Tensor:
        factor = self.factor(features)
        projected = complement_projector.transpose(-1, -2) @ factor
        stiffness = affine_operator + projected @ projected.transpose(-1, -2)
        return 0.5 * (stiffness + stiffness.transpose(-1, -2))

__all__ = ["AffineComplementCholeskyNet"]
