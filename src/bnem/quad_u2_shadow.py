

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from src.bnem.quad_affine_model import AffineComplementCholeskyNet
from src.bnem.quad_affine_schur import (
    add_psd_affine_complement_stabilization,
    mass_affine_complement_projector,
    quad_affine_consistency_operator,
)
from src.bnem.quad_canonical import CanonicalQuad, canonical_quad_with_bubbles
from src.bnem.quad_hierarchical import extract_u1_hierarchical_blocks

@dataclass(frozen=True)
class U2ShadowBatch:
    coupling_blocks: np.ndarray
    omitted_stiffness_blocks: np.ndarray
    total_seconds: float
    analytic_seconds: float
    inference_seconds: float
    restore_seconds: float

@dataclass(frozen=True)
class FixedQuadU2ShadowOperator:
    model: AffineComplementCholeskyNet
    payload: dict
    device: str
    checkpoint_load_seconds: float

    @property
    def bubbles_per_edge(self) -> int:
        return self.model.port_dofs // 8 - 1

    def predict_blocks_from_canonical(
        self,
        canonical: Sequence[CanonicalQuad],
        canonical_features: np.ndarray,
        *,
        poisson: float | Sequence[float],
        batch_size: int = 4096,
    ) -> U2ShadowBatch:
        started = time.perf_counter()
        count = len(canonical)
        features = np.asarray(canonical_features, dtype=np.float32)
        if features.shape != (count, 22):
            raise ValueError("canonical_features must have shape [cell_count,22]")
        nu = np.asarray(poisson, dtype=np.float64)
        if nu.ndim == 0:
            nu = np.full(count, float(nu))
        if nu.shape != (count,):
            raise ValueError("poisson must be scalar or one value per cell")

        analytic_start = time.perf_counter()
        bubbles = self.bubbles_per_edge
        if bubbles < 2:
            raise ValueError("shadow checkpoint must enrich U1 with at least one mode")
        u2_items = [canonical_quad_with_bubbles(item, bubbles) for item in canonical]
        affine: list[np.ndarray] = []
        projector: list[np.ndarray] = []
        for item, value in zip(u2_items, nu):
            consistency, _trace, _force = quad_affine_consistency_operator(
                item.vertices, poisson=float(value), bubbles_per_edge=bubbles
            )
            complement, _mass, _trace = mass_affine_complement_projector(
                item.vertices, bubbles_per_edge=bubbles
            )
            affine.append(consistency)
            projector.append(complement)
        analytic_seconds = time.perf_counter() - analytic_start

        inference_start = time.perf_counter()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, count, batch_size):
                stop = min(count, start + batch_size)
                chunks.append(
                    np.asarray(
                        self.model.factor(
                            torch.as_tensor(features[start:stop], device=self.device)
                        ).detach().cpu(),
                        dtype=np.float64,
                    )
                )
        factors = np.concatenate(chunks)
        inference_seconds = time.perf_counter() - inference_start

        restore_start = time.perf_counter()
        enrichment_dofs = 8 * (bubbles - 1)
        coupling = np.empty((count, enrichment_dofs, 16), dtype=np.float64)
        omitted = np.empty((count, enrichment_dofs, enrichment_dofs), dtype=np.float64)
        for index in range(count):
            matrix = add_psd_affine_complement_stabilization(
                affine[index], projector[index], factors[index]
            )
            native = u2_items[index].stiffness_to_input(matrix)
            _k11, coupling[index], omitted[index] = extract_u1_hierarchical_blocks(native)
        restore_seconds = time.perf_counter() - restore_start
        return U2ShadowBatch(
            coupling_blocks=coupling,
            omitted_stiffness_blocks=omitted,
            total_seconds=float(time.perf_counter() - started),
            analytic_seconds=float(analytic_seconds),
            inference_seconds=float(inference_seconds),
            restore_seconds=float(restore_seconds),
        )

def load_fixed_quad_u2_shadow_operator(
    checkpoint: str | Path, *, device: str = "cpu"
) -> FixedQuadU2ShadowOperator:
    started = time.perf_counter()
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if payload.get("model_kind") != "AffineComplementCholeskyNet":
        raise ValueError("checkpoint is not an AffineComplementCholeskyNet")
    model = AffineComplementCholeskyNet(**dict(payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return FixedQuadU2ShadowOperator(
        model=model,
        payload=payload,
        device=str(device),
        checkpoint_load_seconds=float(time.perf_counter() - started),
    )

__all__ = [
    "FixedQuadU2ShadowOperator",
    "U2ShadowBatch",
    "load_fixed_quad_u2_shadow_operator",
]
