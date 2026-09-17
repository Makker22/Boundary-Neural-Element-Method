from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

@dataclass(frozen=True)
class RecoverySamplingSpec:
    name: str
    physics: str
    geometry_source: str
    sample_points: int
    elements: int
    boundary_samples: int
    output_fields: tuple[str, ...]
    optional_postprocess: bool = True
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RecoveryGate:
    name: str
    value: float
    tolerance: float
    passed: bool

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def relative_l2(value: np.ndarray, reference: np.ndarray, *, eps: float = 1e-12) -> float:
    a = np.asarray(value, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if a.ndim == 2:
        mask = np.all(mask, axis=1)
    return float(np.linalg.norm(a[mask] - b[mask]) / (np.linalg.norm(b[mask]) + eps))

def relative_linf(value: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(value, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if a.ndim == 2:
        mask = np.all(mask, axis=1)
        if not np.any(mask):
            return 0.0
        return float(np.max(np.linalg.norm(a[mask] - b[mask], axis=1)))
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask])))

def hotspot_topk_overlap(value: np.ndarray, reference: np.ndarray, *, fraction: float = 0.1) -> float:
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    b = np.asarray(reference, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return 0.0
    a_idx = np.flatnonzero(mask)
    k = max(1, int(np.ceil(float(fraction) * a_idx.size)))
    a_top = set(a_idx[np.argsort(a[mask])[-k:]].tolist())
    b_top = set(a_idx[np.argsort(b[mask])[-k:]].tolist())
    return float(len(a_top & b_top) / max(1, k))

def gate(name: str, value: float, tolerance: float, *, less_equal: bool = True) -> dict[str, Any]:
    passed = value <= tolerance if less_equal else value >= tolerance
    return asdict(RecoveryGate(name=name, value=float(value), tolerance=float(tolerance), passed=bool(passed)))

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def sampling_spec_dict(spec: RecoverySamplingSpec) -> dict[str, Any]:
    return asdict(spec)
