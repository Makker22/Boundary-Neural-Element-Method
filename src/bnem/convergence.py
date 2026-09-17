from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = 1
VALID_DATASET_ROLES = frozenset({"train", "validation", "test"})

@dataclass(frozen=True)
class OperatorConvergenceGates:
    matrix_relative_frobenius: float = 2.0e-3
    force_action_relative: float = 2.0e-3
    energy_action_relative: float = 2.0e-3

    def as_dict(self) -> dict[str, float]:
        return {
            "matrix_relative_frobenius": float(self.matrix_relative_frobenius),
            "force_action_relative": float(self.force_action_relative),
            "energy_action_relative": float(self.energy_action_relative),
        }

def convergence_certificate_path(sample_path: str | Path) -> Path:
    return Path(sample_path).with_suffix(".convergence.json")

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def geometry_sha256_from_npz(path: str | Path) -> str:
    keys = ("eta_vertices", "eta_edge_bulges", "eta_edge_types", "port_scheme", "bubbles_per_edge")
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in keys:
            if key not in data.files:
                continue
            value = np.asarray(data[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
    return digest.hexdigest()

def polygon_similarity_geometry_sha256(vertices: np.ndarray) -> str:

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError(f"Expected polygon vertices [N,2], got {points.shape}")
    centered = points - np.mean(points, axis=0)
    scale = max(float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))), 1.0e-30)
    normalized = centered / scale
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    count = len(points)
    canonical_rows: list[bytes] = []
    base = np.arange(count, dtype=np.int64)
    for reverse in (False, True):
        sequence = base[::-1] if reverse else base
        for shift in range(count):
            indices = np.roll(sequence, -shift)
            matrix = np.round(distances[np.ix_(indices, indices)], decimals=9)
            canonical_rows.append(matrix.astype("<f8", copy=False).tobytes())
    digest = hashlib.sha256()
    digest.update(str(count).encode("ascii"))
    digest.update(min(canonical_rows))
    return digest.hexdigest()

def polygon_similarity_geometry_sha256_from_npz(path: str | Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        if "eta_vertices" not in data.files:
            raise KeyError(f"{path} does not contain eta_vertices")
        return polygon_similarity_geometry_sha256(np.asarray(data["eta_vertices"], dtype=np.float64))

def _operator_scale(matrix: np.ndarray) -> float:
    symmetric = 0.5 * (matrix + matrix.T)
    return max(float(np.linalg.norm(symmetric, ord=2)), 1.0e-30)

def compare_boundary_operators(
    coarse: np.ndarray,
    fine: np.ndarray,
    *,
    probe_count: int = 128,
    seed: int = 20260705,
) -> dict[str, float]:
    coarse = np.asarray(coarse, dtype=np.float64)
    fine = np.asarray(fine, dtype=np.float64)
    if coarse.shape != fine.shape or coarse.ndim != 2 or coarse.shape[0] != coarse.shape[1]:
        raise ValueError(f"Boundary operators must be equal-size square matrices, got {coarse.shape} and {fine.shape}")

    delta = coarse - fine
    matrix_error = float(
        np.linalg.norm(delta, ord="fro") / max(np.linalg.norm(fine, ord="fro"), 1.0e-30)
    )
    scale = _operator_scale(fine)
    rng = np.random.default_rng(seed)
    probes = rng.normal(size=(max(int(probe_count), 1), fine.shape[0]))
    probe_norms = np.linalg.norm(probes, axis=1)
    force_errors = np.linalg.norm(probes @ delta.T, axis=1) / np.maximum(scale * probe_norms, 1.0e-30)
    energy_errors = np.abs(np.einsum("bi,ij,bj->b", probes, delta, probes)) / np.maximum(
        scale * probe_norms**2,
        1.0e-30,
    )
    return {
        "matrix_relative_frobenius": matrix_error,
        "force_action_relative": float(np.max(force_errors)),
        "energy_action_relative": float(np.max(energy_errors)),
    }

def _comparison_passed(metrics: dict[str, float], gates: OperatorConvergenceGates) -> bool:
    thresholds = gates.as_dict()
    return all(float(metrics[name]) <= threshold for name, threshold in thresholds.items())

def build_operator_convergence_certificate(
    *,
    sample_id: str,
    dataset_role: str,
    levels: Iterable[dict[str, Any]],
    selected_artifact: str | Path,
    gates: OperatorConvergenceGates | None = None,
    probe_count: int = 128,
) -> dict[str, Any]:
    if dataset_role not in VALID_DATASET_ROLES:
        raise ValueError(f"dataset_role must be one of {sorted(VALID_DATASET_ROLES)}, got {dataset_role!r}")
    gates = gates or OperatorConvergenceGates()
    level_rows = list(levels)
    if len(level_rows) < 5:
        raise ValueError("At least five mesh levels are required for a convergence certificate")

    normalized_levels: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for index, row in enumerate(level_rows):
        artifact = Path(row["artifact"]).resolve()
        matrix = np.asarray(row["matrix"], dtype=np.float64)
        if shape is None:
            shape = matrix.shape
        if matrix.shape != shape:
            raise ValueError(f"Mesh level {index} changes boundary operator shape from {shape} to {matrix.shape}")
        matrices.append(matrix)
        normalized_levels.append(
            {
                "index": index,
                "level_id": str(row.get("level_id", index)),
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "mesh": dict(row.get("mesh", {})),
            }
        )

    comparisons: list[dict[str, Any]] = []
    for index in range(1, len(matrices)):
        metrics = compare_boundary_operators(
            matrices[index - 1],
            matrices[index],
            probe_count=probe_count,
            seed=20260705 + index,
        )
        comparisons.append(
            {
                "coarse_level": normalized_levels[index - 1]["level_id"],
                "fine_level": normalized_levels[index]["level_id"],
                "metrics": metrics,
                "passed": _comparison_passed(metrics, gates),
            }
        )

    platform_comparisons = comparisons[-2:]
    passed = len(platform_comparisons) == 2 and all(row["passed"] for row in platform_comparisons)
    selected = Path(selected_artifact).resolve()
    return {
        "schema": "bnem.operator_mesh_convergence",
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "dataset_role": dataset_role,
        "quantity": "condensed_boundary_operator",
        "level_count": len(normalized_levels),
        "platform_level_count": 3,
        "gates": gates.as_dict(),
        "levels": normalized_levels,
        "comparisons": comparisons,
        "selected_artifact": {
            "path": str(selected),
            "sha256": sha256_file(selected),
            "source_level": normalized_levels[-1]["level_id"],
        },
        "passed": bool(passed),
    }

def write_convergence_certificate(certificate: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
    return out

def validate_convergence_certificate(
    sample_path: str | Path,
    *,
    certificate_path: str | Path | None = None,
    allowed_roles: Iterable[str] | None = None,
) -> dict[str, Any]:
    sample = Path(sample_path).resolve()
    cert_path = Path(certificate_path) if certificate_path is not None else convergence_certificate_path(sample)
    if not cert_path.exists():
        raise ValueError(f"Missing mesh-convergence certificate for {sample}: {cert_path}")
    certificate = json.loads(cert_path.read_text(encoding="utf-8"))
    if certificate.get("schema") != "bnem.operator_mesh_convergence":
        raise ValueError(f"Unsupported convergence certificate schema in {cert_path}")
    if int(certificate.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported convergence certificate version in {cert_path}")
    if not bool(certificate.get("passed", False)):
        raise ValueError(f"Mesh convergence did not pass for {sample}")
    if int(certificate.get("level_count", 0)) < 5:
        raise ValueError(f"Convergence certificate for {sample} has fewer than five mesh levels")
    counts = [
        int(level.get("mesh", {}).get("element_count"))
        for level in certificate.get("levels", [])
        if level.get("mesh", {}).get("element_count") is not None
    ]
    if any(coarse > fine for coarse, fine in zip(counts, counts[1:])):
        raise ValueError(f"Mesh levels are not ordered from coarse to fine for {sample}")
    comparisons = list(certificate.get("comparisons", []))
    if len(comparisons) < 2 or not all(bool(row.get("passed", False)) for row in comparisons[-2:]):
        raise ValueError(f"Last three mesh levels are not on a convergence platform for {sample}")
    role = certificate.get("dataset_role")
    if role not in VALID_DATASET_ROLES:
        raise ValueError(f"Invalid dataset role {role!r} in {cert_path}")
    if allowed_roles is not None and role not in set(allowed_roles):
        raise ValueError(f"Sample {sample} has role {role!r}, expected one of {sorted(set(allowed_roles))}")
    expected_hash = str(certificate.get("selected_artifact", {}).get("sha256", ""))
    actual_hash = sha256_file(sample)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"Certified artifact hash mismatch for {sample}")
    certificate["certificate_path"] = str(cert_path.resolve())
    return certificate

def rebuild_certified_dataset_manifest(root: str | Path) -> dict[str, Any]:

    dataset_root = Path(root).resolve()
    manifest_path = dataset_root / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"samples": []}
    )
    previous_role_by_geometry: dict[str, set[str]] = {}
    for row in previous.get("samples", []):
        previous_role_by_geometry.setdefault(str(row.get("geometry_sha256", "")), set()).add(
            str(row.get("dataset_role", ""))
        )
    candidates: list[dict[str, Any]] = []
    geometry_roles: dict[str, set[str]] = {}
    seen_ids: set[str] = set()
    for role in sorted(VALID_DATASET_ROLES):
        role_dir = dataset_root / role
        if not role_dir.exists():
            continue
        for artifact in sorted(role_dir.glob("*.npz")):
            certificate = validate_convergence_certificate(
                artifact,
                allowed_roles={role},
            )
            sample_id = str(certificate["sample_id"])
            if sample_id != artifact.stem:
                raise ValueError(
                    f"Certificate sample_id {sample_id!r} does not match artifact {artifact.stem!r}"
                )
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate certified sample_id across dataset: {sample_id}")
            seen_ids.add(sample_id)
            geometry_hash = geometry_sha256_from_npz(artifact)
            if str(certificate.get("geometry_sha256", "")) != geometry_hash:
                raise ValueError(f"Certificate geometry hash mismatch for {artifact}")
            geometry_roles.setdefault(geometry_hash, set()).add(role)
            candidates.append(
                {
                    "sample_id": sample_id,
                    "dataset_role": role,
                    "artifact": str(artifact.resolve()),
                    "certificate": str(convergence_certificate_path(artifact).resolve()),
                    "sha256": sha256_file(artifact),
                    "geometry_sha256": geometry_hash,
                }
            )
    selected_role_by_geometry: dict[str, str] = {}
    for geometry_hash, roles in geometry_roles.items():
        if len(roles) == 1:
            selected_role_by_geometry[geometry_hash] = next(iter(roles))
            continue
        previous_roles = previous_role_by_geometry.get(geometry_hash, set()) & roles
        if len(previous_roles) != 1:
            raise ValueError(
                "Geometry leakage across dataset roles has no unique trusted manifest role: "
                f"{geometry_hash} -> discovered={sorted(roles)}, previous={sorted(previous_roles)}"
            )
        selected_role_by_geometry[geometry_hash] = next(iter(previous_roles))
    rows = [
        row
        for row in candidates
        if row["dataset_role"] == selected_role_by_geometry[row["geometry_sha256"]]
    ]
    excluded = [
        {
            **row,
            "reason": "geometry_hash_is_assigned_to_another_dataset_role",
            "included_role": selected_role_by_geometry[row["geometry_sha256"]],
        }
        for row in candidates
        if row["dataset_role"] != selected_role_by_geometry[row["geometry_sha256"]]
    ]
    rows.sort(key=lambda row: (row["dataset_role"], row["sample_id"]))
    excluded.sort(key=lambda row: (row["dataset_role"], row["sample_id"]))
    manifest = {
        "schema": "bnem.certified_dataset_manifest",
        "schema_version": 1,
        "samples": rows,
        "excluded_artifacts": excluded,
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    temporary = dataset_root / ".manifest.rebuild.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest

def validate_certified_dataset_manifest(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing certified dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "bnem.certified_dataset_manifest" or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported certified dataset manifest: {manifest_path}")

    geometry_roles: dict[str, set[str]] = {}
    manifest_artifacts: set[Path] = set()
    for row in manifest.get("samples", []):
        artifact = Path(str(row["artifact"])).resolve()
        manifest_artifacts.add(artifact)
        certificate = validate_convergence_certificate(
            artifact,
            certificate_path=row.get("certificate"),
            allowed_roles={str(row["dataset_role"])},
        )
        if str(row.get("sha256", "")) != sha256_file(artifact):
            raise ValueError(f"Manifest artifact hash mismatch for {artifact}")
        geometry_hash = geometry_sha256_from_npz(artifact)
        if str(row.get("geometry_sha256", "")) != geometry_hash:
            raise ValueError(f"Manifest geometry hash mismatch for {artifact}")
        if certificate.get("geometry_sha256") != geometry_hash:
            raise ValueError(f"Certificate geometry hash mismatch for {artifact}")
        geometry_roles.setdefault(geometry_hash, set()).add(str(row["dataset_role"]))

    leaked = {key: sorted(roles) for key, roles in geometry_roles.items() if len(roles) > 1}
    if leaked:
        raise ValueError(f"Geometry leakage across dataset roles: {leaked}")
    excluded_artifacts: set[Path] = set()
    for row in manifest.get("excluded_artifacts", []):
        artifact = Path(str(row["artifact"])).resolve()
        excluded_artifacts.add(artifact)
        certificate = validate_convergence_certificate(
            artifact,
            certificate_path=row.get("certificate"),
            allowed_roles={str(row["dataset_role"])},
        )
        if str(row.get("sha256", "")) != sha256_file(artifact):
            raise ValueError(f"Excluded manifest artifact hash mismatch for {artifact}")
        geometry_hash = geometry_sha256_from_npz(artifact)
        if str(row.get("geometry_sha256", "")) != geometry_hash:
            raise ValueError(f"Excluded manifest geometry hash mismatch for {artifact}")
        if certificate.get("geometry_sha256") != geometry_hash:
            raise ValueError(f"Excluded certificate geometry hash mismatch for {artifact}")
        if not str(row.get("reason", "")):
            raise ValueError(f"Excluded manifest artifact lacks a reason: {artifact}")
    discovered_artifacts = {
        path.resolve()
        for role in VALID_DATASET_ROLES
        for path in (dataset_root / role).glob("*.npz")
        if (dataset_root / role).exists()
    }
    indexed_artifacts = manifest_artifacts | excluded_artifacts
    overlap = manifest_artifacts & excluded_artifacts
    if overlap:
        raise ValueError(f"Artifacts are both included and excluded in manifest: {sorted(map(str, overlap))[:8]}")
    missing = sorted(str(path) for path in discovered_artifacts - indexed_artifacts)
    stale = sorted(str(path) for path in indexed_artifacts - discovered_artifacts)
    if missing or stale:
        raise ValueError(
            "Certified dataset manifest does not exactly cover role artifacts: "
            f"missing={missing[:8]}, stale={stale[:8]}"
        )
    manifest["manifest_path"] = str(manifest_path)
    return manifest

def certified_manifest_artifact_paths(root: str | Path) -> set[Path] | None:

    manifest_path = Path(root).resolve() / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {Path(str(row["artifact"])).resolve() for row in manifest.get("samples", [])}

def exclude_replay_geometry_leakage(
    replay_samples: list[Any], heldout_samples: list[Any]
) -> tuple[list[Any], list[dict[str, Any]]]:






    heldout_by_hash: dict[str, list[str]] = {}
    for sample in heldout_samples:
        path = Path(sample.path).resolve()
        geometry_hash = polygon_similarity_geometry_sha256_from_npz(path)
        heldout_by_hash.setdefault(geometry_hash, []).append(str(path))
    kept: list[Any] = []
    excluded: list[dict[str, Any]] = []
    for sample in replay_samples:
        path = Path(sample.path).resolve()
        geometry_hash = polygon_similarity_geometry_sha256_from_npz(path)
        conflicts = heldout_by_hash.get(geometry_hash)
        if conflicts:
            excluded.append(
                {
                    "replay_artifact": str(path),
                    "geometry_sha256": geometry_hash,
                    "heldout_artifacts": sorted(conflicts),
                    "reason": "replay_train_geometry_occurs_in_primary_validation_or_test",
                }
            )
        else:
            kept.append(sample)
    return kept, excluded
