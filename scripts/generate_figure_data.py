

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
DEFAULT_OUTPUT = ROOT / "generated" / "figure_data"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in records:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    return len(records)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_figure2(output: Path) -> dict[str, int]:
    destination = output / "figure2"
    metrics = _load_json(DATA / "results" / "figure2" / "operator_metrics.json")
    _copy(DATA / "results" / "figure2" / "operator_metrics.json", destination / "operator_metrics.json")
    source = DATA / "figure_inputs" / "figure2" / "per_cell_metrics.csv"
    _copy(source, destination / "per_cell_metrics.csv")
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lower, upper = metrics["display_only"]["poisson_ratio_range"]
    display_rows = [
        row
        for row in rows
        if lower <= float(row["poisson_ratio"]) <= upper
    ]
    _write_csv(destination / "display_metrics.csv", display_rows)
    return {"per_cell_rows": len(rows), "display_rows": len(display_rows)}


def _export_figure3(output: Path, include_fields: bool) -> dict[str, int]:
    destination = output / "figure3"
    for source in sorted((DATA / "results" / "figure3").glob("*.csv")):
        _copy(source, destination / source.name)
    field_count = 0
    if include_fields:
        for source in sorted((DATA / "figure_inputs" / "figure3" / "fields").glob("*.csv")):
            _copy(source, destination / "fields" / source.name)
            field_count += 1
    return {"field_files": field_count}


def _export_figure4(output: Path, include_fields: bool) -> dict[str, int]:
    source = _load_json(DATA / "results" / "figure4" / "elastic_refinement.json")
    rows = source["rows"]
    destination = output / "figure4"
    _write_csv(destination / "convergence.csv", rows)

    target = float(source["target_displacement_relative_l2"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["id"], row["load_case"]), []).append(row)
    selected = []
    for candidates in groups.values():
        admissible = [row for row in candidates if float(row["displacement_relative_l2"]) <= target]
        if admissible:
            selected.append(min(admissible, key=lambda row: (int(row["dofs"]), int(row["elements"]))))
    selected.sort(key=lambda row: (row["id"], row["load_case"]))
    _write_csv(destination / "selected_at_accuracy.csv", selected)
    _write_json(
        destination / "metadata.json",
        {
            key: source[key]
            for key in ("schema", "operator", "operator_layout", "poisson", "loads", "reference_selection", "target_displacement_relative_l2", "summary")
        },
    )

    field_count = 0
    if include_fields:
        source_root = DATA / "figure_inputs" / "figure4"
        for field in sorted(source_root.rglob("*")):
            if field.is_file():
                _copy(field, destination / field.relative_to(source_root))
                field_count += 1
    return {"convergence_rows": len(rows), "selected_rows": len(selected), "field_files": field_count}


def _export_figure5(output: Path) -> dict[str, int]:
    equal_accuracy = _load_json(DATA / "results" / "figure5" / "equal_accuracy.json")
    load_response = _load_json(DATA / "results" / "figure5" / "load_response.json")
    destination = output / "figure5"

    timing_rows = []
    for case_name, case in equal_accuracy["cases"].items():
        for method_name, method in case["methods"].items():
            timing_rows.append(
                {
                    "case": case_name,
                    "poisson": case["poisson"],
                    "method": method_name,
                    "elements": method["elements"],
                    "dofs": method["dofs"],
                    "matrix_nnz": method["matrix_nnz"],
                    "displacement_error_percent_grid61": method["displacement_error_percent_grid61"],
                    "displacement_error_percent_grid121": method["displacement_error_percent_grid121"],
                    "median_seconds": method["compute_total_seconds"]["median"],
                    "minimum_seconds": method["compute_total_seconds"]["minimum"],
                    "maximum_seconds": method["compute_total_seconds"]["maximum"],
                    "speedup_ratio_of_medians": case["speedup_ratio_of_medians"],
                    "global_dof_reduction_percent": case["global_dof_reduction_percent"],
                }
            )
    _write_csv(destination / "timing.csv", timing_rows)
    _copy(DATA / "results" / "figure5" / "convergence.csv", destination / "convergence.csv")

    response_rows = []
    for case_name, case in load_response["cases"].items():
        for method_name, method in case["methods"].items():
            for point in method["points"]:
                response_rows.append(
                    {
                        "case": case_name,
                        "poisson": case["poisson"],
                        "method": method_name,
                        "elements": method["elements"],
                        "dofs": method["dofs"],
                        **point,
                    }
                )
    _write_csv(destination / "load_response.csv", response_rows)
    _write_json(
        destination / "metadata.json",
        {
            "geometry": equal_accuracy["geometry"],
            "accuracy": equal_accuracy["accuracy"],
            "solver": equal_accuracy["solver"],
            "load_control": load_response["load_control"],
        },
    )
    return {"timing_rows": len(timing_rows), "load_response_rows": len(response_rows)}


def generate(output: Path, include_fields: bool = True) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        "figure2": _export_figure2(output),
        "figure3": _export_figure3(output, include_fields),
        "figure4": _export_figure4(output, include_fields),
        "figure5": _export_figure5(output),
    }
    if include_fields:
        for source in sorted((DATA / "geometry").glob("*")):
            if source.is_file():
                _copy(source, output / "geometry" / source.name)

    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema": "bnem.figure_data.v1",
        "counts": counts,
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables-only", action="store_true", help="skip representative field and geometry files")
    args = parser.parse_args()
    manifest = generate(args.output, include_fields=not args.tables_only)
    print(f"wrote {len(manifest['files'])} files to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
