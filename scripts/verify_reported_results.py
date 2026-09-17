

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(array)), "maximum": float(np.max(array))}


def compute_report() -> dict[str, Any]:
    figure2 = read_json(DATA / "results" / "figure2" / "operator_metrics.json")

    heat_rows = read_csv(DATA / "results" / "figure3" / "heat_summary.csv")
    ablation_rows = read_csv(DATA / "results" / "figure3" / "ablation.csv")
    heat_field = [100.0 * float(row["field_relative_l2"]) for row in heat_rows]
    heat_flux = [100.0 * float(row["flux_relative_error"]) for row in heat_rows]

    full = [row for row in ablation_rows if row["method"].startswith("Full")]
    no_projection = [row for row in ablation_rows if row["method"] == "No projection"]
    analytic = [row for row in ablation_rows if row["method"] == "Analytical only"]
    full_error = [100.0 * float(row["field_relative_l2"]) for row in full]
    no_projection_error = [
        100.0 * float(row["field_relative_l2"]) for row in no_projection
    ]
    analytic_over_100 = sum(
        bool(row["field_relative_l2"]) and float(row["field_relative_l2"]) > 1.0
        for row in analytic
    )
    analytic_without_finite_field = sum(not row["field_relative_l2"] for row in analytic)

    refinement = read_json(
        DATA / "results" / "figure4" / "elastic_refinement.json"
    )
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in refinement["rows"]:
        by_case.setdefault((row["id"], row["load_case"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for rows in by_case.values():
        ordered = sorted(rows, key=lambda row: float(row["mesh_size"]), reverse=True)
        selected.append(
            next(row for row in ordered if float(row["displacement_relative_l2"]) <= 0.01)
        )

    figure5 = read_json(DATA / "results" / "figure5" / "equal_accuracy.json")
    convergence = read_csv(DATA / "results" / "figure5" / "convergence.csv")
    load_response = read_json(DATA / "results" / "figure5" / "load_response.json")

    figure5_report: dict[str, Any] = {}
    for case_name, case in figure5["cases"].items():
        bnem = case["methods"]["bnem"]
        fem = case["methods"]["fem"]
        figure5_report[case_name] = {
            "poisson": float(case["poisson"]),
            "bnem_error_percent": float(bnem["displacement_error_percent_grid121"]),
            "fem_error_percent": float(fem["displacement_error_percent_grid121"]),
            "bnem_dofs": int(bnem["dofs"]),
            "fem_dofs": int(fem["dofs"]),
            "bnem_seconds": float(bnem["compute_total_seconds"]["median"]),
            "fem_seconds": float(fem["compute_total_seconds"]["median"]),
            "speedup": float(case["speedup_ratio_of_medians"]),
            "dof_reduction_percent": float(case["global_dof_reduction_percent"]),
        }

    return {
        "figure2": figure2["datasets"],
        "figure3": {
            "geometries": len(heat_rows),
            "temperature_error_percent": summary(heat_field),
            "integrated_flux_error_percent": summary(heat_flux),
            "full_model_temperature_error_percent": summary(full_error),
            "no_projection_temperature_error_percent": summary(no_projection_error),
            "analytical_only_over_100_percent": analytic_over_100,
            "analytical_only_without_finite_field": analytic_without_finite_field,
        },
        "figure4": {
            "load_cases": len(selected),
            "displacement_error_percent": summary(
                [100.0 * float(row["displacement_relative_l2"]) for row in selected]
            ),
            "energy_or_reaction_error_percent": summary(
                [
                    100.0 * float(row["energy_or_reaction_relative_error"])
                    for row in selected
                ]
            ),
            "stress_recovery_error_percent": summary(
                [100.0 * float(row["mises_relative_l2"]) for row in selected]
            ),
            "high_stress_iou_percent_median": float(
                100.0 * np.median([float(row["high_stress_iou"]) for row in selected])
            ),
        },
        "figure5": figure5_report,
        "figure5_convergence_points": len(convergence),
        "figure5_load_solves": sum(
            len(method["points"])
            for case in load_response["cases"].values()
            for method in case["methods"].values()
        ),
    }


def verify(report: dict[str, Any]) -> list[str]:
    expected = {
        ("figure3", "geometries"): (50.0, 0.0),
        ("figure3", "temperature_error_percent", "median"): (0.1429868173, 1.0e-9),
        ("figure3", "temperature_error_percent", "maximum"): (0.4524586147, 1.0e-9),
        ("figure3", "integrated_flux_error_percent", "median"): (0.4635248368, 1.0e-9),
        ("figure3", "integrated_flux_error_percent", "maximum"): (2.9394156030, 1.0e-9),
        ("figure3", "full_model_temperature_error_percent", "median"): (0.1362973627, 1.0e-9),
        ("figure3", "no_projection_temperature_error_percent", "median"): (8.1745007858, 1.0e-9),
        ("figure3", "analytical_only_over_100_percent"): (47.0, 0.0),
        ("figure3", "analytical_only_without_finite_field"): (3.0, 0.0),
        ("figure4", "load_cases"): (150.0, 0.0),
        ("figure4", "displacement_error_percent", "median"): (0.6711857511, 1.0e-9),
        ("figure4", "displacement_error_percent", "maximum"): (0.9930095150, 1.0e-9),
        ("figure4", "energy_or_reaction_error_percent", "median"): (2.2578620934, 1.0e-9),
        ("figure4", "energy_or_reaction_error_percent", "maximum"): (7.1852405751, 1.0e-9),
        ("figure4", "stress_recovery_error_percent", "median"): (10.4450170010, 1.0e-9),
        ("figure4", "stress_recovery_error_percent", "maximum"): (20.8663258528, 1.0e-9),
        ("figure4", "high_stress_iou_percent_median"): (80.0, 1.0e-12),
        ("figure5", "compressible", "speedup"): (9.4424770745, 1.0e-9),
        ("figure5", "compressible", "dof_reduction_percent"): (78.0835947691, 1.0e-9),
        ("figure5", "near_incompressible", "speedup"): (53.1511250111, 1.0e-9),
        ("figure5", "near_incompressible", "dof_reduction_percent"): (91.7456485552, 1.0e-9),
        ("figure5_convergence_points",): (15.0, 0.0),
        ("figure5_load_solves",): (20.0, 0.0),
    }
    failures: list[str] = []
    for path, (target, tolerance) in expected.items():
        value: Any = report
        for key in path:
            value = value[key]
        if abs(float(value) - target) > tolerance:
            failures.append(f"{'.'.join(path)}: {value} != {target}")
    for case_name in ("compressible", "near_incompressible"):
        case = report["figure5"][case_name]
        for method in ("bnem", "fem"):
            if case[f"{method}_error_percent"] > 0.2:
                failures.append(f"figure5.{case_name}.{method} exceeds 0.2%")
    return failures


def main() -> int:
    report = compute_report()
    failures = verify(report)
    output = {"verified": not failures, "failures": failures, "results": report}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
