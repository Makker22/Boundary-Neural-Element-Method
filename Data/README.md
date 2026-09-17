# BNEM data release

This directory contains the compact numerical release for the boundary neural
element method (BNEM).

## Contents

- `training/heat.npz`: 4,000 local heat-operator samples.
- `training/elasticity_compact_labels.npz`: 1,000 quadrilateral geometries at
  ten Poisson ratios, including 0.49.
- `results/figure2/`: local-operator accuracy, structure checks and the matched
  Poisson-ratio range used for display.
- `results/figure3/`: heat generalization, ablation and convergence tables.
- `results/figure4/`: four-level elasticity refinement results for 150 load
  cases.
- `results/figure5/`: equal-accuracy timing, convergence and load-response
  records for the heterogeneous engineering example.
- `figure_inputs/`: per-sample metrics, material parameters and representative
  field samples for visualization.
- `geometry/`: released BNEM mesh banks and display selections.
- `manifest.json`: SHA-256 checksum and size of every released data and model
  artifact.

Run the following commands from the repository root:

```bash
python scripts/verify_reported_results.py
python scripts/verify_integrity.py
python scripts/generate_figure_data.py
```

The compact release excludes multi-gigabyte converged FEM mesh archives and
full-resolution solution histories. The packaged tables were computed from
those archives and preserve the convergence gates, solver settings, numerical
values and file hashes needed to verify the reported results.

Data files are licensed under CC BY 4.0. See `LICENSE`.
