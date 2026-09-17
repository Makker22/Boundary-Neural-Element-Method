from scripts.generate_figure_data import generate


def test_generate_plot_ready_tables(tmp_path):
    manifest = generate(tmp_path / "figure_data", include_fields=False)

    assert manifest["counts"]["figure2"]["per_cell_rows"] == 2012
    assert manifest["counts"]["figure2"]["display_rows"] == 1412
    assert manifest["counts"]["figure4"]["convergence_rows"] == 600
    assert manifest["counts"]["figure4"]["selected_rows"] == 150
    assert manifest["counts"]["figure5"]["timing_rows"] == 4
    assert manifest["counts"]["figure5"]["load_response_rows"] == 20
    assert (tmp_path / "figure_data" / "figure5" / "timing.csv").is_file()
    assert (tmp_path / "figure_data" / "figure2" / "display_metrics.csv").is_file()
    assert (tmp_path / "figure_data" / "manifest.json").is_file()
