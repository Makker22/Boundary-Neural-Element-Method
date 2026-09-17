from scripts.verify_reported_results import compute_report, verify


def test_reported_results_match_release_data() -> None:
    assert verify(compute_report()) == []
