from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "comparative_methods"
    / "performance_analysis"
    / "p0_experiment_manifest.yaml"
)


def test_p0_manifest_keeps_protected_results_descriptive_only() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema"] == "comparator_performance_degradation_p0_v1"
    protected = payload["data_governance"]["protected_aggregate"]
    assert protected["permitted_use"] == "frozen_descriptive_global_surface_only"
    assert {
        "hyperparameter_selection",
        "window_or_layer_selection",
        "checkpoint_selection",
        "repeated_test_inference",
    }.issubset(protected["forbidden_use"])
    assert payload["data_governance"]["mechanism_development"][
        "protected_predictions_or_labels"
    ] == "prohibited"


def test_p0_manifest_requires_fail_closed_and_figure_provenance() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    requirements = payload["common_requirements"]
    assert requirements["fail_closed"]["simulated_or_interpolated_results"] == "prohibited"
    assert requirements["fail_closed"]["missing_required_metadata"].startswith(
        "capability_report"
    )
    figure_requirements = set(requirements["plots"]["require"])
    assert {"source_data", "transformation_code", "figure_manifest"}.issubset(
        figure_requirements
    )
    assert "alt_text" not in figure_requirements
    assert requirements["plots"]["target_journal"] == "unresolved"
