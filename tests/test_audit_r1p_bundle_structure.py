import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/scripts/audit_r1p_bundle_structure.py"
SPEC = importlib.util.spec_from_file_location("r1p_structural_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)

BUNDLE = ROOT / "data/cache/shared_driver_r1_v1/r1_p_development_v1"
CONFIG = (
    ROOT
    / "experiments/configs/physiology_semantic_tokenizer/"
    "r1p_population_frozen_teacher.yaml"
)


def test_formal_r1p_bundle_passes_structure_without_effect_evaluation():
    result = audit_module.audit_bundle(
        BUNDLE, CONFIG, verify_dataset_reader=False
    )
    assert result["status"] == "passed"
    assert result["validation_physiology_effects_evaluated"] is False
    assert (
        result["qualification_registry_state_at_audit"]
        == "frozen_before_audit"
    )
    assert result["qualification_registry_thresholds_informed"] is False
    assert result["counts_and_join"]["train_sample_count"] == 1080
    assert result["counts_and_join"]["validation_sample_count"] == 300
    assert result["coverage_and_leakage"][
        "protected_array_dereference_count"
    ] == 0
    assert result["parameter_bundle"]["roundtrip_array_equal"] is True


def test_sample_order_hash_is_order_sensitive():
    first = audit_module._sample_order_hash(["a", "b"])
    second = audit_module._sample_order_hash(["b", "a"])
    assert first != second


def test_audit_refuses_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "exists"
    output.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--bundle-root",
            str(BUNDLE),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        audit_module.main()
