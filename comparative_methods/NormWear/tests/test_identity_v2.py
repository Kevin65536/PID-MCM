from __future__ import annotations

from pathlib import Path

import yaml

from comparative_methods.NormWear.audit_identity_v2 import audit_identity


METHOD_ROOT = Path(__file__).resolve().parents[1]


def test_identity_chain_and_weights_only_checkpoint_audit_pass() -> None:
    report = audit_identity()
    assert report["status"] == "pass"
    assert report["upstream_clean"] is True
    assert report["checkpoint"]["container_type"] == "OrderedDict"
    assert report["checkpoint"]["tensor_entry_count"] == 261
    assert report["checkpoint"]["tensor_element_count"] == 136_116_425
    assert report["checkpoint"]["weights_only_load"] is True
    assert report["checkpoint"]["strict_model_match"] is True
    assert report["cell_registration"]["protected_test_opened"] is False


def test_a0_cells_and_adapter_decisions_are_frozen_before_scores() -> None:
    config = yaml.safe_load(
        (METHOD_ROOT / "configs/alignment_v2.yaml").read_text(encoding="utf-8")
    )
    supported = [task for task, cell in config["tasks"].items() if cell["supported"]]
    assert supported == [
        "motor_imagery",
        "mental_arithmetic",
        "wg",
        "nback",
        "dsr",
        "visual",
    ]
    assert config["tasks"]["refed_regression"]["unsupported_reason_code"] == (
        "NORMWEAR_NO_PARTIAL_TIME_MASK_OR_MASKED_SEQUENCE_REGRESSION_CONTRACT"
    )
    assert config["adapter"]["model_input_sample_rate_hz"] == 65
    assert config["adapter"]["resampling"]["eeg_ratio_up_down"] == [13, 40]
    assert config["adapter"]["resampling"]["fnirs_ratio_up_down"] == [13, 2]
    assert config["adapter"]["upstream_basic_preproc_applied"] is False
    assert config["adapter"]["channel_pooling"] == (
        "concatenate_in_frozen_delivered_order"
    )
    assert config["adapter"]["decoder_used"] is False
    assert config["adapter"]["optional_msitf_used"] is False
