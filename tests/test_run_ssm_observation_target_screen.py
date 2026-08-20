from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from experiments.run_ssm_observation_target_screen import (
    _mode_xpred_weight,
    _teacher_mode,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/configs/physiology_semantic_tokenizer/ssm_observation_target_screen.yaml"
)


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_xpred_modes_share_teacher_targets_but_keep_fixed_low_weights():
    assert _teacher_mode("SSM-SELF-XPRED-0.02") == "SSM-SELF-XPRED"
    assert _teacher_mode("SSM-SELF-XPRED-0.05") == "SSM-SELF-XPRED"
    assert _mode_xpred_weight("SSM-SELF-XPRED-0.02") == 0.02
    assert _mode_xpred_weight("SSM-SELF-XPRED-0.05") == 0.05
    assert _mode_xpred_weight("SSM-SELF") == 0.0


def test_registered_ssm_observation_screen_config_is_leakage_closed():
    config = _config()
    validate_config(config)
    assert config["source"]["task_labels_enter_ssm"] is False
    assert config["source"]["condition_specific_ssm_parameters"] is False
    assert config["objective"]["vector_quantization"] is False
    assert config["objective"]["direct_latent_alignment"] is False
    assert config["teachers"]["joint_role"] == "privileged_upper_bound_only"


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        ("source", "task_labels_enter_ssm", True, "task labels"),
        ("source", "condition_specific_ssm_parameters", True, "label leakage"),
        ("objective", "vector_quantization", True, "VQ disabled"),
        ("objective", "direct_latent_alignment", True, "alignment"),
        ("objective", "bidirectional_matching", True, "bidirectional"),
        ("statistics", "coupling_endpoint_claim", True, "coupling endpoint"),
    ),
)
def test_screen_config_fails_closed_on_claim_or_leakage_drift(
    section, key, value, message
):
    config = deepcopy(_config())
    config[section][key] = value
    with pytest.raises((ValueError, PermissionError), match=message):
        validate_config(config)
