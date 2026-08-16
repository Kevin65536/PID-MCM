from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from brainfusion_protocol_bridge import (  # noqa: E402
    _nvc_batch_stats,
    _protocol_capability,
    _safe_path,
)


def test_protocol_refuses_protected_paths() -> None:
    with pytest.raises(PermissionError):
        _safe_path(METHOD_ROOT / "runs" / "protected" / "artifact.json")


def test_protocol_capability_is_fail_closed() -> None:
    capability = _protocol_capability()
    assert capability["strict_cross_subject_protected"]["status"] == "locked"
    assert capability["strict_cross_subject_protected"]["protected_test_opened"] is False
    assert capability["long_post_stimulus_fnirs"]["status"] == "unavailable"
    assert capability["long_post_stimulus_fnirs"]["extra_post_interval_context_s"] == 0.0


def test_prefix_nvc_diagnostic_has_expected_pair_count() -> None:
    torch.manual_seed(7)
    eeg = torch.randn(6, 3, 1600)
    hbo = torch.randn(6, 4, 80)
    hbr = torch.randn(6, 4, 80)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    result = _nvc_batch_stats(
        eeg,
        hbo,
        hbr,
        labels,
        window_s=2.0,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert result["eeg_samples"] == 400
    assert result["fnirs_samples"] == 20
    assert result["pair_count"] == 3 * (2 * 4)
    assert result["class_separation"]["class_ids"] == [0, 1]
    assert result["class_separation"]["mean_abs_auc_class_b"] >= 0.0
    assert result["class_separation"]["mean_abs_auc_class_b"] <= 1.0


def test_prefix_nvc_diagnostic_refuses_unavailable_post_context() -> None:
    torch.manual_seed(11)
    eeg = torch.randn(2, 2, 1600)
    hbo = torch.randn(2, 2, 80)
    hbr = torch.randn(2, 2, 80)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="exceeds cached 8-second support"):
        _nvc_batch_stats(
            eeg,
            hbo,
            hbr,
            labels,
            window_s=10.0,
            device=torch.device("cpu"),
            batch_size=2,
        )
