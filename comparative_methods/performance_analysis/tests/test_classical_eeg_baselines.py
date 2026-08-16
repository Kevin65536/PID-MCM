from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from comparative_methods.performance_analysis.classical_eeg_baselines import (
    BANDS,
    PANEL_BY_TASK,
    PublicBoundaryError,
    _bandpower_features,
    _public_manifest,
    extract_features,
    public_split_path,
)


def test_bandpower_feature_shape_and_finiteness() -> None:
    sample_rate = 200.0
    time = np.arange(1600, dtype=np.float64) / sample_rate
    eeg = np.stack(
        [np.sin(2.0 * np.pi * (10.0 + 0.2 * channel) * time) for channel in range(16)],
        axis=0,
    )
    features = _bandpower_features(eeg, sample_rate_hz=sample_rate)
    assert features.shape == (16 * len(BANDS),)
    assert np.isfinite(features).all()
    # The deliberately injected 10 Hz rhythm should dominate the alpha band
    # for the first channel; this catches accidental frequency-axis mistakes.
    alpha_index = 2
    channel_features = features.reshape(16, len(BANDS))[0]
    assert channel_features[alpha_index] > channel_features[0]
    assert channel_features[alpha_index] > channel_features[1]


def test_public_split_path_never_resolves_protected() -> None:
    with pytest.raises(PublicBoundaryError):
        public_split_path(Path("/tmp/protected"), "visual", 0)


def test_public_manifest_rejects_opened_protected_test(tmp_path: Path) -> None:
    path = tmp_path / "outer0.json"
    path.write_text(
        json.dumps(
            {
                "schema": "efrm_target_public_fold_v1",
                "task": "visual",
                "outer_fold": 0,
                "protocol": "strict_cross_subject",
                "protected_test_opened": True,
                "train_indices": [0],
                "validation_indices": [1],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PublicBoundaryError):
        _public_manifest(path, task="visual", outer_fold=0)


class _FakeDataset:
    def __init__(self, sample: dict, row: dict) -> None:
        self.base = [sample]
        self.indices = [0]
        self._row = row

    def __len__(self) -> int:
        return 1

    def lightweight_metadata(self, index: int) -> dict:
        assert index == 0
        return self._row


def test_extract_features_uses_fixed_visual_panel() -> None:
    panel = PANEL_BY_TASK["visual"]
    all_names = list(panel) + ["F7", "F8"]
    rng = np.random.default_rng(3)
    sample = {
        "eeg": rng.normal(size=(len(all_names), 1600)).astype(np.float32),
        "channel_names": {"eeg": all_names},
        "sample_rate_hz": {"eeg": 200.0},
        "analysis_valid_mask": {"eeg": np.ones(1600, dtype=bool)},
        "bad_channel_mask": {"eeg": np.zeros(len(all_names), dtype=bool)},
    }
    dataset = _FakeDataset(
        sample,
        {
            "class_index": 2,
            "subject": "S01",
            "join_key": "visual|S01|r0",
            "event_index": 0,
            "window_offset_s": 0.0,
        },
    )
    batch = extract_features(dataset, [0], task="visual")  # type: ignore[arg-type]
    assert batch.x.shape == (1, 16 * len(BANDS))
    assert batch.y.tolist() == [2]
    assert batch.subjects.tolist() == ["S01"]
    assert batch.skipped == ()


def test_extract_features_records_invalid_support_instead_of_leaking() -> None:
    panel = PANEL_BY_TASK["nback"]
    rng = np.random.default_rng(4)
    sample = {
        "eeg": rng.normal(size=(len(panel), 1600)).astype(np.float32),
        "channel_names": {"eeg": list(panel)},
        "sample_rate_hz": {"eeg": 200.0},
        "analysis_valid_mask": {"eeg": np.r_[np.ones(1599, dtype=bool), False]},
        "bad_channel_mask": {"eeg": np.zeros(len(panel), dtype=bool)},
    }
    dataset = _FakeDataset(
        sample,
        {"class_index": 0, "subject": "VP001", "join_key": "sim|VP001|r0"},
    )
    with pytest.raises(RuntimeError, match="no valid public EEG windows"):
        extract_features(dataset, [0], task="nback")  # type: ignore[arg-type]
