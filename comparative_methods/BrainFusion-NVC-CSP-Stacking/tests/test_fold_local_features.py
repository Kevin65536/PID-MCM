from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = METHOD_ROOT / "adapters"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from brainfusion_gpu.features import (
    BrainFusionFeaturePipeline,
    CSPConfig,
    NVCPairSelector,
    TorchCSP,
)
from brainfusion_gpu.nvc import NVCConfig, brainfusion_nvc_contribution_timeseries


def _signals(seed: int = 17) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0] * 8 + [1] * 8)
    eeg = torch.randn(16, 6, 1600, generator=generator)
    hbo = torch.randn(16, 5, 80, generator=generator)
    hbr = torch.randn(16, 5, 80, generator=generator)
    eeg[labels == 1, 0] += torch.sin(torch.linspace(0, 20, 1600)) * 1.5
    hbo[labels == 1, 0] += torch.linspace(-1, 1, 80)
    hbr[labels == 1, 1] -= torch.linspace(-1, 1, 80)
    return eeg, hbo, hbr, labels


def test_dynamic_nvc_contributions_sum_to_public_pearson_coefficients() -> None:
    eeg, hbo, hbr, _ = _signals()
    convolved, contributions, correlations = brainfusion_nvc_contribution_timeseries(
        eeg, hbo, hbr, NVCConfig()
    )
    assert convolved.shape == (16, 6, 80)
    assert contributions.shape == (16, 60, 80)
    torch.testing.assert_close(contributions.sum(dim=-1), correlations)
    assert bool(torch.isfinite(contributions).all())


def test_dynamic_nvc_rejects_mismatched_or_incomplete_observation_grids() -> None:
    eeg, hbo, hbr, _ = _signals()
    with pytest.raises(ValueError, match="identical EEG-summary and fNIRS grids"):
        brainfusion_nvc_contribution_timeseries(eeg, hbo[..., :-1], hbr[..., :-1])
    with pytest.raises(ValueError, match="incomplete EEG averaging window"):
        brainfusion_nvc_contribution_timeseries(eeg[..., :-1], hbo, hbr)


def test_csp_is_deterministic_finite_and_bound_to_training_identities() -> None:
    eeg, _hbo, _hbr, labels = _signals()
    sample_ids = [f"train-{index}" for index in range(len(labels))]
    first = TorchCSP(CSPConfig(components_per_class=2)).fit_transform(
        eeg, labels, sample_ids=sample_ids
    )
    second_model = TorchCSP(CSPConfig(components_per_class=2))
    second = second_model.fit_transform(eeg, labels, sample_ids=sample_ids)
    torch.testing.assert_close(first, second)
    assert first.shape == (16, 4)
    assert bool(torch.isfinite(first).all())
    assert second_model.fit_sample_identity_sha256_ is not None


def test_nvc_pair_selection_uses_labels_but_is_deterministic_on_training_fold() -> None:
    generator = torch.Generator().manual_seed(42)
    contributions = torch.randn(20, 12, 40, generator=generator) * 0.01
    labels = torch.tensor([0] * 10 + [1] * 10)
    contributions[labels == 1, 7] += 0.1
    ids = [f"train-{index}" for index in range(20)]
    selector = NVCPairSelector(pair_count=4).fit(
        contributions, labels, sample_ids=ids
    )
    assert selector.indices_ is not None
    assert int(selector.indices_[0]) == 7
    assert selector.transform(contributions).shape == (20, 4, 40)


def test_four_view_feature_pipeline_fits_every_state_on_same_outer_train_ids() -> None:
    eeg, hbo, hbr, labels = _signals()
    ids = [f"outer-train-{index}" for index in range(len(labels))]
    pipeline = BrainFusionFeaturePipeline(
        csp_config=CSPConfig(components_per_class=2), nvc_pair_count=12
    )
    features = pipeline.fit_transform(eeg, hbo, hbr, labels, sample_ids=ids)
    assert set(features) == {"eeg", "hbo", "hbr", "nvc"}
    assert all(value.shape == (16, 4) for value in features.values())
    assert all(bool(torch.isfinite(value).all()) for value in features.values())
    audit = pipeline.audit_state()
    assert audit["nvc_pair_count"] == 12
    assert audit["all_fitted_states_share_training_identity"] is True


def test_feature_pipeline_refuses_duplicate_training_identity() -> None:
    eeg, hbo, hbr, labels = _signals()
    duplicate_ids = ["duplicate"] * len(labels)
    with pytest.raises(ValueError, match="unique sample identity"):
        BrainFusionFeaturePipeline().fit(
            eeg, hbo, hbr, labels, sample_ids=duplicate_ids
        )
