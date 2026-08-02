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

from brainfusion_gpu.features import BrainFusionFeaturePipeline, CSPConfig
from brainfusion_gpu.pipeline import BrainFusionFoldPipeline
from brainfusion_gpu.stacking import StackingConfig


def _signals(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0] * 8 + [1] * 8)
    eeg = torch.randn(16, 6, 1600, generator=generator)
    hbo = torch.randn(16, 5, 80, generator=generator)
    hbr = torch.randn(16, 5, 80, generator=generator)
    eeg[labels == 1, 0] += torch.sin(torch.linspace(0, 20, 1600)) * 1.5
    hbo[labels == 1, 0] += torch.linspace(-1, 1, 80)
    hbr[labels == 1, 1] -= torch.linspace(-1, 1, 80)
    return eeg, hbo, hbr, labels


def _pipeline() -> BrainFusionFoldPipeline:
    return BrainFusionFoldPipeline(
        features=BrainFusionFeaturePipeline(
            csp_config=CSPConfig(components_per_class=2), nvc_pair_count=12
        ),
        stacking_config=StackingConfig(
            inner_folds=2,
            seed=17,
            linear_svm_c_values=(0.1,),
            rbf_svm_c_values=(1.0,),
            random_forest_estimators=16,
        ),
    )


def test_complete_fold_pipeline_reloads_with_exact_predictions(tmp_path: Path) -> None:
    eeg, hbo, hbr, labels = _signals(17)
    groups = [f"subject-{index // 2}" for index in range(16)]
    ids = [f"outer-train-{index}" for index in range(16)]
    pipeline = _pipeline().fit(
        eeg, hbo, hbr, labels, groups=groups, sample_ids=ids
    )
    audit = pipeline.audit_state()
    assert audit["all_fitted_state_outer_training_only"] is True
    assert audit["protected_test_opened"] is False

    validation_eeg, validation_hbo, validation_hbr, _ = _signals(73)
    before = pipeline.predict(validation_eeg, validation_hbo, validation_hbr)
    scores_before = pipeline.decision_function(
        validation_eeg, validation_hbo, validation_hbr
    )
    checkpoint = pipeline.save(tmp_path / "public_checkpoint")
    restored = BrainFusionFoldPipeline.load(checkpoint)
    np.testing.assert_array_equal(
        restored.predict(validation_eeg, validation_hbo, validation_hbr), before
    )
    np.testing.assert_allclose(
        restored.decision_function(validation_eeg, validation_hbo, validation_hbr),
        scores_before,
    )
    assert restored.audit_state() == audit


def test_complete_pipeline_refuses_protected_artifact_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="protected"):
        BrainFusionFoldPipeline.load(tmp_path / "protected" / "checkpoint")
