from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from aggregate_formal_results import T95_DF4, scalar_metrics
from build_formal_protocol import (
    fold_manifest,
    sample_random_folds,
    strict_folds,
    validate_folds,
)
from efrm_pytorch.protocol import LODOPretrainingBoundary, SourceTargetBoundary


class FakeClassificationDataset:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            task_type="classification",
            class_names=("A", "B"),
        )
        self.class_to_index = {"A": 0, "B": 1}
        self.rows = [
            {
                "subject": f"S{subject:02d}",
                "condition": condition,
            }
            for subject in range(10)
            for condition in ("A", "B")
            for _repeat in range(3)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def lightweight_metadata(self, index: int) -> dict[str, str]:
        return self.rows[index]

    def metadata_fingerprint(self) -> str:
        return "fake-metadata"


def test_formal_fold_builders_cover_target_exactly() -> None:
    dataset = FakeClassificationDataset()
    subjects = sorted({row["subject"] for row in dataset.rows})
    target_indices = list(range(len(dataset)))
    strict = strict_folds(dataset, subjects)
    random = sample_random_folds(dataset, target_indices)
    validate_folds(
        dataset=dataset,
        target_indices=target_indices,
        folds=strict,
        strict=True,
    )
    validate_folds(
        dataset=dataset,
        target_indices=target_indices,
        folds=random,
        strict=False,
    )
    assert len(strict) == len(random) == 5
    for folds in (strict, random):
        tests = [index for _train, _validation, test in folds for index in test]
        assert len(tests) == len(set(tests)) == len(dataset)


def test_public_fold_hides_protected_indices() -> None:
    dataset = FakeClassificationDataset()
    train, validation, test = strict_folds(
        dataset, sorted({row["subject"] for row in dataset.rows})
    )[0]
    public, protected = fold_manifest(
        task="fake",
        protocol="strict_cross_subject",
        outer_index=0,
        dataset=dataset,
        train=train,
        validation=validation,
        test=test,
        cohort_sha256="cohort",
    )
    assert "test_indices" not in public
    assert public["protected_test"]["indices_sha256"] == protected["test_indices_sha256"]
    assert set(public["train_indices"]).isdisjoint(protected["test_indices"])
    assert set(public["validation_indices"]).isdisjoint(protected["test_indices"])


def test_fold_aggregation_uses_sample_sd_and_t_interval() -> None:
    rows = [{"metric": value} for value in (1.0, 2.0, 3.0, 4.0, 5.0)]
    result = scalar_metrics(rows)["metric"]
    expected_sd = float(np.std([1, 2, 3, 4, 5], ddof=1))
    expected_half_width = T95_DF4 * expected_sd / np.sqrt(5)
    assert result["mean"] == pytest.approx(3.0)
    assert result["sample_sd_ddof_1"] == pytest.approx(expected_sd)
    assert result["t95_df4"] == pytest.approx(
        [3.0 - expected_half_width, 3.0 + expected_half_width]
    )


def test_source_target_boundary_rejects_overlap_and_selects_source_roles(
    tmp_path: Path,
) -> None:
    cohort = {
        "schema": "efrm_source_target_cohort_v1",
        "protocol_id": "efrm_resource_bounded_dual_protocol_v1",
        "target_opened_during_pretraining": False,
        "datasets": {
            "d": {
                "source_subjects": ["train", "validation"],
                "source_train_subjects": ["train"],
                "source_validation_subjects": ["validation"],
                "target_subjects": ["target"],
            }
        },
    }
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(cohort), encoding="utf-8")
    boundary = SourceTargetBoundary(path)

    class FakePretrain:
        rows = [
            {"dataset_id": "d", "subject": "train"},
            {"dataset_id": "d", "subject": "validation"},
            {"dataset_id": "d", "subject": "target"},
        ]

        def __len__(self) -> int:
            return len(self.rows)

        def lightweight_metadata(self, index: int) -> dict[str, str]:
            return self.rows[index]

    dataset = FakePretrain()
    assert boundary.indices_for(dataset, "train") == [0]
    assert boundary.indices_for(dataset, "validation") == [1]
    assert boundary.manifest()["target_subjects_by_dataset"]["d"] == ("target",)

    cohort["datasets"]["d"]["target_subjects"] = ["train"]
    path.write_text(json.dumps(cohort), encoding="utf-8")
    with pytest.raises(RuntimeError, match="overlap"):
        SourceTargetBoundary(path)


def test_lodo_boundary_excludes_target_and_refit_uses_all_source_subjects(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema": "efrm_lodo_pretraining_manifest_v2",
        "protocol_id": "efrm_lodo_full_target_fivefold_v2",
        "target_dataset_exposure": False,
        "excluded_target_dataset": "target",
        "included_datasets": ["source"],
        "datasets": {
            "source": {
                "all_subjects": ["train", "validation"],
                "selection_train_subjects": ["train"],
                "selection_validation_subjects": ["validation"],
            }
        },
    }
    path = tmp_path / "lodo.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakePretrain:
        rows = [
            {"dataset_id": "source", "subject": "train"},
            {"dataset_id": "source", "subject": "validation"},
        ]

        def __len__(self) -> int:
            return len(self.rows)

        def lightweight_metadata(self, index: int) -> dict[str, str]:
            return self.rows[index]

    dataset = FakePretrain()
    selection = LODOPretrainingBoundary(path, stage="selection")
    refit = LODOPretrainingBoundary(path, stage="final_refit")
    assert selection.indices_for(dataset, "train") == [0]
    assert selection.indices_for(dataset, "validation") == [1]
    assert refit.indices_for(dataset, "train") == [0, 1]
    assert refit.indices_for(dataset, "validation") == [1]
    assert refit.manifest()["checkpoint_selection_allowed"] is False

    manifest["target_dataset_exposure"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PermissionError, match="exposure"):
        LODOPretrainingBoundary(path, stage="selection")
