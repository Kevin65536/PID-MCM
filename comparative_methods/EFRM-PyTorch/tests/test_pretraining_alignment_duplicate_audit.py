import json
import sys
from pathlib import Path

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from efrm_pytorch.pretraining_analysis import analyze_alignment_evidence
from efrm_pytorch.visualization import export_alignment_evidence


def test_full_validation_export_with_repeated_samples_fails_closed(tmp_path: Path) -> None:
    metadata = [
        {
            "sample_id": sample_id,
            "dataset_id": "public",
            "subject": subject,
            "record_id": record,
        }
        for sample_id, subject, record in [
            ("a", "s1", "r1"),
            ("b", "s2", "r2"),
            ("a", "s1", "r1"),
        ]
    ]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    evidence = export_alignment_evidence(
        tmp_path,
        eeg_embeddings=embeddings,
        fnirs_embeddings=embeddings,
        metadata=metadata,
        filename="full_validation_clip_alignment_evidence.npz",
    )

    metrics, _, _ = analyze_alignment_evidence(evidence)
    scope = metrics["evidence_scope"]

    assert scope["kind"] == "balanced_public_validation_epoch_with_repeated_samples"
    assert scope["row_count"] == 3
    assert scope["unique_sample_count"] == 2
    assert scope["duplicate_row_count"] == 1
    assert scope["contains_repeated_sample_rows"] is True
    assert scope["diagonal_only_positive_mask"] is True
    assert scope["representative_of_full_validation"] is False
    assert scope["deduplicated_metrics_required"] is True


def test_unique_full_validation_export_remains_eligible(tmp_path: Path) -> None:
    metadata = [
        {
            "sample_id": f"sample-{index}",
            "dataset_id": "public",
            "subject": f"s{index}",
            "record_id": f"r{index}",
        }
        for index in range(3)
    ]
    embeddings = np.eye(3, dtype=np.float32)
    evidence = export_alignment_evidence(
        tmp_path,
        eeg_embeddings=embeddings,
        fnirs_embeddings=embeddings,
        metadata=metadata,
        filename="full_validation_clip_alignment_evidence.npz",
    )

    metrics, _, _ = analyze_alignment_evidence(evidence)
    scope = metrics["evidence_scope"]

    assert scope["kind"] == "full_public_validation"
    assert scope["unique_sample_count"] == 3
    assert scope["duplicate_row_count"] == 0
    assert scope["representative_of_full_validation"] is True
    assert scope["deduplicated_metrics_required"] is False
