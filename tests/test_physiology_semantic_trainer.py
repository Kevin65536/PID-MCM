import json

import pytest
import torch

from experiments.train_physiology_semantic_tokenizer import (
    E0_SCHEMA,
    _coordinate_mask,
    _load_e0_gate,
    _scheduler,
    _validate_loader_subjects,
)


def test_training_requires_real_passed_gate_file(tmp_path):
    config = {
        "data": {"contract": "physiology_semantic_v2", "split": {"train_subjects": [1]}},
        "validation": {"e0_gate_path": str(tmp_path / "gate.json")},
    }
    import hashlib
    split_hash = hashlib.sha256(
        json.dumps(config["data"]["split"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    (tmp_path / "gate.json").write_text(
        json.dumps({
            "schema": E0_SCHEMA, "gate": "G0", "status": "failed", "e0_passed": False,
            "data_contract": "physiology_semantic_v2", "split_sha256": split_hash,
            "cache_source_roots": [],
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        _load_e0_gate(config, require_pass=True)

    (tmp_path / "gate.json").write_text(
        json.dumps(
            {
                "schema": E0_SCHEMA,
                "gate": "G0",
                "status": "gate_passed",
                "e0_passed": True,
                "data_contract": "physiology_semantic_v2",
                "split_sha256": split_hash,
                "cache_source_roots": [],
                "admissible_coordinates": {"eeg": ["r_mean"], "fnirs": ["delta_f_mean"]},
            }
        ),
        encoding="utf-8",
    )
    gate, digest = _load_e0_gate(config, require_pass=True)
    assert gate["e0_passed"] is True
    assert len(digest) == 64


def test_coordinate_mask_uses_only_gate_admitted_names():
    mask = _coordinate_mask(("a", "b", "c"), ["b"])
    assert torch.equal(mask, torch.tensor([False, True, False]))


def test_warmup_cosine_scheduler_reaches_lower_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _scheduler(optimizer, warmup_steps=2, total_steps=6)
    rates = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] > 0
    assert rates[-1] < rates[1]


def test_loader_subject_audit_rejects_missing_declared_subject():
    class Entry:
        def __init__(self, subject_id):
            self.subject_id = subject_id

    class Dataset:
        entries = [Entry(1)]

    class Loader:
        dataset = Dataset()

    loaders = {"train": Loader(), "val": Loader(), "test": Loader()}
    config = {
        "data": {
            "split": {
                "train_subjects": [1, 2],
                "val_subjects": [1],
                "test_subjects": [1],
            }
        }
    }
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        _validate_loader_subjects(loaders, config)
