import json

import pytest
import torch

from experiments.train_physiology_semantic_tokenizer import (
    E0_SCHEMA,
    _coordinate_mask,
    _finalize_epoch_health,
    _implementation_snapshot,
    _load_e0_gate,
    _load_semantic_weight_calibration,
    _loss_from_config,
    _quantizer_reference_tests,
    _quantization_strength_for_epoch,
    _scheduler,
    _teacher_supervision_requested,
    _update_epoch_health,
    _validate_loader_subjects,
    _audit_objective_gradients,
)
from src.losses.physiology_semantic import PhysiologySemanticLoss
from src.tokenizers.physiology_semantic_tokenizer import PhysiologySemanticTokenizer


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


def test_teacher_free_objective_does_not_claim_teacher_supervision():
    assert not _teacher_supervision_requested(
        {"loss": {"state": {"weight": 0}, "prototype": {"weight": 0}, "masked_state": {"weight": 0}}}
    )
    assert _teacher_supervision_requested({"loss": {"state": {"weight": 1.0}}})


def test_formal_semantic_weight_must_match_training_gradient_calibration(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({
            "schema": "physiology_semantic_e2_training_gradient_calibration_v1",
            "calibration_passed": True,
            "validation_target_decoding_read": False,
            "protected_test_opened": False,
            "selected_weight": 0.005,
        }),
        encoding="utf-8",
    )
    config = {
        "loss": {
            "state": {"weight": 0.005},
            "prototype": {"weight": 0.005},
        },
        "validation": {"semantic_weight_calibration_path": str(path)},
    }

    calibration, digest = _load_semantic_weight_calibration(
        config,
        require_pass=True,
    )

    assert calibration["selected_weight"] == 0.005
    assert len(digest) == 64
    config["loss"]["state"]["weight"] = 0.25
    with pytest.raises(ValueError, match="do not match"):
        _load_semantic_weight_calibration(config, require_pass=True)


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


def test_quantization_strength_schedule_matches_archived_epoch_ramp():
    schedule = {
        "enabled": True,
        "start_epoch": 1,
        "ramp_epochs": 5,
        "start_scale": 0.0,
        "end_scale": 1.0,
    }
    values = [_quantization_strength_for_epoch(epoch, schedule) for epoch in range(7)]
    assert values == pytest.approx([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    assert _quantization_strength_for_epoch(0, {}) == 1.0


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


def test_entry_routing_uses_separate_coordinate_allowlists():
    criterion = _loss_from_config(
        {
            "loss": {
                "uncertainty_weighting": False,
                "entry_routing": {
                    "local": {"eeg": ["r_mean"], "fnirs": ["delta_hbo_mean"]},
                    "prototype": {"eeg": ["r_slope"], "fnirs": ["delta_hb_slope"]},
                    "context": {"eeg": ["s_mean"], "fnirs": ["delta_f_mean"]},
                    "coupling": {"eeg": ["r_mean"], "fnirs": ["delta_f_slope"]},
                },
            }
        },
        gate=None,
    )
    assert not criterion.uncertainty_weighting
    assert torch.equal(
        criterion.eeg_local_coordinate_mask,
        torch.tensor([True, False, False, False, False, False]),
    )
    assert criterion.fnirs_local_coordinate_mask.nonzero().flatten().tolist() == [1]
    assert criterion.fnirs_coupling_coordinate_mask.nonzero().flatten().tolist() == [3]


def test_loss_config_selects_semantic_only_e1_reconstruction():
    criterion = _loss_from_config(
        {
            "loss": {
                "reconstruction": {
                    "weight": 1.0,
                    "mode": "semantic_only",
                    "semantic_input": "hard",
                },
                "codebook_balance": {
                    "weight": 0.08,
                    "temperature": 1.5,
                    "eeg_temperature": 2.0,
                    "fnirs_temperature": 1.0,
                    "eeg_scale": 1.0,
                    "fnirs_scale": 0.5,
                },
            }
        },
        gate=None,
    )
    assert criterion.reconstruction_mode == "semantic_only"
    assert criterion.reconstruction_semantic_input == "hard"
    assert criterion.weights["balance"] == pytest.approx(0.08)
    assert criterion.balance_temperature == pytest.approx(1.5)
    assert criterion.eeg_balance_temperature == pytest.approx(2.0)
    assert criterion.fnirs_balance_temperature == pytest.approx(1.0)
    assert criterion.eeg_balance_scale == pytest.approx(1.0)
    assert criterion.fnirs_balance_scale == pytest.approx(0.5)


def test_quantizer_reference_artifact_checks_all_invariants():
    result = _quantizer_reference_tests()
    assert result["all_passed"]
    assert all(result["checks"].values())


def test_implementation_snapshot_hashes_active_e1_sources(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("experiment: {name: test}\n", encoding="utf-8")
    snapshot = _implementation_snapshot(config)
    assert snapshot["schema"] == "physiology_semantic_implementation_snapshot_v1"
    assert len(snapshot["run_config_sha256"]) == 64
    assert "src/losses/physiology_semantic.py" in snapshot["files_sha256"]
    assert all(len(value) == 64 for value in snapshot["files_sha256"].values())


def test_epoch_health_aggregates_assignments_across_batches():
    class Output:
        pass

    aggregate = {}
    for ids in (torch.tensor([[0, 0]]), torch.tensor([[1, 1]])):
        quantizer = Output()
        quantizer.hard_ids = ids
        quantizer.posterior = torch.zeros(1, 2, 4)
        quantizer.health = {
            "prototype_drift": torch.tensor(0.5),
            "revived_codes": torch.tensor(0.0),
            "total_revivals": torch.tensor(0.0),
        }
        output = Output()
        output.quantizer = quantizer
        _update_epoch_health(aggregate, "eeg", output, valid_mask=None)

    health = _finalize_epoch_health(aggregate)["eeg"]
    assert health["epoch_active_codes"] == 2
    assert health["valid_tokens"] == 4
    assert health["effective_codes"] == pytest.approx(2.0)


def test_target_family_development_gate_is_scoped_and_hash_bound(tmp_path):
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    (sidecar / "manifest.json").write_text('{"schema":"sidecar"}\n', encoding="utf-8")
    import hashlib
    split = {"train_subject_keys": ["d|s1"], "val_subject_keys": ["d|s2"], "test_subject_keys": ["d|s3"]}
    config = {
        "data": {
            "contract": "physiology_semantic_measurement_local_v1",
            "cache_root": "cache",
            "split": split,
            "auxiliary_target": {
                "root": str(sidecar), "family": "family", "version": "v1",
            },
        },
        "validation": {
            "target_family_gate_path": str(tmp_path / "gate.json"),
            "promotion_eligible": False,
        },
    }
    (tmp_path / "gate.json").write_text(json.dumps({
        "schema": "physiology_semantic_target_family_gate_v1",
        "gate": "E0_OPTIONAL_TARGET_FAMILY_DEVELOPMENT",
        "status": "development_passed_protected_test_closed",
        "target_family_development_passed": True,
        "target_family": "family",
        "target_version": "v1",
        "data_contract": config["data"]["contract"],
        "cache_root": "cache",
        "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        "sidecar_manifest_sha256": hashlib.sha256((sidecar / "manifest.json").read_bytes()).hexdigest(),
        "protected_test_opened": False,
    }), encoding="utf-8")

    gate, digest = _load_e0_gate(config, require_pass=True)

    assert gate["target_family_development_passed"]
    assert len(digest) == 64


def test_gradient_audit_reports_modality_isolation_and_cosines():
    torch.manual_seed(17)
    model = PhysiologySemanticTokenizer(
        eeg_encoder_dim=32,
        fnirs_encoder_dim=24,
        semantic_dim=8,
        eeg_residual_dim=8,
        fnirs_residual_dim=4,
        codebook_size=16,
    )
    criterion = PhysiologySemanticLoss(
        state_weight=0.0,
        prototype_weight=0.0,
        masked_state_weight=0.0,
        reconstruction_weight=1.0,
        balance_weight=0.1,
        reconstruction_mode="semantic_only",
    )
    outputs = model(torch.randn(2, 6, 4000), torch.randn(2, 2, 200))
    losses = criterion(outputs, None)

    audit = _audit_objective_gradients(
        model,
        losses,
        criterion,
        {
            "strict": True,
            "objectives": [
                "eeg_reconstruction", "fnirs_reconstruction",
                "eeg_balance", "fnirs_balance",
            ],
        },
        global_step=0,
    )

    assert audit["all_contracts_passed"]
    assert audit["objectives"]["eeg_reconstruction"]["gradient_norm"] > 0
    assert not any(
        name.startswith("fnirs_branch.")
        for name in audit["objectives"]["eeg_reconstruction"]["reachable_parameters"]
    )
