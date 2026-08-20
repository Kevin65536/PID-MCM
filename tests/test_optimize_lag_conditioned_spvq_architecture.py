from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from experiments import run_lag_conditioned_spvq as reviewed
from experiments.optimize_lag_conditioned_spvq_architecture import (
    CANDIDATES,
    CANDIDATE_IDS,
    _DevelopmentSelectionPermit,
    REFERENCE_CANDIDATE_ID,
    TASKS,
    _assert_target_mapping,
    _candidate_override_payload,
    _combined_cross_entropy,
    _issue_development_selection_permit,
    _stratified_dataset_view,
    _validated_development_checkpoint,
    _write_json_atomic,
    candidate_runtime_config,
    orchestrate_optimization,
    prepare_development_task,
    select_global_candidate,
    validate_optimization_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/physiology_semantic_tokenizer/lag_conditioned_spvq_architecture_optimization.yaml"
BASE_PATH = ROOT / "experiments/configs/physiology_semantic_tokenizer/lag_conditioned_spvq.yaml"


def config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_registered_binding_and_exact_five_candidates():
    value = config()
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    validate_optimization_config(value, config_path=CONFIG_PATH, base_config=base)
    assert tuple(row["candidate_id"] for row in value["candidates"]) == CANDIDATE_IDS
    assert len(value["candidates"]) == 5
    assert CANDIDATE_IDS == (
        "reference_h23_lag01",
        "lag05_h23",
        "h13_lag01",
        "h12_lag01",
        "reference_h23_lag01_long",
    )
    assert value["execution"]["multi_seed_repetition"] is False
    assert value["sample_budget"]["samples_per_subject_class"] == 8


def test_runner_direct_cli_import_path_is_executable():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/optimize_lag_conditioned_spvq_architecture.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Fit-selection-only architecture optimization" in completed.stdout


def test_binding_rejects_in_memory_drift_and_protected_open():
    value = config()
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    drift = deepcopy(value)
    drift["execution"]["variant"] = "N1"
    with pytest.raises(ValueError, match="execution"):
        validate_optimization_config(drift, base_config=base)
    closed = deepcopy(value)
    closed["experiment"]["protected_open"] = True
    with pytest.raises((ValueError, PermissionError), match="experiment|protected"):
        validate_optimization_config(closed, base_config=base)


def test_rejects_nonintegral_budget_and_multi_seed():
    value = config()
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    nonintegral = deepcopy(value)
    nonintegral["sample_budget"]["samples_per_subject_class"] = 8.5
    with pytest.raises(ValueError, match="integral|sample_budget"):
        validate_optimization_config(nonintegral, base_config=base)
    multiseed = deepcopy(value)
    multiseed["execution"]["seed"] = [20260820, 20260821]
    with pytest.raises(ValueError, match="execution"):
        validate_optimization_config(multiseed, base_config=base)


def test_development_preparation_rejects_calls_without_global_decision_capability(
    tmp_path,
):
    with pytest.raises(PermissionError, match="post-selection capability"):
        prepare_development_task(config(), SimpleNamespace())
    decision = select_global_candidate(
        {
            (task, candidate["candidate_id"]): _candidate_result(
                task, candidate["candidate_id"], 0.5, 0.4, 1.0
            )
            for task in TASKS
            for candidate in CANDIDATES
        }
    )
    decision_path = tmp_path / "global_candidate_selection.json"
    _write_json_atomic(decision_path, decision)
    with pytest.raises(PermissionError, match="only be issued"):
        _DevelopmentSelectionPermit("a" * 64, decision_path)
    permit = _issue_development_selection_permit(decision, decision_path)
    digest = permit.global_selection_digest
    permit.consume(TASKS[0], digest)
    with pytest.raises(PermissionError, match="already materialized"):
        permit.consume(TASKS[0], digest)
    with pytest.raises(PermissionError, match="does not match"):
        permit.consume(TASKS[1], "b" * 64)
    permit.consume_application(TASKS[0], REFERENCE_CANDIDATE_ID, digest)
    with pytest.raises(PermissionError, match="already applied"):
        permit.consume_application(TASKS[0], REFERENCE_CANDIDATE_ID, digest)


def _candidate_result(task_id, candidate_id, primary, coupling, ce, representation=1.0):
    return {
        "task_id": task_id,
        "candidate_id": candidate_id,
        "seed": 20260820,
        "variant": "M1",
        "status": "completed",
        "protected_open": False,
        "protected_measured_access_count": 0,
        "development_used": False,
        "development_values_seen": False,
        "all_fit_parameter_codes_active": True,
        "complete_registered_task_support": True,
        "derangement_nonoverlap_verified": True,
        "pretrain_steps": 1,
        "vq_steps": 1,
        "head_steps": 1,
        "trainable_parameter_count": 100,
        "selection_primary_metric": primary,
        "selection_coupling_only_subject_equal_macro_f1": coupling,
        "selection_combined_cross_entropy": ce,
        "selection_fixed_native_plus_lag_loss": representation,
    }


def test_global_selector_uses_task_mean_then_tie_breakers_and_reference_threshold():
    results = {}
    for task in TASKS:
        for index, candidate in enumerate(CANDIDATES):
            candidate_id = candidate["candidate_id"]
            results[(task, candidate_id)] = _candidate_result(
                task,
                candidate_id,
                primary=0.50 + (0.005 if index == 1 else 0.0),
                coupling=0.40 + (0.01 if index == 2 else 0.0),
                ce=1.0 - (0.1 if index == 2 else 0.0),
            )
    # Candidate 2 wins the mean primary metric, but the configured minimum
    # descriptive improvement forces the registered reference recommendation.
    selection = select_global_candidate(results)
    assert selection["proposed_candidate_id"] == "lag05_h23"
    assert selection["selected_candidate_id"] == "lag05_h23"
    assert selection["recommended_candidate_id"] == REFERENCE_CANDIDATE_ID

    tied = {}
    for task in TASKS:
        for candidate in CANDIDATES:
            tied[(task, candidate["candidate_id"])] = _candidate_result(
                task, candidate["candidate_id"], 0.5, 0.4, 1.0
            )
    tied_selection = select_global_candidate(tied)
    assert tied_selection["selected_candidate_id"] == REFERENCE_CANDIDATE_ID

    ce_tied = {}
    for task in TASKS:
        for candidate in CANDIDATES:
            candidate_id = candidate["candidate_id"]
            ce_tied[(task, candidate_id)] = _candidate_result(
                task,
                candidate_id,
                primary=0.5,
                coupling=0.4,
                ce=0.8 if candidate_id == "lag05_h23" else 1.0,
            )
    ce_selection = select_global_candidate(ce_tied)
    assert ce_selection["proposed_candidate_id"] == "lag05_h23"
    assert ce_selection["selected_candidate_id"] == "lag05_h23"
    assert ce_selection["recommended_candidate_id"] == REFERENCE_CANDIDATE_ID


def test_global_selector_applies_numeric_tolerance_before_step_tie_break():
    results = {}
    for task in TASKS:
        for candidate in CANDIDATES:
            candidate_id = candidate["candidate_id"]
            primary = 0.4
            steps = 20
            if candidate_id == REFERENCE_CANDIDATE_ID:
                primary = 0.5
                steps = 10
            elif candidate_id == "lag05_h23":
                primary = 0.5 + 0.5e-8
                steps = 5
            result = _candidate_result(task, candidate_id, primary, 0.4, 1.0)
            result["pretrain_steps"] = steps
            result["trainable_parameter_count"] = 100
            results[(task, candidate_id)] = result
    selection = select_global_candidate(results)
    assert selection["proposed_candidate_id"] == "lag05_h23"
    assert selection["numeric_tie_tolerance"] == 1.0e-8


def test_global_selector_rejects_unknown_duplicate_and_provenance_drift():
    rows = [
        _candidate_result(task, candidate["candidate_id"], 0.5, 0.4, 1.0)
        for task in TASKS
        for candidate in CANDIDATES
    ]
    with pytest.raises(ValueError, match="duplicate"):
        select_global_candidate(rows + [deepcopy(rows[0])])
    unknown = {
        (task, candidate["candidate_id"]): _candidate_result(
            task, candidate["candidate_id"], 0.5, 0.4, 1.0
        )
        for task in TASKS
        for candidate in CANDIDATES
    }
    unknown[(TASKS[0], "unknown_candidate")] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="unregistered candidate"):
        select_global_candidate(unknown)
    drifted = deepcopy(rows)
    drifted[0].pop("status")
    with pytest.raises(ValueError, match="status provenance"):
        select_global_candidate(drifted)
    incomplete = deepcopy(rows)
    incomplete[0].pop("all_fit_parameter_codes_active")
    with pytest.raises(ValueError, match="lacks boolean"):
        select_global_candidate(incomplete)


def test_global_selector_excludes_invalid_candidate_and_requires_controls():
    results = {
        (task, candidate["candidate_id"]): _candidate_result(
            task, candidate["candidate_id"], 0.5, 0.4, 1.0
        )
        for task in TASKS
        for candidate in CANDIDATES
    }
    for task in TASKS:
        results[(task, "h13_lag01")]["all_fit_parameter_codes_active"] = False
    selection = select_global_candidate(results)
    assert "h13_lag01" not in selection["ranking"]
    assert selection["validity"]["h13_lag01"]["rankable"] is False

    for task in TASKS:
        results[(task, REFERENCE_CANDIDATE_ID)][
            "all_fit_parameter_codes_active"
        ] = False
    with pytest.raises(RuntimeError, match="required comparison control"):
        select_global_candidate(results)


def test_candidate_override_payload_has_no_obsolete_depth_or_rank_lookup():
    for candidate in CANDIDATES:
        assert _candidate_override_payload(candidate) == {
            "eeg_shared_history_tokens": candidate["eeg_shared_history_tokens"],
            "fnirs_shared_history_tokens": candidate["fnirs_shared_history_tokens"],
            "lag_loss_weight": candidate["lag_loss_weight"],
            "step_multiplier": candidate["step_multiplier"],
        }


def test_integer_targets_must_match_canonical_conditions():
    valid = SimpleNamespace(
        condition=np.asarray(["LMI", "RMI", "LMI"]),
        target=np.asarray([0, 1, 0], dtype=np.int64),
    )
    _assert_target_mapping(valid, "motor_imagery")
    invalid = SimpleNamespace(
        condition=valid.condition,
        target=np.asarray([1, 1, 0], dtype=np.int64),
    )
    with pytest.raises(PermissionError, match="integer targets"):
        _assert_target_mapping(invalid, "motor_imagery")
    for values in (
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray(["0", "1", "0"]),
        np.asarray([False, True, False]),
    ):
        with pytest.raises(PermissionError, match="integer dtype"):
            _assert_target_mapping(
                SimpleNamespace(condition=valid.condition, target=values),
                "motor_imagery",
            )


def test_candidate_runtime_applies_only_registered_history_objective_and_budget_fields():
    value = config()
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    for candidate in CANDIDATES:
        runtime = candidate_runtime_config(
            base, value, candidate, seed=value["execution"]["seed"]
        )
        assert runtime["model"]["encoder_depth"] == 2
        assert runtime["head"]["coupling_rank"] == 8
        assert runtime["model"]["eeg_shared_history_tokens"] == candidate[
            "eeg_shared_history_tokens"
        ]
        assert runtime["model"]["fnirs_shared_history_tokens"] == candidate[
            "fnirs_shared_history_tokens"
        ]
        assert runtime["objective"]["lag_loss_weight_candidates"] == [
            candidate["lag_loss_weight"]
        ]
        assert runtime["training"]["amp"] is False
        assert runtime["training"]["betas"] == [0.9, 0.98]


def test_all_candidates_have_equal_model_parameter_count():
    value = config()
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    prepared = SimpleNamespace(
        task_id="motor_imagery",
        parameter=SimpleNamespace(
            eeg_channel_names=("EEG",),
            fnirs_channel_names=("fNIRS",),
            eeg_native=SimpleNamespace(feature_names=("a", "b", "c", "d", "e")),
            fnirs_native=SimpleNamespace(feature_names=("a", "b", "c", "d")),
        ),
    )
    counts = []
    for candidate in CANDIDATES:
        runtime = candidate_runtime_config(
            base, value, candidate, seed=value["execution"]["seed"]
        )
        model = reviewed._lc_spvq_model(prepared, runtime)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1


def test_sample_registry_is_seeded_record_round_robin_and_event_time_spread():
    rows = []
    for condition in ("left_hand", "right_hand"):
        for record_id in ("R1", "R2", "R3"):
            for event_index in range(10):
                rows.append(
                    SimpleNamespace(
                        subject="subject_01",
                        condition=condition,
                        record_id=record_id,
                        event_time_ms=float(event_index * 1000),
                        fnirs_event_time_ms=float(event_index * 1000 + 100),
                        sample_id=f"{condition}-{record_id}-{event_index}",
                    )
                )
    dataset = SimpleNamespace(
        rows=tuple(rows),
        derangement="present",
        measured_access_count=99,
        protected_measured_access_count=99,
    )
    first = _stratified_dataset_view(
        dataset, samples_per_subject_class=8, sample_registry_seed=20260824
    )
    second = _stratified_dataset_view(
        dataset, samples_per_subject_class=8, sample_registry_seed=20260824
    )
    assert tuple(row.sample_id for row in first.rows) == tuple(
        row.sample_id for row in second.rows
    )
    assert len(first.rows) == 16
    assert first.measured_access_count == 0
    assert first.protected_measured_access_count == 0
    for condition in ("left_hand", "right_hand"):
        condition_rows = [row for row in first.rows if row.condition == condition]
        record_counts = sorted(
            sum(row.record_id == record_id for row in condition_rows)
            for record_id in ("R1", "R2", "R3")
        )
        assert record_counts == [2, 3, 3]
        for record_id in ("R1", "R2", "R3"):
            times = [
                row.event_time_ms
                for row in condition_rows
                if row.record_id == record_id
            ]
            assert max(times) - min(times) >= 4000.0


def _fake_partition(role, count=4):
    return SimpleNamespace(
        role=role,
        sample_id=np.asarray([f"sample-{i}" for i in range(count)]),
        subject=np.asarray(["s1"] * count),
        condition=np.asarray(["a"] * count),
    )


def test_orchestration_defers_development_and_writes_outputs(tmp_path):
    value = config()
    base = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    fit_tasks = {
        task: SimpleNamespace(
            task_id=task,
            dataset_id="synthetic",
            parameter=_fake_partition("fit_parameter"),
            selection=_fake_partition("fit_selection"),
            measured_access_count=8,
            protected_measured_access_count=0,
        )
        for task in TASKS
    }
    events = []

    def fake_runner(fit_task, base_config, optimization_config, candidate, **kwargs):
        assert not hasattr(fit_task, "development")
        events.append(("candidate", fit_task.task_id, candidate["candidate_id"]))
        return {
            "task_id": fit_task.task_id,
            "candidate_id": candidate["candidate_id"],
            "candidate_role": candidate["role"],
            "seed": 20260820,
            "variant": "M1",
            "status": "completed",
            "protected_open": False,
            "protected_measured_access_count": 0,
            "development_used": False,
            "development_values_seen": False,
            "all_fit_parameter_codes_active": True,
            "complete_registered_task_support": True,
            "derangement_nonoverlap_verified": True,
            "pretrain_steps": 1,
            "vq_steps": 1,
            "head_steps": 1,
            "trainable_parameter_count": 100,
            "selection_primary_metric": 0.5,
            "selection_coupling_only_subject_equal_macro_f1": 0.4,
            "selection_combined_cross_entropy": 1.0,
            "selection_fixed_native_plus_lag_loss": 0.8,
            "step_curves": [
                {
                    "stage": "continuous_pretrain_eval",
                    "step": 1,
                    "selection_fixed_native_plus_lag_loss": 0.8,
                },
                {
                    "stage": "task_head_eval",
                    "step": 1,
                    "selection_fixed_native_plus_lag_loss": 0.8,
                    "selection_coupling_plus_private_subject_equal_macro_f1": 0.5,
                },
            ],
        }

    def fake_prepare(*args, **kwargs):
        events.append(("development_prepare", args[1].task_id))
        assert sum(event[0] == "candidate" for event in events[:-1]) == len(TASKS) * len(CANDIDATES)
        return SimpleNamespace(
            task_id=args[1].task_id,
            dataset_id="synthetic",
            partition=_fake_partition("development_apply"),
            measured_access_count=4,
            protected_measured_access_count=0,
        )

    def fake_evaluate(*args, **kwargs):
        events.append(("development_evaluate", args[3]["candidate_id"]))
        return (
            {
                "coupling_plus_private": {"subject_equal_macro_f1": 0.5},
                "development_combined_cross_entropy": 1.0,
            },
            {"prediction": np.asarray([1], dtype=np.int64)},
        )

    result = orchestrate_optimization(
        value,
        base,
        tmp_path / "run",
        device=torch.device("cpu"),
        fit_tasks=fit_tasks,
        candidate_runner=fake_runner,
        development_preparer=fake_prepare,
        development_evaluator=fake_evaluate,
    )
    first_development = next(i for i, event in enumerate(events) if event[0] == "development_prepare")
    assert all(event[0] == "candidate" for event in events[:first_development])
    assert result["selection"]["development_values_used"] is False
    assert (tmp_path / "run" / "candidate_summary.csv").is_file()
    assert (tmp_path / "run" / "fit_selection_sample_registry.json").is_file()
    assert (tmp_path / "run" / "development_sample_registry.json").is_file()
    assert (tmp_path / "run" / "development_comparison.json").is_file()
    assert (tmp_path / "run" / "selection_development_curves.png").is_file()
    assert (tmp_path / "run" / "selection_development_curves.pdf").is_file()
    assert (tmp_path / "run" / "curve_figure_source_data.csv").is_file()


def test_combined_cross_entropy_uses_reviewed_export_key():
    value = _combined_cross_entropy(
        {
            "coupling_plus_private_logits": np.asarray(
                [[3.0, 0.0], [0.0, 3.0]], dtype=np.float32
            ),
            "target": np.asarray([0, 1], dtype=np.int64),
        }
    )
    assert 0.0 < value < 0.1
    with pytest.raises(KeyError, match="coupling_plus_private_logits"):
        _combined_cross_entropy(
            {
                "combined_logits": np.zeros((2, 2), dtype=np.float32),
                "target": np.asarray([0, 1], dtype=np.int64),
            }
        )


def test_development_checkpoint_is_candidate_local_and_provenance_bound(tmp_path):
    candidate = CANDIDATES[0]
    candidate_dir = tmp_path / candidate["candidate_id"]
    checkpoint_path = candidate_dir / "checkpoints" / "head_best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    payload = {
        "schema": "lc_spvq_architecture_optimization_checkpoint_v1",
        "task_id": TASKS[0],
        "variant": "M1",
        "candidate_id": candidate["candidate_id"],
        "candidate_config_overrides": _candidate_override_payload(candidate),
        "seed": 20260820,
        "stage": "task_head",
        "fit_selection_score": 0.5,
        "model_state": {},
        "lag_objective_state": {},
        "optimizer_state": {},
        "rng_state": {},
        "quantization_strength": 1.0,
        "posterior_temperature": 0.1,
        "protected_open": False,
        "development_used": False,
    }
    torch.save(payload, checkpoint_path)
    loaded = _validated_development_checkpoint(
        candidate_dir,
        "checkpoints/head_best.pt",
        task_id=TASKS[0],
        candidate=candidate,
        seed=20260820,
        device=torch.device("cpu"),
    )
    assert loaded["candidate_id"] == candidate["candidate_id"]
    with pytest.raises(PermissionError, match="candidate-local"):
        _validated_development_checkpoint(
            candidate_dir,
            "../head_best.pt",
            task_id=TASKS[0],
            candidate=candidate,
            seed=20260820,
            device=torch.device("cpu"),
        )
    payload["candidate_id"] = CANDIDATE_IDS[1]
    torch.save(payload, checkpoint_path)
    with pytest.raises(PermissionError, match="candidate_id"):
        _validated_development_checkpoint(
            candidate_dir,
            "checkpoints/head_best.pt",
            task_id=TASKS[0],
            candidate=candidate,
            seed=20260820,
            device=torch.device("cpu"),
        )


def test_atomic_json_rejects_overwrite(tmp_path):
    path = tmp_path / "manifest.json"
    _write_json_atomic(path, {"ok": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_json_atomic(path, {"ok": False})
