import csv
import json
from pathlib import Path

from experiments.summarize_ssm_observation_target_screen import summarize


MODES = (
    "SSM-SELF",
    "SSM-SELF-XPRED-0.02",
    "SSM-SELF-XPRED-0.05",
)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run(path: Path, seed: int, eeg_delta: float):
    path.mkdir()
    manifest = {
        "schema": "ssm_observation_target_screen_v1",
        "protected_open": False,
        "determinism": {"torch_deterministic_algorithms": True},
        "inputs": [{"path": "fixed", "sha256": "abc"}],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for task in ("motor_imagery", "word_generation"):
        for mode in MODES:
            rows.append(
                {
                    "task_id": task,
                    "mode": mode,
                    "seed": seed,
                    "representation_selection_score": 0.2,
                    "selection_eeg_clean_delta_r2_vs_condition_time_mean": eeg_delta,
                    "selection_fnirs_clean_delta_r2_vs_condition_time_mean": 0.3,
                    "selection_private_only_subject_equal_macro_f1": 0.5,
                    "selection_private_plus_shared_marginal_subject_equal_macro_f1": 0.5,
                    "selection_private_shared_interaction_subject_equal_macro_f1": 0.5,
                    "selection_interaction_macro_f1_increment": 0.0,
                }
            )
    _write_csv(path / "results.csv", rows)
    _write_csv(
        path / "teacher_provenance.csv",
        [{"task_id": "motor_imagery", "labels_used": "False"}],
    )
    _write_csv(
        path / "provenance_uncertainty_control.csv",
        [{"task_id": "motor_imagery", "subject_equal_balanced_accuracy": 0.5}],
    )


def test_multiseed_gate_fails_when_any_seed_has_negative_eeg_r2(tmp_path):
    runs = []
    for index, delta in enumerate((0.1, -0.01, 0.2)):
        path = tmp_path / f"run{index}"
        _run(path, 10 + index, delta)
        runs.append(path)
    output = summarize(runs, tmp_path / "summary")
    gate = json.loads((output / "vq_stage_gate.json").read_text(encoding="utf-8"))
    assert gate["advance_to_independent_k16_vq"] is False
    assert set(gate["mode_pass"].values()) == {False}
    assert gate["q0_q1_status"] == "deferred"
    assert gate["protected_open"] is False
