import json
from pathlib import Path

import yaml

from experiments.evaluate_e1_retention_gate import _evaluate_run


def _write_run(tmp_path: Path, fnirs_minimum: float = 30.0) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "diagnostics").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "training_complete",
                "protected_test_opened": False,
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    factors = {"revival_stop_after_steps": 200, "codebook_size": 128}
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"validation": {"registered_factors": factors}}),
        encoding="utf-8",
    )
    (run_dir / "implementation_snapshot.json").write_text(
        json.dumps({"files_sha256": {"model.py": "abc"}}), encoding="utf-8"
    )
    rows = []
    for index in range(7):
        rows.append(
            {
                "global_step": 231 + 33 * index,
                "validation": {
                    "eeg": {
                        "total_revivals": 10,
                        "effective_codes": 40.0 + index,
                        "epoch_active_codes": 80,
                        "epoch_active_fraction": 0.625,
                        "effective_rank": 64,
                        "nearest_neighbor_cosine": 0.95,
                        "quantization_strength": 1.0,
                    },
                    "fnirs": {
                        "total_revivals": 12,
                        "effective_codes": fnirs_minimum + index,
                        "epoch_active_codes": 100,
                        "epoch_active_fraction": 0.78125,
                        "effective_rank": 64,
                        "nearest_neighbor_cosine": 0.94,
                        "quantization_strength": 1.0,
                    },
                },
            }
        )
    (run_dir / "diagnostics" / "quantizer_health.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run_dir


def _rule() -> dict:
    return {
        "minimum_retention_epochs": 7,
        "require_completed_runs": True,
        "require_registered_factors_match": True,
        "protected_test_opened": False,
        "total_revivals_constant_in_retention_window": True,
        "minimum_effective_codes_in_retention_window": {"eeg": 32.0, "fnirs": 24.0},
        "minimum_final_epoch_active_fraction": {"eeg": 0.5, "fnirs": 0.75},
        "required_final_effective_rank": {"eeg": 64, "fnirs": 64},
        "maximum_final_nearest_neighbor_cosine": {"eeg": 0.99, "fnirs": 0.99},
        "required_final_quantization_strength": {"eeg": 1.0, "fnirs": 1.0},
    }


def test_e1_retention_gate_accepts_complete_healthy_training_only_run(tmp_path):
    run_dir = _write_run(tmp_path)

    decision = _evaluate_run(
        run_dir,
        {"revival_stop_after_steps": 200, "codebook_size": 128},
        _rule(),
    )

    assert decision["passed"] is True
    assert decision["modalities"]["fnirs"]["minimum_effective_codes"] == 30.0


def test_e1_retention_gate_rejects_transient_fnirs_effective_collapse(tmp_path):
    run_dir = _write_run(tmp_path, fnirs_minimum=20.0)

    decision = _evaluate_run(
        run_dir,
        {"revival_stop_after_steps": 200, "codebook_size": 128},
        _rule(),
    )

    assert decision["passed"] is False
    failed = [
        check["name"]
        for check in decision["modalities"]["fnirs"]["checks"]
        if not check["passed"]
    ]
    assert failed == ["minimum_effective_codes"]
