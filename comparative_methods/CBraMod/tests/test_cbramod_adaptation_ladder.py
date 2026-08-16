from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from comparative_methods.CBraMod.scripts import run_adaptation_ladder as ladder


METHOD_ROOT = Path(__file__).resolve().parents[1]


def test_pilot_config_is_public_fixed_and_declares_capacity_order() -> None:
    config = yaml.safe_load(
        (METHOD_ROOT / "configs/adaptation_ladder_pilot.yaml").read_text(encoding="utf-8")
    )
    assert config["schema"] == ladder.CONFIG_SCHEMA
    assert config["mode"] == "public_development_only"
    assert config["protected_test_default"] == "locked"
    assert config["task"] == "motor_imagery"
    assert config["outer_fold"] == 0
    assert tuple(config["ladder"]["capacities"]) == ladder.CAPACITIES
    assert config["ladder"]["epochs"] > 0
    assert config["ladder"]["batch_size"] > 0


def test_output_guard_refuses_protected_path(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        ladder.output_guard(tmp_path / "protected" / "out")


def test_cached_head_model_has_expected_feature_contract() -> None:
    model = ladder.CBraModLatentClassifier(
        torch.nn.Identity(), output_dim=2, head_kind="mlp", hidden_dim=8
    )
    x = torch.randn(5, 200)
    # The cache route uses the 200-D latent as the model's input to the head.
    logits = model.head(x)
    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_cache_resolution_requires_one_public_fold_cache() -> None:
    if not (METHOD_ROOT / "checkpoints/pretrained_weights.pth").is_file():
        pytest.skip("local CBraMod assets are not available")
    alignment, _ = ladder.load_alignment_config(METHOD_ROOT / "configs/alignment_v2.yaml")
    fold = ladder.load_public_fold(alignment, task="motor_imagery", outer_fold=0)
    arrays, manifest, cache_path, manifest_path = ladder.cache_for_fold(
        fold=fold,
        alignment=alignment,
        cache_root=ladder.resolve_repo_path("comparative_methods/CBraMod/runs/feature_cache_v2"),
    )
    assert manifest["protected_test_opened"] is False
    assert cache_path.is_file() and manifest_path.is_file()
    assert arrays["features"].shape[1] == 200


def test_dry_run_writes_public_manifest(tmp_path: Path) -> None:
    if not (METHOD_ROOT / "checkpoints/pretrained_weights.pth").is_file():
        pytest.skip("local CBraMod assets are not available")
    args = type("Args", (), {
        "config": METHOD_ROOT / "configs/adaptation_ladder_pilot.yaml",
        "output_dir": tmp_path / "ladder",
        "device": "cuda:1",
        "dry_run": True,
    })()
    report = ladder.run(args)
    assert report["status"] == "pass"
    assert report["protected_test_opened"] is False
    path = tmp_path / "ladder" / "dry_run_manifest.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["feature_cache"]["schema"] == ladder.FEATURE_SCHEMA
    assert saved["fixed_budget"]["validation_used_for_selection"] is False

