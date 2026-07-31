from __future__ import annotations

from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOTS = {
    "biot": REPO_ROOT / "comparative_methods/BIOT",
    "cbramod": REPO_ROOT / "comparative_methods/CBraMod",
    "reve": REPO_ROOT / "comparative_methods/REVE",
    "normwear_eeg_fnirs_adapted": REPO_ROOT / "comparative_methods/NormWear",
    "efrm_sync_200_10_variable_channel_v1": REPO_ROOT / "comparative_methods/EFRM-PyTorch",
    "brainfusion_nvc_csp_stacking_reimplementation": (
        REPO_ROOT / "comparative_methods/BrainFusion-NVC-CSP-Stacking"
    ),
    "sta_net_eeg_fnirs_supervised": REPO_ROOT / "comparative_methods/STA-Net-PyTorch",
}
TARGET_DATASETS = {
    "eeg_fnirs_single_trial",
    "simultaneous_eeg_nirs",
    "visual_cognitive_motivation",
    "refed",
}
EXPECTED_TASKS = {
    "motor_imagery",
    "mental_arithmetic",
    "wg",
    "nback",
    "dsr",
    "visual",
    "refed_regression",
}


def _yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def test_all_fixed_methods_have_complete_b0_manifests() -> None:
    for expected_method_id, method_root in METHOD_ROOTS.items():
        manifest = _yaml(method_root / "sources/method_manifest.yaml")
        assert manifest["schema"] == "comparative_method_manifest_v1"
        assert manifest["method_id"] == expected_method_id
        assert len(str(manifest["upstream"]["revision"])) == 40
        assert set(manifest["target_corpus_overlap"]) == TARGET_DATASETS
        assert str(manifest["gate_status"]["B0_source_fixed"]).startswith("pass")
        for artifact in manifest["checkpoint"].get("artifacts", []):
            if artifact["availability"] in {"downloaded", "source_bundled"}:
                assert int(artifact["size_bytes"]) > 0
                assert len(str(artifact["sha256"])) == 64
                for trusted_file in artifact.get("trusted_code_files", []):
                    assert int(trusted_file["size_bytes"]) > 0
                    assert len(str(trusted_file["sha256"])) == 64


def test_shared_fivefold_contract_is_machine_readable_and_complete() -> None:
    protocol = _yaml(
        REPO_ROOT
        / "comparative_methods/EFRM-PyTorch/sources/lodo_full_target_fivefold_v2.yaml"
    )
    assert protocol["method"]["downstream_seeds"] == [17, 42, 73]
    assert protocol["method"]["primary_transfer_mode"] == "linear_probe"
    assert protocol["fold_registry"] == {
        "scope": "complete_target_dataset",
        "authority": "method_neutral_shared_comparison_registry",
        "exact_same_registry_required_for_method_ranking": True,
        "protected_test_default": "locked",
        "outer_seed": 42,
        "inner_seed": "43_plus_outer_index",
        "outer_folds": 5,
        "inner_folds": 3,
        "selected_inner_fold": 0,
    }
    assert set(protocol["tasks"]) == EXPECTED_TASKS
    assert protocol["downstream"]["protocols"]["strict_cross_subject"]["role"] == "primary"
    assert protocol["reporting"]["classification_primary"] == "macro_f1"
    assert protocol["reporting"]["regression_primary"] == "native_ccc"


def test_comparison_weights_upstreams_and_runs_cannot_be_tracked_accidentally() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "comparative_methods"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    forbidden_suffixes = (".ckpt", ".pt", ".pth", ".safetensors")
    assert not [path for path in tracked if path.endswith(forbidden_suffixes)]
    assert not [path for path in tracked if "/upstream/" in path]
    retained_sta_net_evidence = (
        "comparative_methods/STA-Net-PyTorch/runs/fivefold/"
        "20260727_sta_net_no_artifact_mask_converged_5fold_v1/"
    )
    unexpected_runs = [
        path
        for path in tracked
        if "/runs/" in path
        and not path.endswith("/.gitkeep")
        and not path.startswith(retained_sta_net_evidence)
    ]
    assert not unexpected_runs

    ignored_examples = [
        "comparative_methods/BIOT/upstream/model/biot.py",
        "comparative_methods/CBraMod/checkpoints/pretrained_weights.pth",
        "comparative_methods/REVE/checkpoints/reve-base/model.safetensors",
        "comparative_methods/NormWear/runs/formal/result.json",
    ]
    for path in ignored_examples:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, path
