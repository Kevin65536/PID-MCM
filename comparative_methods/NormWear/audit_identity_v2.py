"""Read-only identity and checkpoint-structure audit for NormWear adapter v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
CONFIG_SCHEMA = "normwear_adapter_alignment_v2"
METHOD_ID = "normwear_eeg_fnirs_adapted"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def audit_identity(method_root: Path = METHOD_ROOT) -> dict[str, Any]:
    manifest = load_yaml(method_root / "sources/method_manifest.yaml")
    config = load_yaml(method_root / "configs/alignment_v2.yaml")
    if manifest.get("method_id") != METHOD_ID or config.get("method_id") != METHOD_ID:
        raise ValueError("NormWear method identity drifted")
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("NormWear alignment config schema drifted")
    if config.get("protected_test_default") != "locked":
        raise PermissionError("protected test must remain locked")

    upstream = method_root / str(manifest["upstream"]["local_path"])
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != str(manifest["upstream"]["revision"]):
        raise ValueError(f"NormWear upstream revision drifted: {revision}")
    dirty = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("NormWear upstream checkout is dirty")

    source_reports: list[dict[str, Any]] = []
    for item in manifest["source_fidelity"]["files"]:
        path = method_root / str(item["local_path"])
        actual = sha256_file(path)
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"NormWear source size drifted: {path}")
        if actual != str(item["sha256"]):
            raise ValueError(f"NormWear source hash drifted: {path}")
        source_reports.append(
            {"local_path": str(item["local_path"]), "sha256": actual}
        )

    artifact = next(
        item
        for item in manifest["checkpoint"]["artifacts"]
        if item["artifact_id"] == "normwear_pretrain"
    )
    checkpoint_path = method_root / str(artifact["local_path"])
    if checkpoint_path.stat().st_size != int(artifact["size_bytes"]):
        raise ValueError("NormWear checkpoint size drifted")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != str(artifact["sha256"]):
        raise ValueError("NormWear checkpoint hash drifted")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if type(state).__name__ != "OrderedDict" or len(state) != 261:
        raise ValueError("unexpected NormWear checkpoint container")
    expected_shapes = {
        "cls_token": (1, 1, 768),
        "pos_embed": (1, 560, 768),
        "patch_embed.proj.weight": (768, 3, 9, 5),
        "norm.weight": (768,),
        "decoder_embed.weight": (512, 768),
    }
    for key, shape in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"unexpected NormWear tensor shape for {key}")
    if not all(
        bool(torch.isfinite(value).all())
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    ):
        raise ValueError("NormWear checkpoint contains non-finite tensors")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from modules.normwear import NormWear  # type: ignore[import-not-found]

    model = NormWear(
        img_size=(387, 65),
        patch_size=(9, 5),
        mask_scheme="random",
        mask_prob=0.8,
        use_cwt=True,
        nvar=4,
        comb_freq=False,
    )
    strict_result = model.load_state_dict(state, strict=True)
    if strict_result.missing_keys or strict_result.unexpected_keys:
        raise ValueError("NormWear checkpoint does not strictly match the pinned model")

    tasks = config["tasks"]
    supported = [task for task, cell in tasks.items() if cell["supported"]]
    unsupported = [task for task, cell in tasks.items() if not cell["supported"]]
    if supported != [
        "motor_imagery",
        "mental_arithmetic",
        "wg",
        "nback",
        "dsr",
        "visual",
    ]:
        raise ValueError("NormWear supported cell registration drifted")
    if unsupported != ["refed_regression"]:
        raise ValueError("NormWear unsupported cell registration drifted")

    return {
        "schema": "normwear_identity_audit_v2",
        "status": "pass",
        "method_id": METHOD_ID,
        "upstream_revision": revision,
        "upstream_clean": True,
        "source_files": source_reports,
        "checkpoint": {
            "artifact_id": "normwear_pretrain",
            "sha256": checkpoint_hash,
            "size_bytes": checkpoint_path.stat().st_size,
            "container_type": type(state).__name__,
            "tensor_entry_count": len(state),
            "tensor_element_count": sum(int(value.numel()) for value in state.values()),
            "expected_shapes": {key: list(shape) for key, shape in expected_shapes.items()},
            "weights_only_load": True,
            "strict_model_match": True,
        },
        "cell_registration": {
            "supported": supported,
            "unsupported": unsupported,
            "protected_test_opened": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_identity()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
