"""Audit pinned comparison sources and local binary assets without loading weights."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

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
REQUIRED_OVERLAP_KEYS = {
    "eeg_fnirs_single_trial",
    "simultaneous_eeg_nirs",
    "visual_cognitive_motivation",
    "refed",
}
LOCAL_AVAILABILITY = {"downloaded", "source_bundled"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "sources/method_manifest.yaml"
    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be a mapping")
    return manifest


def main() -> int:
    failures: list[str] = []
    observations: list[str] = []

    for expected_method_id, method_root in METHOD_ROOTS.items():
        try:
            manifest = _load_manifest(method_root)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            failures.append(str(exc))
            continue

        method_id = manifest.get("method_id")
        if method_id != expected_method_id:
            failures.append(
                f"{method_root}: method_id={method_id!r}, expected {expected_method_id!r}"
            )

        overlap = manifest.get("target_corpus_overlap", {})
        missing_overlap = REQUIRED_OVERLAP_KEYS - set(overlap)
        if missing_overlap:
            failures.append(
                f"{method_root}: missing target overlap decisions {sorted(missing_overlap)}"
            )

        upstream = manifest.get("upstream", {})
        revision = str(upstream.get("revision", ""))
        local_path_value = upstream.get("local_path")
        if len(revision) != 40:
            failures.append(f"{method_root}: invalid upstream revision {revision!r}")
        if local_path_value:
            checkout = (method_root / str(local_path_value)).resolve()
            if not checkout.exists():
                failures.append(f"{method_root}: missing upstream checkout {checkout}")
            else:
                try:
                    actual_revision = _git_head(checkout)
                except subprocess.CalledProcessError as exc:
                    failures.append(f"{checkout}: git revision check failed: {exc}")
                else:
                    if actual_revision != revision:
                        failures.append(
                            f"{checkout}: revision {actual_revision}, expected {revision}"
                        )
                    else:
                        observations.append(
                            f"OK source {expected_method_id}: {actual_revision[:12]}"
                        )

        checkpoint = manifest.get("checkpoint", {})
        for artifact in checkpoint.get("artifacts", []):
            availability = artifact.get("availability")
            local_path_value = artifact.get("local_path")
            label = artifact.get("artifact_id", "<unnamed>")
            if availability not in LOCAL_AVAILABILITY:
                observations.append(
                    f"INFO asset {expected_method_id}/{label}: {availability}"
                )
                continue
            if not local_path_value:
                failures.append(
                    f"{method_root}: local artifact {label!r} has no local_path"
                )
                continue
            asset_path = method_root / str(local_path_value)
            if not asset_path.is_file():
                failures.append(f"{asset_path}: expected local asset is missing")
                continue
            expected_size = artifact.get("size_bytes")
            actual_size = asset_path.stat().st_size
            if expected_size is not None and actual_size != int(expected_size):
                failures.append(
                    f"{asset_path}: size {actual_size}, expected {expected_size}"
                )
                continue
            expected_hash = artifact.get("sha256")
            if expected_hash:
                actual_hash = _sha256(asset_path)
                if actual_hash != expected_hash:
                    failures.append(
                        f"{asset_path}: sha256 {actual_hash}, expected {expected_hash}"
                    )
                    continue
            observations.append(f"OK asset {expected_method_id}/{label}: {actual_size} B")

    for observation in observations:
        print(observation)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"PASS {len(METHOD_ROOTS)} method manifests and all required local assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
