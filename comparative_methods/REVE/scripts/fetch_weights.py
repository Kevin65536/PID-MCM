"""Fetch official REVE assets after the user accepts the authors' model terms."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REVISIONS = {
    "base": ("brain-bzh/reve-base", "fa9a2163a4b7c0a42c8e28b56077ef9c368944dc"),
    "large": ("brain-bzh/reve-large", "ef50ef670d25e19f41898e462af05a503562eaeb"),
    "positions": (
        "brain-bzh/reve-positions",
        "befa5b57a455b77cf302daf610c2e9ed8140bace",
    ),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        choices=("none", "base", "large", "both"),
        default="base",
        help="Encoder snapshot to fetch after accepting its Hugging Face terms.",
    )
    return parser.parse_args()


def _download(repo_id: str, revision: str, target: Path, token: str | None) -> None:
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        token=token,
    )


def main() -> int:
    args = _arguments()
    method_root = Path(__file__).resolve().parents[1]
    target_root = method_root / "checkpoints"
    target_root.mkdir(parents=True, exist_ok=True)

    position_repo, position_revision = REVISIONS["positions"]
    _download(
        position_repo,
        position_revision,
        target_root / "reve-positions",
        token=os.environ.get("HF_TOKEN"),
    )

    if args.encoder == "none":
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "REVE encoder access is gated. Review and accept the model's "
            "Responsible Use Agreement on Hugging Face, then set HF_TOKEN "
            "locally and rerun. Do not commit or share the token."
        )

    variants = ("base", "large") if args.encoder == "both" else (args.encoder,)
    for variant in variants:
        repo_id, revision = REVISIONS[variant]
        _download(repo_id, revision, target_root / f"reve-{variant}", token=token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
