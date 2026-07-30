#!/usr/bin/env python3
"""Build a standardized Token Physiology Atlas from v3 exports or a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.export_physiology_semantic_tokens import (
    canonical_config_sha256,
    compact_export_to_assignments,
    run as run_export,
)
from src.analysis.token_physiology_atlas import (
    build_token_physiology_atlas,
    load_token_export,
    prepare_measurement_feature_caches,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "configs"
    / "physiology_semantic_tokenizer"
    / "token_physiology_atlas.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in value.split(",") if item.strip())
    if not splits or len(set(splits)) != len(splits):
        raise argparse.ArgumentTypeError(
            "--splits must be a non-empty comma-separated list without duplicates"
        )
    invalid = sorted(set(splits) - {"train", "val", "test"})
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported split(s): {invalid}")
    return splits


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(
        item.strip().lower().lstrip(".")
        for item in value.split(",")
        if item.strip()
    )
    invalid = sorted(set(formats) - {"png", "pdf", "svg"})
    if not formats or invalid or len(set(formats)) != len(formats):
        raise argparse.ArgumentTypeError(
            "--formats must be a unique comma-separated subset of png,pdf,svg"
        )
    return formats


def _parse_export(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--export must use SPLIT=PATH syntax")
    split, raw_path = value.split("=", 1)
    split = split.strip()
    if split not in {"train", "val", "test"}:
        raise argparse.ArgumentTypeError(
            f"unsupported export split {split!r}; use train, val, or test"
        )
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("--export path must be non-empty")
    return split, Path(raw_path).expanduser()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Atlas config must be a YAML mapping")
    if config.get("schema_version") != "physiology_token_atlas_config_v1":
        raise ValueError(
            "Atlas config schema must be physiology_token_atlas_config_v1"
        )
    if config.get("input", {}).get("unit") != "canonical_robust_sd":
        raise ValueError(
            "Atlas input.unit must be canonical_robust_sd; physical voltage or "
            "concentration units cannot be inferred from this export"
        )
    if bool(config.get("features", {}).get("fnirs", {}).get("band_power", False)):
        raise ValueError(
            "Short-patch fNIRS band power is outside the Atlas contract"
        )
    return dict(config)


def _assignment_cache_valid(
    path: Path,
    *,
    split: str,
    checkpoint_sha256: str,
    config_sha256: str,
    analysis_view_contract_sha256: str,
    max_batches: int | None,
    atlas_config: Mapping[str, Any],
    measurement_cache_dir: Path,
    required_extra_fields: tuple[str, ...],
    allow_test: bool,
) -> bool:
    if not path.is_file() or not path.with_suffix(
        path.suffix + ".manifest.json"
    ).is_file():
        return False
    try:
        payload, manifest = load_token_export(
            path,
            expected_split=split,
            allow_test=allow_test,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    structurally_valid = bool(
        manifest.get("checkpoint_sha256") == checkpoint_sha256
        and manifest.get("config_sha256") == config_sha256
        and manifest.get("analysis_view_contract_sha256")
        == analysis_view_contract_sha256
        and manifest.get("max_batches") == max_batches
        and bool(manifest.get("npz_sha256"))
        and (
            manifest.get("include_patches")
            or (
                manifest.get("compacted_after_measurement_feature_extraction")
                and set(manifest.get("measurement_caches", {}))
                == {"eeg", "fnirs"}
            )
        )
        and manifest.get("include_assignment_diagnostics")
        and manifest.get("include_reconstruction_diagnostics")
        and set(required_extra_fields).issubset(
            set(manifest.get("requested_extra_fields", ()))
        )
        and all(f"{modality}_posterior" in payload for modality in ("eeg", "fnirs"))
    )
    if not structurally_valid:
        return False
    if manifest.get("include_patches"):
        return True
    try:
        prepare_measurement_feature_caches(
            path,
            config=atlas_config,
            measurement_cache_dir=measurement_cache_dir,
            expected_split=split,
            allow_test=allow_test,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _resolved_model_config(
    checkpoint_path: Path,
    model_config_path: str | None,
) -> tuple[dict[str, Any], str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config")
    if config is None:
        if not model_config_path:
            raise ValueError(
                "Checkpoint has no embedded config; --model-config is required"
            )
        config = yaml.safe_load(
            Path(model_config_path).read_text(encoding="utf-8")
        )
    if not isinstance(config, Mapping):
        raise ValueError("Resolved tokenizer configuration must be a mapping")
    resolved = dict(config)
    return resolved, canonical_config_sha256(resolved)


def _analysis_view_contract_sha256(config: Mapping[str, Any]) -> str:
    contract = {
        "cache_schema": "token_physiology_measurement_cache_v2",
        "input": config.get("input", {}),
        "features": config.get("features", {}),
        "state_fields": sorted(
            str(field)
            for field in config.get("associations", {}).get(
                "state_fields", ()
            )
        ),
    }
    return hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _exports_from_checkpoint(
    args: argparse.Namespace,
    *,
    splits: tuple[str, ...],
    cache_root: Path,
    atlas_config: Mapping[str, Any],
) -> dict[str, Path]:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    _, config_hash = _resolved_model_config(checkpoint, args.model_config)
    analysis_view_contract_hash = _analysis_view_contract_sha256(atlas_config)
    extra_fields = tuple(
        dict.fromkeys(
            str(field)
            for field in atlas_config.get("associations", {}).get(
                "state_fields", ()
            )
        )
    )
    replay_scope = (
        "full_split"
        if args.max_batches is None
        else f"first_{int(args.max_batches)}_batches"
    )
    assignment_dir = (
        cache_root
        / "assignments"
        / checkpoint_hash[:24]
        / config_hash[:24]
        / analysis_view_contract_hash[:24]
        / replay_scope
    )
    exports: dict[str, Path] = {}
    for split in splits:
        output = assignment_dir / f"{split}.npz"
        valid_cache = _assignment_cache_valid(
            output,
            split=split,
            checkpoint_sha256=checkpoint_hash,
            config_sha256=config_hash,
            analysis_view_contract_sha256=analysis_view_contract_hash,
            max_batches=args.max_batches,
            atlas_config=atlas_config,
            measurement_cache_dir=cache_root / "measurements",
            required_extra_fields=extra_fields,
            allow_test=args.allow_test,
        )
        if not valid_cache or args.force:
            replace_invalid_cache = (
                output.exists()
                or output.with_suffix(output.suffix + ".manifest.json").exists()
            )
            export_args = argparse.Namespace(
                checkpoint=str(checkpoint),
                config=args.model_config,
                split=split,
                output=str(output),
                top_k=None,
                max_batches=args.max_batches,
                include_patches=True,
                include_assignment_diagnostics=True,
                include_reconstruction_diagnostics=True,
                extra_fields=extra_fields,
                analysis_view_contract_sha256=analysis_view_contract_hash,
                allow_test=args.allow_test,
                force=bool(args.force or replace_invalid_cache),
                device=args.device,
            )
            run_export(export_args)
        exports[split] = output
    return exports


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.atlas_config).resolve()
    config = _load_config(config_path)
    configured_splits = tuple(config.get("analysis", {}).get("splits", ("train", "val")))
    splits = args.splits or configured_splits
    if "test" in splits and not args.allow_test:
        raise ValueError(
            "The protected test split is sealed; --allow-test is required "
            "independently of the YAML configuration"
        )
    output_dir = Path(args.output_dir).resolve()
    cache_root = (
        Path(args.measurement_cache_dir).resolve()
        if args.measurement_cache_dir
        else output_dir.parent / ".token_physiology_cache"
    )

    if args.checkpoint:
        exports = _exports_from_checkpoint(
            args,
            splits=splits,
            cache_root=cache_root,
            atlas_config=config,
        )
        for split, export_path in exports.items():
            references = prepare_measurement_feature_caches(
                export_path,
                config=config,
                measurement_cache_dir=cache_root / "measurements",
                expected_split=split,
                allow_test=args.allow_test,
            )
            compact_export_to_assignments(
                export_path,
                measurement_caches=references,
            )
    else:
        parsed = dict(args.export or ())
        if len(parsed) != len(args.export or ()):
            raise ValueError("Each --export split may be specified only once")
        missing = sorted(set(splits) - set(parsed))
        extra = sorted(set(parsed) - set(splits))
        if missing or extra:
            raise ValueError(
                f"--export entries must exactly match --splits; missing={missing}, "
                f"extra={extra}"
            )
        exports = {split: path.resolve() for split, path in parsed.items()}

    configured_bootstrap = int(
        config.get("profiles", {}).get("bootstrap", {}).get("iterations", 1000)
    )
    configured_null = int(
        config.get("sequence", {}).get("null", {}).get("permutations", 200)
    )
    if args.tier == "core":
        tier_bootstrap, tier_null = 0, 0
    elif args.tier == "statistical":
        tier_bootstrap, tier_null = configured_bootstrap, 0
    else:
        tier_bootstrap, tier_null = configured_bootstrap, configured_null
    bootstrap_iterations = (
        args.bootstrap_iterations
        if args.bootstrap_iterations is not None
        else tier_bootstrap
    )
    coupling_permutations = (
        args.coupling_permutations
        if args.coupling_permutations is not None
        else tier_null
    )
    run_information_ledger = bool(
        args.information_ledger or args.tier in {"statistical", "full"}
    )
    figure_formats = (
        args.formats
        if args.formats is not None
        else ("png",)
        if args.tier == "core"
        else None
    )
    result = build_token_physiology_atlas(
        exports,
        output_dir,
        config=config,
        measurement_cache_dir=cache_root / "measurements",
        allow_test=args.allow_test,
        force=args.force,
        bootstrap_iterations=bootstrap_iterations,
        coupling_permutations=coupling_permutations,
        formats=figure_formats,
        plots=not args.no_plots,
        information_ledger=run_information_ledger,
    )
    print(
        json.dumps(
            {
                "atlas": str(result),
                "splits": list(splits),
                "protected_test_opened": "test" in splits,
                "tier": args.tier,
                "bootstrap_iterations": bootstrap_iterations,
                "coupling_permutations": coupling_permutations,
                "information_ledger": run_information_ledger,
            },
            sort_keys=True,
        )
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-config",
        default=str(DEFAULT_CONFIG),
        help="Versioned Atlas analysis YAML.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--export",
        action="append",
        type=_parse_export,
        metavar="SPLIT=PATH",
        help="Analyze an existing v3 export; repeat once per requested split.",
    )
    source.add_argument(
        "--checkpoint",
        help="Replay a frozen checkpoint into checkpoint-assignment caches.",
    )
    parser.add_argument(
        "--model-config",
        help="Tokenizer config, required only if the checkpoint does not embed it.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--splits",
        type=_parse_splits,
        help="Comma-separated splits; defaults to analysis.splits in Atlas YAML.",
    )
    parser.add_argument(
        "--measurement-cache-dir",
        help="Shared cache root; defaults beside the output directory.",
    )
    parser.add_argument(
        "--tier",
        choices=("core", "statistical", "full"),
        default="core",
        help=(
            "Automation cost tier: core disables bootstrap/null, statistical "
            "adds configured bootstrap, full also runs the coupling null."
        ),
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        help="Override subject-bootstrap iterations; use 0 for a fast smoke.",
    )
    parser.add_argument(
        "--coupling-permutations",
        type=int,
        help="Override whole-window circular-shift null iterations.",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        help="Comma-separated figure formats, e.g. png,pdf,svg.",
    )
    parser.add_argument(
        "--information-ledger",
        action="store_true",
        help=(
            "Run grouped-ridge representation probes in core tier; statistical "
            "and full tiers enable them automatically."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used only for checkpoint replay (default: cpu).",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly authorize opening the protected test split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace exact Atlas/assignment artifacts.",
    )
    args = parser.parse_args(argv)
    for name in ("max_batches", "bootstrap_iterations", "coupling_permutations"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_batches == 0:
        parser.error("--max-batches must be positive when provided")
    return args


if __name__ == "__main__":
    run(parse_args())
