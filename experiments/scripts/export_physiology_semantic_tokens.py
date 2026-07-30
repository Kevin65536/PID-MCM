#!/usr/bin/env python3
"""Export versioned physiology-semantic representations from a frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.factory import create_configured_multimodal_dataloaders
from src.teachers.physical_state_teacher import PhysicalStateTeacher
from src.tokenizers.registry import create_tokenizer
import src.tokenizers  # noqa: F401


EXPORT_SCHEMA = "physiology_semantic_export_v3"


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash a resolved model/data configuration independently of its file path."""

    return hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collated_channel_names(
    value: Any,
    *,
    batch_size: int,
    expected_channels: int,
) -> np.ndarray:
    """Normalize default-collated per-sample channel names to ``[B,C]``.

    PyTorch's default collator transposes a per-sample list of channel names
    into ``list[tuple[str, ...]]`` with length ``C``.  A few custom loaders
    already emit ``[B,C]`` arrays.  The export contract accepts both while
    rejecting ambiguous layouts rather than silently losing channel identity.
    """

    array = np.asarray(value, dtype=np.str_)
    if array.shape == (expected_channels, batch_size):
        array = array.T
    if array.shape != (batch_size, expected_channels):
        raise ValueError(
            "selected_eeg_channels must collate to [B,C] or [C,B], "
            f"got {array.shape}"
        )
    return np.ascontiguousarray(array)


def _sample_aligned_field(
    value: Any,
    *,
    field_name: str,
    batch_size: int,
) -> np.ndarray:
    """Convert one requested categorical state/context field without pickle."""

    if isinstance(value, torch.Tensor):
        array = _numpy(value)
    else:
        if isinstance(value, str):
            value = [value] * batch_size
        array = np.asarray(value)
    if array.ndim == 0 or array.ndim > 2 or array.shape[0] != batch_size:
        raise ValueError(
            f"Requested export field {field_name!r} must be sample- or "
            "token-aligned with at most two dimensions; "
            f"observed shape {array.shape}"
        )
    if array.dtype.kind in {"O", "S"}:
        array = array.astype(np.str_)
    if array.dtype.kind not in {"b", "i", "u", "f", "U", "S"}:
        raise ValueError(
            f"Requested export field {field_name!r} has unsupported dtype "
            f"{array.dtype}"
        )
    return np.asarray(array)


def _assignment_diagnostics(output: Any) -> Dict[str, np.ndarray]:
    posterior = output.quantizer.posterior.float()
    safe = posterior.clamp_min(torch.finfo(posterior.dtype).tiny)
    entropy = -(posterior * safe.log()).sum(dim=-1)
    if posterior.shape[-1] >= 2:
        top_two = posterior.topk(2, dim=-1).values
        margin = top_two[..., 0] - top_two[..., 1]
    else:
        margin = torch.ones_like(entropy)
    latent_code_l2 = (output.semantic_latent - output.quantizer.quantized).square().sum(
        dim=-1
    ).sqrt()
    return {
        "posterior_entropy": _numpy(entropy),
        "posterior_top1_top2_margin": _numpy(margin),
        "latent_code_l2": _numpy(latent_code_l2),
    }


def _patch_reconstruction_diagnostics(output: Any) -> Dict[str, np.ndarray]:
    diagnostics: Dict[str, np.ndarray] = {}
    for name, reconstruction in (
        ("expected", output.reconstruction),
        ("semantic_only", output.semantic_reconstruction),
        ("hard", output.hard_reconstruction),
        ("hard_semantic_only", output.hard_semantic_reconstruction),
        ("residual_only", output.residual_reconstruction),
    ):
        mse = (reconstruction.float() - output.patches.float()).square().mean(dim=(-1, -2))
        diagnostics[f"{name}_reconstruction_mse"] = _numpy(mse)
    return diagnostics


def build_export_batch(
    outputs: Mapping[str, Any],
    teacher: Any | None,
    batch: Mapping[str, Any],
    top_k: int | None = None,
    *,
    include_patches: bool = False,
    include_assignment_diagnostics: bool = False,
    include_reconstruction_diagnostics: bool = False,
    extra_fields: Iterable[str] = (),
) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    for modality in ("eeg", "fnirs"):
        output = outputs[modality]
        payload[f"{modality}_hard_ids"] = _numpy(output.quantizer.hard_ids)
        payload[f"{modality}_semantic_latent"] = _numpy(output.semantic_latent)
        payload[f"{modality}_codebook_embedding"] = _numpy(output.quantizer.quantized)
        if top_k is None:
            payload[f"{modality}_posterior"] = _numpy(output.quantizer.posterior)
        else:
            probabilities, indices = output.quantizer.posterior.topk(top_k, dim=-1)
            payload[f"{modality}_posterior_topk_indices"] = _numpy(indices)
            payload[f"{modality}_posterior_topk_probabilities"] = _numpy(probabilities)
        payload[f"{modality}_expected_embedding"] = _numpy(output.quantizer.expected_embedding)
        payload[f"{modality}_residual"] = _numpy(output.residual)
        if include_patches:
            payload[f"{modality}_patches"] = _numpy(output.patches)
        if include_assignment_diagnostics:
            for name, values in _assignment_diagnostics(output).items():
                payload[f"{modality}_{name}"] = values
        if include_reconstruction_diagnostics:
            for name, values in _patch_reconstruction_diagnostics(output).items():
                payload[f"{modality}_{name}"] = values
        token_masks = batch.get("token_valid_mask", {})
        if modality in token_masks:
            payload[f"{modality}_token_valid_mask"] = _numpy(token_masks[modality].bool())
    if teacher is not None:
        payload["teacher_full_summary"] = _numpy(teacher.full_summary)
        payload["teacher_full_uncertainty"] = _numpy(teacher.full_uncertainty)
        payload["teacher_valid_mask"] = _numpy(teacher.valid_mask)
        payload["teacher_context_valid_mask"] = _numpy(teacher.context_valid_mask)
        for modality in ("eeg", "fnirs"):
            payload[f"{modality}_target"] = _numpy(getattr(teacher, f"{modality}_target"))
            payload[f"{modality}_target_uncertainty"] = _numpy(
                getattr(teacher, f"{modality}_uncertainty")
            )
            for entry, mask in teacher.entry_masks[modality].items():
                payload[f"{modality}_{entry}_target_valid_mask"] = _numpy(mask)
    for key in ("subject_id", "label", "crop_start_s", "has_auxiliary_target"):
        if key in batch:
            payload[key] = _numpy(batch[key])
    string_keys = (
        "sample_id", "target_sample_key", "subject_key", "dataset_id", "subject",
        "record_id", "task_namespace", "cache_entry_id", "source_name", "source_task",
        "anchor", "label_name", "dependency_group_id",
        "auxiliary_target_rejection_reason",
    )
    for key in string_keys:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, str):
            value = [value]
        payload[key] = np.asarray(value, dtype=np.str_)
    if "selected_eeg_channels" in batch:
        batch_size = int(payload["eeg_hard_ids"].shape[0])
        channel_count = int(outputs["eeg"].patches.shape[-2])
        payload["selected_eeg_channels"] = _collated_channel_names(
            batch["selected_eeg_channels"],
            batch_size=batch_size,
            expected_channels=channel_count,
        )
    batch_size = int(payload["eeg_hard_ids"].shape[0])
    for field_name in tuple(dict.fromkeys(str(name) for name in extra_fields)):
        if field_name in payload:
            continue
        if field_name not in batch:
            raise ValueError(
                f"Requested export field {field_name!r} is absent from the batch"
            )
        payload[field_name] = _sample_aligned_field(
            batch[field_name],
            field_name=field_name,
            batch_size=batch_size,
        )
    return payload


def _concatenate(chunks: Iterable[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    chunks = list(chunks)
    if not chunks:
        raise ValueError("No batches were exported")
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks[1:]):
        raise ValueError("Export batch schemas differ")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in sorted(keys)}


def _deterministic_analysis_loader(
    loader: Any,
    *,
    use_pinned_memory: bool,
) -> Any:
    """Replay every dataset item once in stable index order.

    Training loaders normally shuffle and drop the final partial batch.  Those
    semantics are correct for optimization but would make an analysis cache
    incomplete and order-dependent.
    """

    # Lightweight tests and a few programmatic consumers supply an already
    # materialized iterable of batches.  It has no sampler semantics to repair.
    if not isinstance(loader, DataLoader):
        return loader
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=bool(
            use_pinned_memory and getattr(loader, "pin_memory", False)
        ),
        collate_fn=loader.collate_fn,
    )


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _assert_targets_available(targets: Iterable[Path], *, force: bool) -> None:
    """Fail before expensive inference when an export target already exists."""

    if force:
        return
    existing = [str(target) for target in targets if target.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing export artifact(s): "
            + ", ".join(existing)
            + ". Pass --force to replace them atomically."
        )


def _stage_npz(target: Path, payload: Mapping[str, np.ndarray]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stage_text(target: Path, content: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _commit_staged_file(staged: Path, target: Path, *, force: bool) -> None:
    """Atomically publish one staged file, with no-replace semantics by default."""

    if force:
        os.replace(staged, target)
        return
    try:
        # Both paths are in the same directory.  A hard link publishes the
        # complete inode atomically and, unlike os.replace, fails if target
        # appeared after the initial availability check.
        os.link(staged, target)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite export artifact created concurrently: {target}. "
            "Pass --force to replace it atomically."
        ) from error
    else:
        staged.unlink()


def _write_export_atomically(
    output: Path,
    payload: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    """Stage both export files completely before publishing either one."""

    manifest_output = _manifest_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_targets_available((output, manifest_output), force=force)
    staged_output: Path | None = None
    staged_manifest: Path | None = None
    try:
        staged_output = _stage_npz(output, payload)
        effective_manifest = {
            **manifest,
            "npz_sha256": _sha256_path(staged_output),
        }
        staged_manifest = _stage_text(
            manifest_output,
            json.dumps(effective_manifest, indent=2, sort_keys=True) + "\n",
        )
        _commit_staged_file(staged_output, output, force=force)
        staged_output = None
        _commit_staged_file(staged_manifest, manifest_output, force=force)
        staged_manifest = None
        return effective_manifest
    finally:
        if staged_output is not None:
            staged_output.unlink(missing_ok=True)
        if staged_manifest is not None:
            staged_manifest.unlink(missing_ok=True)


def _validate_measurement_cache_reference(
    *,
    modality: str,
    reference: Mapping[str, Any],
    assignment_payload: Mapping[str, np.ndarray],
    assignment_manifest: Mapping[str, Any],
) -> None:
    """Fail closed before raw patches are removed from an assignment export."""

    required_reference_fields = {
        "path",
        "measurement_cache_key",
        "npz_sha256",
        "feature_spec_hash",
        "source_sample_order_sha256",
    }
    missing = sorted(required_reference_fields - set(reference))
    if missing:
        raise ValueError(
            f"{modality} measurement-cache reference is missing fields: {missing}"
        )
    cache_path = Path(str(reference["path"])).resolve()
    sidecar_path = cache_path.with_suffix(".manifest.json")
    if not cache_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            f"{modality} measurement cache is incomplete: "
            f"{cache_path}, {sidecar_path}"
        )
    cache_manifest = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checks = {
        "cache schema": cache_manifest.get("schema")
        == "token_physiology_measurement_cache_v2",
        "modality": cache_manifest.get("modality") == modality,
        "measurement key": cache_manifest.get("measurement_cache_key")
        == reference["measurement_cache_key"],
        "feature specification": cache_manifest.get("feature_spec_hash")
        == reference["feature_spec_hash"],
        "sample order in reference": reference["source_sample_order_sha256"]
        == assignment_manifest.get("sample_order_sha256"),
        "sample order in sidecar": cache_manifest.get(
            "source_sample_order_sha256"
        )
        == assignment_manifest.get("sample_order_sha256"),
        "reference content hash": reference["npz_sha256"]
        == _sha256_path(cache_path),
        "sidecar content hash": cache_manifest.get("npz_sha256")
        == reference["npz_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{modality} measurement-cache reference failed: {failed}"
        )

    sample_key = str(assignment_manifest["sample_key_array"])
    sample_ids = np.asarray(assignment_payload[sample_key], dtype=np.str_)
    hard_ids = np.asarray(assignment_payload[f"{modality}_hard_ids"])
    mask_key = f"{modality}_token_valid_mask"
    assignment_mask = (
        np.asarray(assignment_payload[mask_key], dtype=bool)
        if mask_key in assignment_payload
        else np.ones(hard_ids.shape, dtype=bool)
    )
    with np.load(cache_path, allow_pickle=False) as cache:
        required_arrays = {
            "token_features",
            "feature_names",
            "feature_units",
            "sample_ids",
            "token_valid_mask",
            "channel_feature_values",
            "channel_feature_valid_mask",
            "channel_valid_mask",
            "valid_sample_fraction",
        }
        missing_arrays = sorted(required_arrays - set(cache.files))
        if missing_arrays:
            raise ValueError(
                f"{modality} measurement cache is missing arrays: {missing_arrays}"
            )
        token_features = np.asarray(cache["token_features"])
        feature_names = np.asarray(cache["feature_names"])
        feature_units = np.asarray(cache["feature_units"])
        channel_features = np.asarray(cache["channel_feature_values"])
        channel_feature_valid = np.asarray(cache["channel_feature_valid_mask"])
        channel_valid = np.asarray(cache["channel_valid_mask"])
        valid_fraction = np.asarray(cache["valid_sample_fraction"])
        if (
            token_features.ndim != 3
            or token_features.shape[:2] != hard_ids.shape
            or token_features.shape[-1] != len(feature_names)
            or len(feature_names) != len(feature_units)
        ):
            raise ValueError(
                f"{modality} measurement cache token grid does not align"
            )
        if (
            channel_features.ndim != 4
            or channel_features.shape[:2] != hard_ids.shape
            or channel_feature_valid.shape != channel_features.shape
            or channel_valid.shape != channel_features.shape[:3]
            or valid_fraction.shape != channel_features.shape[:3]
        ):
            raise ValueError(
                f"{modality} channel feature grid does not align"
            )
        if not np.array_equal(
            np.asarray(cache["sample_ids"], dtype=np.str_),
            sample_ids,
        ):
            raise ValueError(
                f"{modality} measurement cache sample IDs do not align"
            )
        if not np.array_equal(
            np.asarray(cache["token_valid_mask"], dtype=bool),
            assignment_mask,
        ):
            raise ValueError(
                f"{modality} measurement cache token mask does not align"
            )
        if modality == "eeg":
            assignment_has_channels = "selected_eeg_channels" in assignment_payload
            cache_has_channels = "selected_eeg_channels" in cache.files
            if assignment_has_channels != cache_has_channels:
                raise ValueError(
                    "EEG channel identity presence differs between assignment "
                    "and measurement cache"
                )
            if assignment_has_channels and not np.array_equal(
                np.asarray(cache["selected_eeg_channels"], dtype=np.str_),
                np.asarray(
                    assignment_payload["selected_eeg_channels"],
                    dtype=np.str_,
                ),
            ):
                raise ValueError(
                    "EEG channel identities do not align with measurement cache"
                )


def compact_export_to_assignments(
    output: str | Path,
    *,
    measurement_caches: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Drop raw patches after verified checkpoint-independent caches exist.

    The compact export remains fully analyzable because its manifest records
    content-hashed measurement-cache references for both modalities.
    """

    path = Path(output).resolve()
    manifest_path = _manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Export pair is incomplete: {path}, {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("Only v3 physiology-semantic exports can be compacted")
    if (
        manifest.get("npz_sha256") is not None
        and manifest["npz_sha256"] != _sha256_path(path)
    ):
        raise ValueError("Assignment export NPZ hash does not match its manifest")
    missing_references = [
        modality for modality in ("eeg", "fnirs") if modality not in measurement_caches
    ]
    if missing_references:
        raise ValueError(
            f"Measurement-cache references missing for: {missing_references}"
        )
    with np.load(path, allow_pickle=False) as archive:
        original_payload = {key: archive[key] for key in archive.files}
    for modality in ("eeg", "fnirs"):
        _validate_measurement_cache_reference(
            modality=modality,
            reference=measurement_caches[modality],
            assignment_payload=original_payload,
            assignment_manifest=manifest,
        )
    if not bool(manifest.get("include_patches", False)):
        return path
    payload = {
        key: value
        for key, value in original_payload.items()
        if key not in {"eeg_patches", "fnirs_patches"}
    }
    before_hash = _sha256_path(path)
    sample_count = int(payload["eeg_hard_ids"].shape[0])
    compact_manifest = {
        **manifest,
        "include_patches": False,
        "raw_patches_stored": False,
        "compacted_after_measurement_feature_extraction": True,
        "precompaction_sha256": before_hash,
        "measurement_caches": {
            modality: dict(reference)
            for modality, reference in measurement_caches.items()
        },
        "sample_aligned_arrays": [
            key
            for key, value in payload.items()
            if key not in {"eeg_codebook", "fnirs_codebook"}
            and value.shape
            and value.shape[0] == sample_count
        ],
        "arrays": {key: list(value.shape) for key, value in payload.items()},
    }
    _write_export_atomically(path, payload, compact_manifest, force=True)
    return path


def run(args: argparse.Namespace) -> Path:
    split = str(getattr(args, "split", "val"))
    allow_test = bool(getattr(args, "allow_test", False))
    if split == "test" and not allow_test:
        raise ValueError(
            "The protected test split is sealed by default; pass --allow-test "
            "only for an explicitly authorized final evaluation."
        )
    output = Path(args.output).resolve()
    manifest_output = _manifest_path(output)
    force = bool(getattr(args, "force", False))
    _assert_targets_available((output, manifest_output), force=force)
    max_batches = getattr(args, "max_batches", None)
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided")
    top_k = getattr(args, "top_k", None)
    if top_k is not None and int(top_k) <= 0:
        raise ValueError("top_k must be positive when provided")

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        config_path = getattr(args, "config", None)
        if not config_path:
            raise ValueError("Checkpoint has no embedded config; --config is required")
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    device = torch.device(str(getattr(args, "device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA replay requested but CUDA is unavailable: {device}")
    model = create_tokenizer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_cfg = config.get("data", {}).get("auxiliary_target", {}) or {}
    teacher_adapter = PhysicalStateTeacher(
        target_family=str(target_cfg.get("family")),
        target_version=str(target_cfg.get("version")),
    )
    dataloader = _deterministic_analysis_loader(
        create_configured_multimodal_dataloaders(config)[split],
        use_pinned_memory=device.type == "cuda",
    )

    chunks = []
    last_outputs: Mapping[str, Any] | None = None
    include_patches = bool(getattr(args, "include_patches", False))
    include_assignment_diagnostics = bool(
        getattr(args, "include_assignment_diagnostics", False)
    )
    include_reconstruction_diagnostics = bool(
        getattr(args, "include_reconstruction_diagnostics", False)
    )
    extra_fields = tuple(
        dict.fromkeys(str(name) for name in getattr(args, "extra_fields", ()) or ())
    )
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            token_valid_masks = batch.get("token_valid_mask")
            device_masks = (
                {
                    modality: mask.to(device, non_blocking=True)
                    for modality, mask in token_valid_masks.items()
                }
                if token_valid_masks is not None
                else None
            )
            outputs = model(
                batch["eeg"].to(device, non_blocking=True),
                batch["fnirs"].to(device, non_blocking=True),
                token_valid_masks=device_masks,
            )
            last_outputs = outputs
            teacher = teacher_adapter(batch["teacher"]) if "teacher" in batch else None
            chunks.append(
                build_export_batch(
                    outputs,
                    teacher,
                    batch,
                    top_k=top_k,
                    include_patches=include_patches,
                    include_assignment_diagnostics=include_assignment_diagnostics,
                    include_reconstruction_diagnostics=include_reconstruction_diagnostics,
                    extra_fields=extra_fields,
                )
            )
    if not chunks:
        raise ValueError(f"No batches were available for split {split!r}")
    payload = _concatenate(chunks)
    assert last_outputs is not None
    for modality in ("eeg", "fnirs"):
        payload[f"{modality}_codebook"] = _numpy(
            last_outputs[modality].quantizer.codebook
        )
    sample_key_name = next(
        (
            key
            for key in ("sample_id", "cache_entry_id", "target_sample_key")
            if key in payload
        ),
        None,
    )
    if sample_key_name is None:
        raise ValueError(
            "Export requires sample_id, cache_entry_id, or target_sample_key "
            "to certify deterministic sample order"
        )
    sample_hash = hashlib.sha256(
        "\n".join(payload[sample_key_name].tolist()).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": EXPORT_SCHEMA,
        "cache_role": "checkpoint_assignment",
        "checkpoint_independent": False,
        "split": split,
        "protected_test_opened": split == "test" and allow_test,
        "sample_count": int(payload["eeg_hard_ids"].shape[0]),
        "sample_order_sha256": sample_hash,
        "sample_key_array": sample_key_name,
        "checkpoint_sha256": _sha256_path(checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "config_sha256": canonical_config_sha256(config),
        "analysis_view_contract_sha256": getattr(
            args,
            "analysis_view_contract_sha256",
            None,
        ),
        "top_k": top_k,
        "max_batches": max_batches,
        "replay_scope": (
            "full_split"
            if max_batches is None
            else f"first_{int(max_batches)}_batches"
        ),
        "include_patches": include_patches,
        "include_assignment_diagnostics": include_assignment_diagnostics,
        "include_reconstruction_diagnostics": include_reconstruction_diagnostics,
        "requested_extra_fields": list(extra_fields),
        "deterministic_replay": True,
        "shuffle": False,
        "drop_last": False,
        "device": str(device),
        "static_arrays": ["eeg_codebook", "fnirs_codebook"],
        "sample_aligned_arrays": [
            key
            for key, value in payload.items()
            if key not in {"eeg_codebook", "fnirs_codebook"}
            and value.shape
            and value.shape[0] == payload["eeg_hard_ids"].shape[0]
        ],
        "arrays": {key: list(value.shape) for key, value in payload.items()},
    }
    manifest = _write_export_atomically(output, payload, manifest, force=force)
    print(json.dumps({"output": str(output), "samples": manifest["sample_count"]}, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        help=(
            "Store only top-k posterior entries; omit for the full posterior "
            "required by Atlas soft profiles."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Positive deterministic replay limit for smoke runs.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-patches",
        action="store_true",
        help=(
            "Include tokenizer-aligned raw patches for downstream feature "
            "extraction."
        ),
    )
    parser.add_argument("--include-assignment-diagnostics", action="store_true")
    parser.add_argument("--include-reconstruction-diagnostics", action="store_true")
    parser.add_argument(
        "--extra-field",
        dest="extra_fields",
        action="append",
        default=[],
        help=(
            "Additional sample-aligned categorical state/context field to "
            "export; repeat as needed."
        ),
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly authorize opening the protected test split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace existing output and manifest files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
