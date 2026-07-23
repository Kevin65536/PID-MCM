#!/usr/bin/env python3
"""Evaluate E2 frozen target probes and prototype signatures on development subjects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.export_physiology_semantic_tokens import (  # noqa: E402
    _concatenate,
    build_export_batch,
)
from src.analysis.semantic_token_evaluation import (  # noqa: E402
    bootstrap_subject_mean,
    fit_grouped_ridge_probe,
    match_prototype_signatures,
    prototype_signatures,
    r2_per_coordinate,
    subject_level_r2,
)
from src.data.factory import create_configured_multimodal_dataloaders  # noqa: E402
from src.teachers.physical_state_teacher import PhysicalStateTeacher  # noqa: E402
from src.tokenizers.registry import create_tokenizer  # noqa: E402
import src.tokenizers  # noqa: E402,F401


SCHEMA = "physiology_semantic_e2_evaluation_v1"
COORDINATES = {
    "eeg": ("r_mean", "r_slope", "r_logvar", "s_mean", "s_slope", "s_logvar"),
    "fnirs": (
        "delta_f_mean", "delta_hbo_mean", "delta_hb_mean",
        "delta_f_slope", "delta_hbo_slope", "delta_hb_slope",
        "delta_f_logvar", "delta_hbo_logvar", "delta_hb_logvar",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_prototype_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    """Write the registered Parquet artifact when Arrow is available."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows)), path)
    return True


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _collect_split(
    model: torch.nn.Module,
    loader: Any,
    teacher_adapter: PhysicalStateTeacher,
    device: torch.device,
    *,
    max_batches: int | None,
) -> dict[str, np.ndarray]:
    chunks = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _move(batch, device)
            outputs = model(
                batch["eeg"], batch["fnirs"], token_valid_masks=batch.get("token_valid_mask")
            )
            teacher = teacher_adapter(batch["teacher"])
            chunks.append(build_export_batch(outputs, teacher, batch))
    return _concatenate(chunks)


def _deterministic_evaluation_loader(loader: Any) -> DataLoader:
    """Reuse the frozen dataset without training shuffle or drop-last behavior."""

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )


def _representation_arrays(payload: Mapping[str, np.ndarray], modality: str) -> dict[str, np.ndarray]:
    hard_ids = np.asarray(payload[f"{modality}_hard_ids"], dtype=np.int64)
    posterior = np.asarray(payload[f"{modality}_posterior"], dtype=np.float64)
    codebook_size = posterior.shape[-1]
    return {
        "continuous_latent": np.asarray(payload[f"{modality}_semantic_latent"], dtype=np.float64),
        "hard_id": np.eye(codebook_size, dtype=np.float64)[hard_ids],
        "posterior": posterior,
        "codebook_embedding": np.asarray(
            payload[f"{modality}_codebook_embedding"], dtype=np.float64
        ),
    }


def _coordinate_indices(
    config: Mapping[str, Any],
    modality: str,
    *,
    optional: bool = False,
) -> tuple[list[str], np.ndarray]:
    evaluation = config.get("validation", {}).get("e2_evaluation", {})
    field = "optional_signature_coordinates" if optional else "signature_coordinates"
    configured = evaluation.get(field, {}).get(modality)
    if optional and configured is None:
        configured = ()
    if configured is None:
        configured = config.get("loss", {}).get("entry_routing", {}).get("prototype", {}).get(modality, ())
    configured = [str(value) for value in configured]
    unknown = sorted(set(configured) - set(COORDINATES[modality]))
    if unknown:
        raise ValueError(f"Unknown {modality} E2 signature coordinates: {unknown}")
    return configured, np.asarray([COORDINATES[modality].index(name) for name in configured], dtype=int)


def _flatten_for_probe(
    payload: Mapping[str, np.ndarray],
    modality: str,
    representation: np.ndarray,
    coordinate_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(payload[f"{modality}_prototype_target_valid_mask"], dtype=bool)
    if f"{modality}_token_valid_mask" in payload:
        valid &= np.asarray(payload[f"{modality}_token_valid_mask"], dtype=bool)
    target = np.asarray(payload[f"{modality}_target"], dtype=np.float64)[..., coordinate_indices]
    subjects = np.asarray(payload["subject_key"], dtype=np.str_)
    repeated_subjects = np.repeat(subjects[:, None], valid.shape[1], axis=1)
    return representation[valid], target[valid], repeated_subjects[valid], valid


def _hard_null_distribution(
    probe: Any,
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    standardized_train = (train_features - probe.feature_mean) / probe.feature_scale
    standardized_validation = (validation_features - probe.feature_mean) / probe.feature_scale
    values = []
    for _ in range(iterations):
        shuffled = train_target[rng.permutation(len(train_target))]
        model = Ridge(alpha=probe.alpha).fit(standardized_train, shuffled)
        prediction = model.predict(standardized_validation)
        values.append(float(np.mean(r2_per_coordinate(validation_target, prediction))))
    return values


def _evaluate_probe_family(
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    modality: str,
    coordinate_names: Sequence[str],
    coordinate_indices: np.ndarray,
    *,
    alphas: Sequence[float],
    bootstrap_iterations: int,
    null_iterations: int,
    seed: int,
) -> dict[str, Any]:
    train_representations = _representation_arrays(train, modality)
    validation_representations = _representation_arrays(validation, modality)
    results: dict[str, Any] = {}
    for representation_name in train_representations:
        train_x, train_y, train_subjects, _ = _flatten_for_probe(
            train, modality, train_representations[representation_name], coordinate_indices
        )
        validation_x, validation_y, validation_subjects, _ = _flatten_for_probe(
            validation, modality, validation_representations[representation_name], coordinate_indices
        )
        probe, cv = fit_grouped_ridge_probe(train_x, train_y, train_subjects, alphas=alphas)
        prediction = probe.predict(validation_x)
        coordinate_r2 = r2_per_coordinate(validation_y, prediction)
        by_subject = subject_level_r2(validation_y, prediction, validation_subjects)
        result = {
            "coordinate_names": list(coordinate_names),
            "coordinate_r2": coordinate_r2.tolist(),
            "mean_r2": float(np.mean(coordinate_r2)),
            "subject_r2": by_subject,
            "subject_bootstrap": bootstrap_subject_mean(
                by_subject,
                iterations=bootstrap_iterations,
                seed=seed + len(results),
            ),
            "probe_selection": cv,
            "train_token_count": int(len(train_x)),
            "validation_token_count": int(len(validation_x)),
        }
        if representation_name == "hard_id":
            null = _hard_null_distribution(
                probe, train_x, train_y, validation_x, validation_y,
                iterations=null_iterations, seed=seed,
            )
            result["shuffled_target_null"] = {
                "iterations": null_iterations,
                "mean_r2_q95": float(np.quantile(null, 0.95)),
                "observed_above_q95": bool(
                    float(np.mean(coordinate_r2)) > np.quantile(null, 0.95)
                ),
            }
        results[representation_name] = result
    return results


def _run_metadata(run_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / "checkpoints" / "last.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Checkpoint has no resolved config: {checkpoint_path}")
    return checkpoint_path, dict(config), checkpoint


def evaluate_run(
    run_dir: Path,
    output_dir: Path,
    *,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    checkpoint_path, config, checkpoint = _run_metadata(run_dir)
    e2_cfg = config.get("validation", {}).get("e2_evaluation", {})
    row = str(e2_cfg.get("row", config.get("validation", {}).get("e2_row", "unregistered")))
    seed = int(config.get("training", {}).get("seed", 0))
    label = f"{row}_seed{seed}"
    model = create_tokenizer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_cfg = config.get("data", {}).get("auxiliary_target", {}) or {}
    teacher_adapter = PhysicalStateTeacher(
        target_family=str(target_cfg.get("family")),
        target_version=str(target_cfg.get("version")),
    ).to(device)
    loaders = create_configured_multimodal_dataloaders(config)
    train = _collect_split(
        model,
        _deterministic_evaluation_loader(loaders["train"]),
        teacher_adapter,
        device,
        max_batches=max_batches,
    )
    validation = _collect_split(
        model,
        _deterministic_evaluation_loader(loaders["val"]),
        teacher_adapter,
        device,
        max_batches=max_batches,
    )
    representation_dir = output_dir / "representations"
    representation_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(representation_dir / f"{label}_train.npz", **train)
    np.savez_compressed(representation_dir / f"{label}_validation.npz", **validation)

    alphas = e2_cfg.get("ridge_alphas", (0.1, 1.0, 10.0, 100.0))
    bootstrap_iterations = int(e2_cfg.get("bootstrap_iterations", 2000))
    null_iterations = int(e2_cfg.get("null_iterations", 256))
    run_result: dict[str, Any] = {
        "row": row,
        "seed": seed,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "representations": {},
        "optional_representations": {},
        "prototype_signatures": {},
    }
    prototype_rows: list[dict[str, Any]] = []
    for modality in ("eeg", "fnirs"):
        coordinate_names, coordinate_indices = _coordinate_indices(config, modality)
        modality_results = _evaluate_probe_family(
            train, validation, modality, coordinate_names, coordinate_indices,
            alphas=alphas,
            bootstrap_iterations=bootstrap_iterations,
            null_iterations=null_iterations,
            seed=seed,
        )
        optional_names, optional_indices = _coordinate_indices(
            config, modality, optional=True
        )
        if optional_names:
            run_result["optional_representations"][modality] = _evaluate_probe_family(
                train, validation, modality, optional_names, optional_indices,
                alphas=alphas,
                bootstrap_iterations=bootstrap_iterations,
                null_iterations=null_iterations,
                seed=seed + 1000,
            )

        prototype_names = list(coordinate_names) + [
            name for name in optional_names if name not in coordinate_names
        ]
        prototype_indices = np.asarray(
            [COORDINATES[modality].index(name) for name in prototype_names], dtype=int
        )
        train_target = np.asarray(train[f"{modality}_target"], dtype=np.float64)[..., prototype_indices]
        validation_target = np.asarray(validation[f"{modality}_target"], dtype=np.float64)[..., prototype_indices]
        train_mask = np.asarray(train[f"{modality}_prototype_target_valid_mask"], dtype=bool)
        validation_mask = np.asarray(validation[f"{modality}_prototype_target_valid_mask"], dtype=bool)
        codebook_size = int(train[f"{modality}_posterior"].shape[-1])
        train_signatures, train_counts = prototype_signatures(
            train[f"{modality}_hard_ids"], train_target, train_mask, codebook_size=codebook_size
        )
        validation_signatures, validation_counts = prototype_signatures(
            validation[f"{modality}_hard_ids"], validation_target, validation_mask,
            codebook_size=codebook_size,
        )
        run_result["prototype_signatures"][modality] = {
            "coordinate_names": prototype_names,
            "train": train_signatures.tolist(),
            "train_counts": train_counts.tolist(),
            "validation": validation_signatures.tolist(),
            "validation_counts": validation_counts.tolist(),
            "within_run_train_validation_match": match_prototype_signatures(
                train_signatures, validation_signatures, train_counts, validation_counts
            ),
        }
        for split_name, signatures, counts in (
            ("train", train_signatures, train_counts),
            ("validation", validation_signatures, validation_counts),
        ):
            for code in range(codebook_size):
                prototype_rows.append({
                    "row": row,
                    "seed": seed,
                    "modality": modality,
                    "split": split_name,
                    "code": code,
                    "count": int(counts[code]),
                    **{
                        coordinate: float(signatures[code, index])
                        for index, coordinate in enumerate(prototype_names)
                    },
                })
        run_result["representations"][modality] = modality_results

    _write_csv(output_dir / "per_run" / label / "prototype_signatures.csv", prototype_rows)
    _write_json(output_dir / "per_run" / label / "state_decoding.json", run_result)
    return run_result


def _aggregate(results: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    stability: dict[str, Any] = {"schema": "physiology_semantic_e2_prototype_stability_v1", "pairs": []}
    for result in results:
        for modality in ("eeg", "fnirs"):
            hard = result["representations"][modality]["hard_id"]
            rows.append({
                "row": result["row"],
                "seed": result["seed"],
                "modality": modality,
                "hard_token_mean_r2": hard["mean_r2"],
                "hard_token_null_q95": hard["shuffled_target_null"]["mean_r2_q95"],
                "hard_token_above_null": hard["shuffled_target_null"]["observed_above_q95"],
                "continuous_latent_mean_r2": result["representations"][modality]["continuous_latent"]["mean_r2"],
                "posterior_mean_r2": result["representations"][modality]["posterior"]["mean_r2"],
                "codebook_embedding_mean_r2": result["representations"][modality]["codebook_embedding"]["mean_r2"],
                "optional_hard_token_mean_r2": (
                    result.get("optional_representations", {}).get(modality, {}).get("hard_id", {}).get("mean_r2")
                ),
            })
    for left_index, left in enumerate(results):
        for right in results[left_index + 1:]:
            if left["row"] != right["row"]:
                continue
            for modality in ("eeg", "fnirs"):
                left_signature = left["prototype_signatures"][modality]
                right_signature = right["prototype_signatures"][modality]
                stability["pairs"].append({
                    "row": left["row"],
                    "modality": modality,
                    "left_seed": left["seed"],
                    "right_seed": right["seed"],
                    **match_prototype_signatures(
                        np.asarray(left_signature["train"]),
                        np.asarray(right_signature["train"]),
                        np.asarray(left_signature["train_counts"]),
                        np.asarray(right_signature["train_counts"]),
                    ),
                })
    return rows, stability


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for E2 evaluation but unavailable")
    results = [
        evaluate_run(Path(run).resolve(), output_dir, device=device, max_batches=args.max_batches)
        for run in args.run
    ]
    objective_rows, stability = _aggregate(results)
    all_prototype_rows: list[dict[str, Any]] = []
    for result in results:
        for modality, payload in result["prototype_signatures"].items():
            names = payload["coordinate_names"]
            for split_name in ("train", "validation"):
                signatures = np.asarray(payload[split_name])
                counts = np.asarray(payload[f"{split_name}_counts"])
                for code in range(len(signatures)):
                    all_prototype_rows.append({
                        "row": result["row"], "seed": result["seed"],
                        "modality": modality, "split": split_name,
                        "code": code, "count": int(counts[code]),
                        **{name: float(signatures[code, index]) for index, name in enumerate(names)},
                    })
    _write_csv(output_dir / "prototype_signatures.csv", all_prototype_rows)
    parquet_written = _write_prototype_parquet(
        output_dir / "prototype_signatures.parquet", all_prototype_rows
    )
    if args.require_parquet and not parquet_written:
        raise RuntimeError("--require-parquet needs pyarrow; CSV evidence was written but is not a substitute")
    _write_json(output_dir / "state_decoding.json", {"schema": SCHEMA, "runs": results})
    _write_json(output_dir / "prototype_stability.json", stability)
    _write_csv(output_dir / "objective_ablation.csv", objective_rows)

    gradient_rows = []
    for result in results:
        audit_path = Path(result["run_dir"]) / "diagnostics" / "gradient_entry_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
        gradient_rows.append({
            "row": result["row"], "seed": result["seed"], "audit": audit,
        })
    _write_json(output_dir / "gradient_entry_audit.json", {
        "schema": "physiology_semantic_e2_gradient_audit_collection_v1",
        "runs": gradient_rows,
    })
    _write_json(output_dir / "manifest.json", {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(results),
        "rows": sorted({str(result["row"]) for result in results}),
        "protected_test_opened": False,
        "prototype_parquet_written": parquet_written,
        "artifacts": [
            "state_decoding.json", "prototype_signatures.csv",
            *( ["prototype_signatures.parquet"] if parquet_written else [] ),
            "prototype_stability.json", "objective_ablation.csv", "gradient_entry_audit.json",
        ],
    })
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Completed T0/T1/T2 run directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--require-parquet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps({"output_dir": str(output)}, sort_keys=True))
