#!/usr/bin/env python3
"""Evaluate one EFRM checkpoint on the complete public-validation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from efrm_pytorch.data import (
    CachedEFRMPretrainDataset,
    EFRMPairedWindowAdapter,
    EFRMSyncPretrainDataset,
)
from efrm_pytorch.model import EFRMSyncModel
from efrm_pytorch.pretraining_analysis import analyze_alignment_evidence
from efrm_pytorch.training import evaluate_pretrain_batch, move_batch
from efrm_pytorch.visualization import export_alignment_evidence, retrieval_metrics
from train_pretrain import _build_boundary, _loader


def _positive_vs_negative_auc(
    positive: np.ndarray, negative: np.ndarray
) -> float:
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


def _two_item_group_pairing_test(
    cosine: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int = 100_000,
    seed: int = 20260724,
) -> dict[str, object]:
    deltas = []
    pair_count = 0
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 2:
            raise ValueError(
                "within-record-condition pairing test requires two trials per group"
            )
        values = cosine[np.ix_(indices, indices)]
        deltas.append(
            (values[0, 0] + values[1, 1])
            - (values[0, 1] + values[1, 0])
        )
        pair_count += 2
    delta = np.asarray(deltas, dtype=np.float64)
    observed = float(delta.sum())
    rng = np.random.default_rng(seed)
    exceed = 0
    batch_size = 10_000
    for start in range(0, permutations, batch_size):
        size = min(batch_size, permutations - start)
        signs = (
            rng.integers(0, 2, size=(size, len(delta)), dtype=np.int8) * 2 - 1
        )
        exceed += int(np.count_nonzero(signs @ delta >= observed))
    return {
        "paired_identity_minus_swap_cosine_mean": observed / pair_count,
        "paired_sign_flip_p_one_sided": (exceed + 1) / (permutations + 1),
        "paired_sign_flip_permutations": permutations,
        "paired_sign_flip_seed": seed,
    }


def _stratified_metrics(
    cosine: np.ndarray, metadata: list[dict[str, object]]
) -> dict[str, object]:
    size = len(cosine)
    off_diagonal = ~np.eye(size, dtype=bool)
    positive = np.diag(cosine)
    subjects = np.asarray([str(row["subject"]) for row in metadata])
    conditions = np.asarray([str(row["condition"]) for row in metadata])
    records = np.asarray([str(row["join_key"]) for row in metadata])
    masks = {
        "same_subject_negative": off_diagonal
        & (subjects[:, None] == subjects[None, :]),
        "cross_subject_negative": subjects[:, None] != subjects[None, :],
        "same_condition_negative": off_diagonal
        & (conditions[:, None] == conditions[None, :]),
        "different_condition_negative": conditions[:, None] != conditions[None, :],
        "same_record_negative": off_diagonal
        & (records[:, None] == records[None, :]),
        "same_record_same_condition_negative": off_diagonal
        & (records[:, None] == records[None, :])
        & (conditions[:, None] == conditions[None, :]),
    }
    strata = {}
    for name, mask in masks.items():
        negative = cosine[mask]
        strata[name] = {
            "count": int(mask.sum()),
            "cosine_mean": float(negative.mean()),
            "positive_vs_negative_auc": _positive_vs_negative_auc(
                positive, negative
            ),
        }

    def grouped_retrieval(labels: np.ndarray) -> tuple[dict[str, object], dict[str, object]]:
        by_group: dict[str, object] = {}
        pair_count = 0
        eeg_top1_count = 0.0
        fnirs_top1_count = 0.0
        eeg_reciprocal_rank_sum = 0.0
        fnirs_reciprocal_rank_sum = 0.0
        chance_top1_count = 0.0
        for label in sorted(set(labels)):
            indices = np.flatnonzero(labels == label)
            if len(indices) < 2:
                continue
            values = retrieval_metrics(cosine[np.ix_(indices, indices)])
            row = {
                "pair_count": len(indices),
                "chance_top1": 1.0 / len(indices),
                "eeg_to_fnirs_top1": values["eeg_to_fnirs"]["top1"],
                "fnirs_to_eeg_top1": values["fnirs_to_eeg"]["top1"],
                "eeg_to_fnirs_mrr": values["eeg_to_fnirs"]["mrr"],
                "fnirs_to_eeg_mrr": values["fnirs_to_eeg"]["mrr"],
            }
            by_group[str(label)] = row
            pair_count += len(indices)
            chance_top1_count += 1.0
            eeg_top1_count += row["eeg_to_fnirs_top1"] * len(indices)
            fnirs_top1_count += row["fnirs_to_eeg_top1"] * len(indices)
            eeg_reciprocal_rank_sum += row["eeg_to_fnirs_mrr"] * len(indices)
            fnirs_reciprocal_rank_sum += row["fnirs_to_eeg_mrr"] * len(indices)
        summary = {
            "group_count": len(by_group),
            "pair_count": pair_count,
            "chance_top1": chance_top1_count / pair_count,
            "eeg_to_fnirs_top1": eeg_top1_count / pair_count,
            "fnirs_to_eeg_top1": fnirs_top1_count / pair_count,
            "eeg_to_fnirs_mrr": eeg_reciprocal_rank_sum / pair_count,
            "fnirs_to_eeg_mrr": fnirs_reciprocal_rank_sum / pair_count,
        }
        return summary, by_group

    subject_summary, within_subject = grouped_retrieval(subjects)
    record_summary, _ = grouped_retrieval(records)
    record_condition_labels = np.asarray([
        f"{record}|{condition}"
        for record, condition in zip(records, conditions, strict=True)
    ])
    record_condition_summary, _ = grouped_retrieval(record_condition_labels)
    record_condition_summary.update(
        _two_item_group_pairing_test(cosine, record_condition_labels)
    )
    return {
        "negative_strata": strata,
        "grouped_retrieval": {
            "within_subject": subject_summary,
            "within_record": record_summary,
            "within_record_same_condition": record_condition_summary,
        },
        "within_subject_retrieval": within_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best_alignment.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("checkpoint evaluation requires CUDA")
    torch.cuda.set_device(device)

    cache_root = Path(config["data"]["cache_root"])
    if not cache_root.is_absolute():
        cache_root = REPO_ROOT / cache_root
    config["data"]["cache_root"] = str(cache_root)
    dataset = EFRMSyncPretrainDataset(
        cache_root=cache_root,
        dataset_ids=tuple(config["data"]["dataset_ids"]),
        task_namespaces=tuple(config["data"].get("task_namespaces", ())),
        seed=int(config["training"]["seed"]),
        adapter=EFRMPairedWindowAdapter(
            duration_s=float(config["data"]["window_duration_s"]),
            eeg_rate_hz=float(config["data"]["eeg_sample_rate_hz"]),
            fnirs_rate_hz=float(config["data"]["fnirs_sample_rate_hz"]),
            eeg_patch_samples=int(config["model"]["eeg_patch_samples"]),
            fnirs_patch_samples=int(config["model"]["fnirs_patch_samples"]),
            require_full_analysis_support=bool(
                config["data"]["require_full_analysis_support"]
            ),
        ),
    )
    _, boundary = _build_boundary(config, dataset)
    boundary_manifest = boundary.manifest()
    validation_indices = boundary.indices_for(dataset, "validation")
    tensor_cache = (
        METHOD_ROOT / "runs/cache" /
        f"tensors_{boundary_manifest['boundary_sha256']}"
    )
    validation = CachedEFRMPretrainDataset(
        dataset, validation_indices, tensor_cache, build=False
    )
    inventory_cache = (
        METHOD_ROOT / "runs/cache" /
        f"inventory_{boundary_manifest['boundary_sha256']}.json"
    )
    loader, sampler = _loader(
        validation,
        batch_size=int(config["training"]["effective_batch_size"]),
        seed=int(config["training"]["seed"]) + 1,
        workers=args.num_workers,
        inventory_diverse=True,
        inventory_cache_path=inventory_cache,
    )

    model_config = config["model"]
    model = EFRMSyncModel(
        eeg_patch_samples=int(model_config["eeg_patch_samples"]),
        fnirs_patch_samples=int(model_config["fnirs_patch_samples"]),
        mask_ratio=float(model_config["mask_ratio"]),
        embed_dim=int(model_config["embedding_dim"]),
        depth=int(model_config["encoder_depth"]),
        num_heads=int(model_config["encoder_heads"]),
        decoder_embed_dim=int(model_config["decoder_embedding_dim"]),
        decoder_depth=int(model_config["decoder_depth"]),
        decoder_num_heads=int(model_config["decoder_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        clip_logit_multiplier=float(model_config["clip_logit_multiplier"]),
        activation_checkpointing=False,
    ).to(device)
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["boundary_sha256"] != boundary_manifest["boundary_sha256"]:
        raise RuntimeError("checkpoint boundary does not match public validation")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sampler.set_epoch(int(checkpoint["epoch"]))

    eeg_embeddings: list[torch.Tensor] = []
    fnirs_embeddings: list[torch.Tensor] = []
    metadata: list[dict[str, object]] = []
    with torch.no_grad():
        for raw_batch in loader:
            _, evidence = evaluate_pretrain_batch(
                model,
                move_batch(raw_batch, device),
                chunk_size=args.chunk_size,
                amp_dtype=torch.bfloat16,
                eeg_reconstruction_weight=float(
                    config["loss"]["eeg_reconstruction_weight"]
                ),
                fnirs_reconstruction_weight=float(
                    config["loss"]["fnirs_reconstruction_weight"]
                ),
                clip_alignment_weight=float(config["loss"]["clip_alignment_weight"]),
            )
            eeg_embeddings.append(evidence["eeg_embedding"].cpu())
            fnirs_embeddings.append(evidence["fnirs_embedding"].cpu())
            metadata.extend([
                {key: raw_batch[key][index] for key in (
                    "sample_id", "dataset_id", "subject", "record_id", "join_key",
                    "task_namespace", "condition", "crop_start_s", "duration_s",
                )}
                for index in range(len(raw_batch["sample_id"]))
            ])

    output_dir = run_dir / "analysis/checkpoints"
    evidence_path = export_alignment_evidence(
        output_dir,
        eeg_embeddings=torch.cat(eeg_embeddings),
        fnirs_embeddings=torch.cat(fnirs_embeddings),
        metadata=metadata,
        logit_multiplier=float(model_config["clip_logit_multiplier"]),
        filename=f"{checkpoint_path.stem}_full_validation_alignment_evidence.npz",
    )
    metrics, arrays, evidence_metadata = analyze_alignment_evidence(evidence_path)
    result = {
        "schema": "efrm_checkpoint_full_validation_alignment_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "boundary_sha256": boundary_manifest["boundary_sha256"],
        "protected_test_opened": False,
        "alignment": metrics,
        "stratified_alignment": _stratified_metrics(
            np.asarray(arrays["cosine_similarity"], dtype=np.float64),
            evidence_metadata,
        ),
    }
    destination = output_dir / f"{checkpoint_path.stem}_full_validation_metrics.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "checkpoint": checkpoint_path.name,
        "epoch_1_based": int(checkpoint["epoch"]) + 1,
        "pair_count": metrics["pair_count"],
        "eeg_to_fnirs_top1": metrics["eeg_to_fnirs"]["top1"],
        "fnirs_to_eeg_top1": metrics["fnirs_to_eeg"]["top1"],
        "positive_vs_all_negative_auc": metrics["positive_vs_all_negative_auc"],
        "permutation_p": metrics["identity_pair_permutation_p_one_sided"],
        "metrics": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
