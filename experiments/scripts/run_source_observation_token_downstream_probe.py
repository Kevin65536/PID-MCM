#!/usr/bin/env python
"""Lightweight downstream probes for exported source/observation token sequences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


BRANCHES = (
    "eeg_source_tokens",
    "fnirs_source_tokens",
    "eeg_observation_tokens",
    "fnirs_observation_tokens",
)

TASKS = {
    "nback_vs_wg": {
        "source_task": None,
        "label_names": ("nback", "wg"),
    },
    "mental_arithmetic_bl_vs_ma": {
        "source_task": "mental_arithmetic",
        "label_names": ("BL", "MA"),
    },
    "motor_lmi_vs_rmi": {
        "source_task": "motor_imagery",
        "label_names": ("LMI", "RMI"),
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_split(run_dir: Path, split: str) -> Dict[str, np.ndarray]:
    path = run_dir / "tokens" / f"{split}_tokens.npz"
    if not path.exists():
        raise FileNotFoundError(f"Token split not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def available_splits(run_dir: Path) -> list[str]:
    token_dir = run_dir / "tokens"
    splits = []
    for path in sorted(token_dir.glob("*_tokens.npz")):
        splits.append(path.name[: -len("_tokens.npz")])
    preferred = {"train": 0, "val": 1, "test": 2}
    return sorted(splits, key=lambda item: (preferred.get(item, 99), item))


def vocab_sizes(run_dir: Path, splits: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, int]:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    token_semantics = manifest.get("token_semantics", {}) if isinstance(manifest.get("token_semantics"), dict) else {}
    sizes: Dict[str, int] = {}
    for branch in BRANCHES:
        semantic_key = branch.replace("_tokens", "_vocab_size")
        if semantic_key in token_semantics:
            sizes[branch] = int(token_semantics[semantic_key])
        else:
            sizes[branch] = max(int(np.max(split[branch])) + 1 for split in splits.values() if branch in split)
    return sizes


def task_mask(data: Mapping[str, np.ndarray], task_name: str) -> np.ndarray:
    spec = TASKS[task_name]
    labels = np.asarray(data["label_name"]).astype(str)
    mask = np.isin(labels, np.asarray(spec["label_names"], dtype=str))
    source_task = spec["source_task"]
    if source_task is not None:
        tasks = np.asarray(data["source_task"]).astype(str)
        mask &= tasks == str(source_task)
    return mask


def label_vector(data: Mapping[str, np.ndarray], task_name: str, mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray(data["label_name"]).astype(str)[mask]
    classes = list(TASKS[task_name]["label_names"])
    class_to_id = {label: index for index, label in enumerate(classes)}
    return np.asarray([class_to_id[label] for label in labels], dtype=np.int64), classes


def unigram_features(data: Mapping[str, np.ndarray], mask: np.ndarray, sizes: Mapping[str, int]) -> sparse.csr_matrix:
    rows = []
    for branch in BRANCHES:
        tokens = data[branch][mask].astype(np.int64)
        vocab = int(sizes[branch])
        n_samples = tokens.shape[0]
        sample_ids = np.repeat(np.arange(n_samples), tokens.shape[1])
        code_ids = tokens.reshape(-1)
        valid = (code_ids >= 0) & (code_ids < vocab)
        values = np.ones(valid.sum(), dtype=np.float32) / max(tokens.shape[1], 1)
        rows.append(sparse.csr_matrix((values, (sample_ids[valid], code_ids[valid])), shape=(n_samples, vocab)))
    return sparse.hstack(rows, format="csr")


def bigram_features(data: Mapping[str, np.ndarray], mask: np.ndarray, sizes: Mapping[str, int]) -> sparse.csr_matrix:
    rows = []
    for branch in BRANCHES:
        tokens = data[branch][mask].astype(np.int64)
        vocab = int(sizes[branch])
        n_samples = tokens.shape[0]
        if tokens.shape[1] < 2:
            rows.append(sparse.csr_matrix((n_samples, vocab * vocab), dtype=np.float32))
            continue
        starts = tokens[:, :-1].reshape(-1)
        ends = tokens[:, 1:].reshape(-1)
        pair_ids = starts * vocab + ends
        sample_ids = np.repeat(np.arange(n_samples), tokens.shape[1] - 1)
        valid = (starts >= 0) & (starts < vocab) & (ends >= 0) & (ends < vocab)
        values = np.ones(valid.sum(), dtype=np.float32) / max(tokens.shape[1] - 1, 1)
        rows.append(
            sparse.csr_matrix((values, (sample_ids[valid], pair_ids[valid])), shape=(n_samples, vocab * vocab))
        )
    return sparse.hstack(rows, format="csr")


def build_features(
    data: Mapping[str, np.ndarray],
    task_name: str,
    feature_kind: str,
    sizes: Mapping[str, int],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, list[str]]:
    mask = task_mask(data, task_name)
    y, classes = label_vector(data, task_name, mask)
    if feature_kind == "bot":
        x = unigram_features(data, mask, sizes)
    elif feature_kind == "bot_bigram":
        x = sparse.hstack(
            [unigram_features(data, mask, sizes), bigram_features(data, mask, sizes)],
            format="csr",
        )
    else:
        raise ValueError(f"Unsupported feature kind: {feature_kind}")
    subjects = np.asarray(data.get("subject_id", np.full(mask.sum(), -1))).astype(np.int64)[mask]
    return x, y, subjects, classes


def fit_and_score(
    train_x: sparse.csr_matrix,
    train_y: np.ndarray,
    eval_x: sparse.csr_matrix,
    eval_y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, Dict[str, float]]:
    model = LogisticRegression(
        solver="saga",
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        random_state=seed,
    )
    model.fit(train_x, train_y)
    pred = model.predict(eval_x)
    return pred, {
        "balanced_accuracy": float(balanced_accuracy_score(eval_y, pred)),
        "macro_f1": float(f1_score(eval_y, pred, average="macro", zero_division=0)),
    }


def run_probe(run_dir: Path, output_dir: Path, *, seed: int) -> Dict[str, Any]:
    splits = {split: load_split(run_dir, split) for split in available_splits(run_dir)}
    if "train" not in splits:
        raise KeyError("Token export must include a train split")
    sizes = vocab_sizes(run_dir, splits)
    rows = []
    confusion_payload: Dict[str, Any] = {}
    for task_name in TASKS:
        train_mask = task_mask(splits["train"], task_name)
        if train_mask.sum() == 0:
            continue
        for feature_kind in ("bot", "bot_bigram"):
            train_x, train_y, train_subjects, classes = build_features(splits["train"], task_name, feature_kind, sizes)
            if np.unique(train_y).size < 2:
                continue
            for split_name in ("val", "test"):
                if split_name not in splits:
                    continue
                eval_mask = task_mask(splits[split_name], task_name)
                if eval_mask.sum() == 0:
                    continue
                eval_x, eval_y, eval_subjects, _ = build_features(splits[split_name], task_name, feature_kind, sizes)
                if np.unique(eval_y).size < 2:
                    continue
                pred, metrics = fit_and_score(train_x, train_y, eval_x, eval_y, seed=seed)
                cm = confusion_matrix(eval_y, pred, labels=list(range(len(classes))))
                key = f"{task_name}/{feature_kind}/{split_name}"
                confusion_payload[key] = {
                    "classes": classes,
                    "matrix": cm.tolist(),
                }
                rows.append({
                    "task": task_name,
                    "feature": feature_kind,
                    "split": split_name,
                    "train_samples": int(train_y.shape[0]),
                    "eval_samples": int(eval_y.shape[0]),
                    "train_subjects": int(np.unique(train_subjects).size),
                    "eval_subjects": int(np.unique(eval_subjects).size),
                    **metrics,
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0])
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema_version": "source_observation_token_downstream_probe_v1",
        "token_run_dir": str(run_dir),
        "vocab_sizes": sizes,
        "rows": rows,
        "confusion": confusion_payload,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    summary = run_probe(Path(args.token_run_dir).resolve(), Path(args.output_dir).resolve(), seed=args.seed)
    print(json.dumps({"rows": len(summary["rows"]), "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
