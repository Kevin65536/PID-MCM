"""Leakage-safe reuse of the shared STA-Net split registry for EFRM pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import EFRMSyncPretrainDataset
from .tasks import EFRMUnifiedTaskDataset, get_task_spec


BOUNDARY_SCHEMA = "efrm_pretraining_boundary_v1"
PUBLIC_SPLIT_SCHEMAS = {"sta_net_split_registry_v2", "sta_net_subject_split_v1"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PublicSplitSubjects:
    task: str
    dataset_id: str
    manifest_path: str
    manifest_sha256: str
    manifest_schema: str
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    allowed_subjects: tuple[str, ...]
    metadata_sha256: str | None


def _assert_public_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    if "protected" in {part.lower() for part in path.parts}:
        raise PermissionError(f"refusing to read a protected split manifest: {path}")
    if manifest.get("schema") not in PUBLIC_SPLIT_SCHEMAS:
        raise ValueError(f"unsupported public split schema in {path}: {manifest.get('schema')}")
    if manifest.get("protected_test_opened", manifest.get("reserved_test_opened", False)):
        raise PermissionError(f"split manifest reports an opened protected test: {path}")
    forbidden = {"test_indices", "reserved_test_indices", "protected_indices"}.intersection(manifest)
    if forbidden:
        raise PermissionError(f"public manifest exposes protected indices: {sorted(forbidden)}")


def load_public_split_subjects(
    path: str | Path,
    *,
    task: str | None = None,
    cache_root: str = "data/cache/physiology_semantic_clean_v1",
) -> PublicSplitSubjects:
    """Resolve train/validation subjects without ever reading protected files."""

    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_public_manifest(manifest_path, manifest)
    task_key = str(task or manifest.get("task") or manifest_path.parent.name)
    spec = get_task_spec(task_key)
    dataset = EFRMUnifiedTaskDataset(spec, cache_root=cache_root)

    if manifest["schema"] == "sta_net_split_registry_v2":
        train_indices, validation_indices = dataset.validate_shared_public_split(manifest_path)
        train = {dataset.lightweight_metadata(index)["subject"] for index in train_indices}
        validation = {
            dataset.lightweight_metadata(index)["subject"] for index in validation_indices
        }
        metadata_sha256 = str(manifest.get("metadata_sha256") or "") or None
    else:
        train = {str(value) for value in manifest["train_subjects"]}
        validation = {str(value) for value in manifest["validation_subjects"]}
        reserved = {str(value) for value in manifest.get("reserved_test_subjects", ())}
        if (train | validation) & reserved:
            raise RuntimeError(f"public and reserved subject sets overlap in {manifest_path}")
        available = {
            dataset.lightweight_metadata(index)["subject"] for index in range(len(dataset))
        }
        unknown = (train | validation) - available
        if unknown:
            raise RuntimeError(f"split contains subjects absent from task ordering: {sorted(unknown)}")
        metadata_sha256 = dataset.metadata_fingerprint()

    if not train or not validation or train & validation:
        raise RuntimeError(f"train/validation subjects are empty or overlapping in {manifest_path}")
    return PublicSplitSubjects(
        task=task_key,
        dataset_id=spec.dataset_id,
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        manifest_schema=str(manifest["schema"]),
        train_subjects=tuple(sorted(train)),
        validation_subjects=tuple(sorted(validation)),
        allowed_subjects=tuple(sorted(train | validation)),
        metadata_sha256=metadata_sha256,
    )


class PretrainingBoundary:
    """Subject admission boundary shared by SSL train and validation loaders.

    If several task manifests use the same dataset, the common checkpoint keeps
    only subjects admitted by every manifest. A validation role in any task wins
    over a training role, preventing a downstream validation subject from
    entering self-supervised training through a second task.
    """

    def __init__(self, sources: Sequence[PublicSplitSubjects], *, mode: str) -> None:
        if not sources:
            raise ValueError("at least one public split manifest is required")
        self.sources = tuple(sources)
        self.mode = str(mode)
        by_dataset: dict[str, list[PublicSplitSubjects]] = {}
        for source in self.sources:
            by_dataset.setdefault(source.dataset_id, []).append(source)

        self.train_subjects: dict[str, tuple[str, ...]] = {}
        self.validation_subjects: dict[str, tuple[str, ...]] = {}
        for dataset_id, rows in by_dataset.items():
            common_allowed = set(rows[0].allowed_subjects)
            for row in rows[1:]:
                common_allowed.intersection_update(row.allowed_subjects)
            validation = common_allowed.intersection(
                set().union(*(set(row.validation_subjects) for row in rows))
            )
            train = common_allowed - validation
            if not train or not validation or train & validation:
                raise RuntimeError(
                    f"strict common boundary is empty or overlapping for {dataset_id}: "
                    f"train={len(train)}, validation={len(validation)}"
                )
            self.train_subjects[dataset_id] = tuple(sorted(train))
            self.validation_subjects[dataset_id] = tuple(sorted(validation))

    @classmethod
    def from_manifests(
        cls,
        paths: Sequence[str | Path],
        *,
        tasks: Sequence[str] | None = None,
        mode: str = "development_public_only",
        cache_root: str = "data/cache/physiology_semantic_clean_v1",
    ) -> "PretrainingBoundary":
        if tasks is not None and len(tasks) != len(paths):
            raise ValueError("tasks must be omitted or have one entry per manifest")
        rows = [
            load_public_split_subjects(
                path,
                task=None if tasks is None else tasks[index],
                cache_root=cache_root,
            )
            for index, path in enumerate(paths)
        ]
        return cls(rows, mode=mode)

    def indices_for(self, dataset: EFRMSyncPretrainDataset, role: str) -> list[int]:
        if role not in {"train", "validation"}:
            raise ValueError("role must be train or validation")
        admitted = self.train_subjects if role == "train" else self.validation_subjects
        indices: list[int] = []
        for index in range(len(dataset)):
            row = dataset.lightweight_metadata(index)
            if row["subject"] in admitted.get(row["dataset_id"], ()):
                indices.append(index)
        if not indices:
            raise RuntimeError(f"no pretraining windows survive the {role} boundary")
        return indices

    def assert_disjoint(self) -> None:
        for dataset_id in sorted(set(self.train_subjects) | set(self.validation_subjects)):
            overlap = set(self.train_subjects.get(dataset_id, ())).intersection(
                self.validation_subjects.get(dataset_id, ())
            )
            if overlap:
                raise RuntimeError(f"pretraining subject leakage in {dataset_id}: {sorted(overlap)}")

    def manifest(self) -> dict[str, Any]:
        self.assert_disjoint()
        result = {
            "schema": BOUNDARY_SCHEMA,
            "mode": self.mode,
            "protected_test_opened": False,
            "combination_rule": (
                "intersection_of_allowed_subjects_per_dataset; "
                "validation_role_wins_over_training_role"
            ),
            "train_subjects_by_dataset": self.train_subjects,
            "validation_subjects_by_dataset": self.validation_subjects,
            "sources": [asdict(source) for source in self.sources],
        }
        result["boundary_sha256"] = _stable_hash(result)
        return result


def role_counts(
    dataset: EFRMSyncPretrainDataset,
    indices: Iterable[int],
) -> dict[str, Any]:
    datasets: dict[str, int] = {}
    subjects: dict[str, set[str]] = {}
    records: dict[str, set[str]] = {}
    total = 0
    for index in indices:
        row = dataset.lightweight_metadata(index)
        dataset_id = row["dataset_id"]
        total += 1
        datasets[dataset_id] = datasets.get(dataset_id, 0) + 1
        subjects.setdefault(dataset_id, set()).add(row["subject"])
        records.setdefault(dataset_id, set()).add(row["join_key"])
    return {
        "sample_count": total,
        "sample_count_by_dataset": datasets,
        "subject_count_by_dataset": {key: len(value) for key, value in subjects.items()},
        "record_count_by_dataset": {key: len(value) for key, value in records.items()},
    }
