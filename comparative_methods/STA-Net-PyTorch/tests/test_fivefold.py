import sys
from pathlib import Path

METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sta_net_pytorch.splits import build_sample_random_registry, validate_public_manifest


class FakeClassificationDataset:
    class Spec:
        key = "fake"
        task_type = "classification"

    spec = Spec()
    rows = [
        {
            "subject": f"s{index // 10}",
            "record_id": f"r{index // 5}",
            "join_key": f"j{index // 5}",
            "condition": str(index % 2),
            "class_index": index % 2,
            "window_offset_s": float(index),
            "event_index": index,
            "trial_group": f"g{index // 2}",
        }
        for index in range(100)
    ]

    def __len__(self):
        return len(self.rows)

    def lightweight_metadata(self, index):
        return dict(self.rows[index])


def test_sample_random_registry_partitions_every_sample_once_and_stratifies():
    dataset = FakeClassificationDataset()
    public, protected = build_sample_random_registry(
        dataset, seed=7, outer_folds=5, inner_folds=3
    )
    assert len(public) == len(protected) == 5
    assert sorted(index for fold in protected for index in fold["test_indices"]) == list(range(100))
    assert all(len(fold["test_indices"]) == 20 for fold in protected)
    assert all(
        len({dataset.rows[index]["class_index"] for index in fold["test_indices"]}) == 2
        for fold in protected
    )
    for public_fold, protected_fold in zip(public, protected, strict=True):
        train = set(public_fold["train_indices"])
        validation = set(public_fold["validation_indices"])
        test = set(protected_fold["test_indices"])
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test == set(range(100))
        assert public_fold["dependency_group_isolation"] is False
        assert validate_public_manifest(dataset, public_fold) == (
            public_fold["train_indices"],
            public_fold["validation_indices"],
        )
