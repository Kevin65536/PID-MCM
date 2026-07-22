"""Project-adapted EFRM implementation for synchronized EEG-fNIRS data."""

from .data import (
    EFRMPairedWindowAdapter,
    EFRMSyncPretrainDataset,
    CachedEFRMPretrainDataset,
    InventoryDiverseBatchSampler,
    RecordGroupedBatchSampler,
    collate_efrm_pairs,
)
from .model import EFRMDownstreamModel, EFRMSyncModel, VariableChannelMAE
from .protocol import PretrainingBoundary, PublicSplitSubjects, load_public_split_subjects
from .tasks import EFRMTaskSpec, EFRMUnifiedTaskDataset, TASK_SPECS, collate_efrm_task
from .training import cached_pretrain_backward, evaluate_pretrain_batch

__all__ = [
    "EFRMDownstreamModel",
    "CachedEFRMPretrainDataset",
    "EFRMPairedWindowAdapter",
    "EFRMSyncModel",
    "EFRMSyncPretrainDataset",
    "EFRMTaskSpec",
    "EFRMUnifiedTaskDataset",
    "InventoryDiverseBatchSampler",
    "RecordGroupedBatchSampler",
    "PretrainingBoundary",
    "PublicSplitSubjects",
    "TASK_SPECS",
    "VariableChannelMAE",
    "collate_efrm_pairs",
    "collate_efrm_task",
    "cached_pretrain_backward",
    "evaluate_pretrain_batch",
    "load_public_split_subjects",
]
