"""Self-contained PyTorch STA-Net comparison implementation."""

from .data import (
    TASK_SPECS,
    STANetSampleAdapter,
    STANetSpatialProjector,
    STANetTaskSpec,
    STANetUnifiedTaskDataset,
    collate_sta_net,
    get_sta_net_task_spec,
    task_contract_sha256,
)
from .model import STANet, STANetConfig, STANetObjective

__all__ = [
    "TASK_SPECS",
    "STANet",
    "STANetConfig",
    "STANetObjective",
    "STANetSampleAdapter",
    "STANetSpatialProjector",
    "STANetTaskSpec",
    "STANetUnifiedTaskDataset",
    "collate_sta_net",
    "get_sta_net_task_spec",
    "task_contract_sha256",
]
