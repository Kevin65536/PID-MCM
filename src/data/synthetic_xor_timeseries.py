"""Synthetic XOR time-series dataset for PID sanity checks.

We generate binary sequences X1, X2 ~ Bernoulli(0.5) i.i.d. per time step and define
Y = XOR(X1, X2) per time step.

In the classical PID setting with explicit target Y, XOR is pure synergy:
I(X1;Y)=I(X2;Y)=0, I(X1,X2;Y)=1 bit (per time step, under independence).

This dataset is used to verify that the framework can isolate XOR-relevant
information into a synergy token when the target is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class XORTimeseriesConfig:
    n_samples: int = 20000
    seq_length: int = 64
    seed: int = 42


class XORTimeseriesDataset(Dataset):
    def __init__(self, cfg: XORTimeseriesConfig):
        super().__init__()
        self.cfg = cfg
        rng = np.random.RandomState(cfg.seed)

        x1 = rng.randint(0, 2, size=(cfg.n_samples, cfg.seq_length)).astype(np.float32)
        x2 = rng.randint(0, 2, size=(cfg.n_samples, cfg.seq_length)).astype(np.float32)
        y = (x1.astype(np.int32) ^ x2.astype(np.int32)).astype(np.float32)

        self.x1 = torch.from_numpy(x1)
        self.x2 = torch.from_numpy(x2)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x1": self.x1[idx],
            "x2": self.x2[idx],
            "y": self.y[idx],
        }
