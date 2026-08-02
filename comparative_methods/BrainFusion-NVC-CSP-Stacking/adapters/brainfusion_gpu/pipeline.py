"""Serializable end-to-end fold-local BrainFusion feature and stacking pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .features import BrainFusionFeaturePipeline
from .stacking import FoldLocalStackingClassifier, StackingConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if "protected" in {part.lower() for part in resolved.parts}:
        raise PermissionError(f"refusing protected BrainFusion artifact path: {resolved}")
    return resolved


class BrainFusionFoldPipeline:
    """One outer-fold pipeline whose complete fitted state is train-bound."""

    def __init__(
        self,
        *,
        features: BrainFusionFeaturePipeline | None = None,
        stacking_config: StackingConfig = StackingConfig(),
    ) -> None:
        self.features = features or BrainFusionFeaturePipeline()
        self.stacking = FoldLocalStackingClassifier(stacking_config)

    @staticmethod
    def _numpy_features(values: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        return {
            name: tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            for name, tensor in values.items()
        }

    def fit(
        self,
        eeg: torch.Tensor,
        hbo: torch.Tensor,
        hbr: torch.Tensor,
        labels: torch.Tensor,
        *,
        groups: Sequence[str],
        sample_ids: Sequence[str],
    ) -> "BrainFusionFoldPipeline":
        feature_values = self.features.fit_transform(
            eeg, hbo, hbr, labels, sample_ids=sample_ids
        )
        self.stacking.fit(
            self._numpy_features(feature_values),
            labels.detach().cpu().numpy(),
            groups=groups,
            sample_ids=sample_ids,
        )
        if self.features.fit_sample_identity_sha256_ != (
            self.stacking.fit_sample_identity_sha256_
        ):
            raise RuntimeError("BrainFusion feature and stacking training identities differ")
        return self

    def predict(
        self, eeg: torch.Tensor, hbo: torch.Tensor, hbr: torch.Tensor
    ) -> np.ndarray:
        return self.stacking.predict(
            self._numpy_features(self.features.transform(eeg, hbo, hbr))
        )

    def decision_function(
        self, eeg: torch.Tensor, hbo: torch.Tensor, hbr: torch.Tensor
    ) -> np.ndarray:
        return self.stacking.decision_function(
            self._numpy_features(self.features.transform(eeg, hbo, hbr))
        )

    def audit_state(self) -> dict[str, Any]:
        feature_audit = self.features.audit_state()
        stacking_audit = self.stacking.audit_state()
        shared = feature_audit["fit_sample_identity_sha256"]
        if shared != stacking_audit["fit_sample_identity_sha256"]:
            raise RuntimeError("BrainFusion fitted-state identity mismatch")
        return {
            "schema": "brainfusion_fold_local_pipeline_audit_v2",
            "fit_sample_identity_sha256": shared,
            "feature_state": feature_audit,
            "stacking_state": stacking_audit,
            "all_fitted_state_outer_training_only": True,
            "protected_test_opened": False,
        }

    def save(self, directory: str | Path) -> Path:
        audit = self.audit_state()
        root = _public_path(directory)
        root.mkdir(parents=True, exist_ok=True)
        feature_path = root / "feature_state.pt"
        stacking_path = root / "stacking.joblib"
        manifest_path = root / "manifest.json"
        torch.save(self.features.state_dict(), feature_path)
        self.stacking.save(stacking_path)
        manifest = {
            "schema": "brainfusion_fold_local_pipeline_checkpoint_v2",
            "fit_sample_identity_sha256": audit["fit_sample_identity_sha256"],
            "feature_state_sha256": _sha256_file(feature_path),
            "stacking_sha256": _sha256_file(stacking_path),
            "audit": audit,
            "protected_test_opened": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return root

    @classmethod
    def load(
        cls, directory: str | Path, *, device: torch.device | str = "cpu"
    ) -> "BrainFusionFoldPipeline":
        root = _public_path(directory)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "brainfusion_fold_local_pipeline_checkpoint_v2":
            raise ValueError("invalid BrainFusion pipeline checkpoint schema")
        if manifest.get("protected_test_opened") is not False:
            raise PermissionError("BrainFusion checkpoint reports protected access")
        feature_path = root / "feature_state.pt"
        stacking_path = root / "stacking.joblib"
        if _sha256_file(feature_path) != manifest["feature_state_sha256"]:
            raise ValueError("BrainFusion feature checkpoint hash drifted")
        if _sha256_file(stacking_path) != manifest["stacking_sha256"]:
            raise ValueError("BrainFusion stacking checkpoint hash drifted")
        feature_state = torch.load(feature_path, map_location="cpu", weights_only=True)
        output = cls(
            features=BrainFusionFeaturePipeline.from_state_dict(
                feature_state, device=device
            ),
            stacking_config=StackingConfig(),
        )
        output.stacking = FoldLocalStackingClassifier.load(stacking_path)
        if output.audit_state() != manifest["audit"]:
            raise ValueError("BrainFusion pipeline checkpoint audit drifted")
        return output
