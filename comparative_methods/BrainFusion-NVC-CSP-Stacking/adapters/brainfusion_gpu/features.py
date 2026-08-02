"""Fold-local NVC selection and CSP features for the BrainFusion reimplementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
import torch

from .nvc import NVCConfig, brainfusion_nvc_contribution_timeseries


def _identity_hash(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_fit_identities(sample_ids: Sequence[str], sample_count: int) -> str:
    identities = tuple(str(value) for value in sample_ids)
    if len(identities) != sample_count or len(set(identities)) != len(identities):
        raise ValueError("fold-local fitting requires one unique sample identity per trial")
    return _identity_hash(identities)


@dataclass(frozen=True)
class CSPConfig:
    components_per_class: int = 2
    regularization: float = 0.1
    variance_floor: float = 1e-12


class TorchCSP:
    """Deterministic regularized one-vs-rest CSP fitted on one outer train fold."""

    def __init__(self, config: CSPConfig = CSPConfig()) -> None:
        if config.components_per_class <= 0:
            raise ValueError("CSP components_per_class must be positive")
        if not 0.0 <= config.regularization < 1.0:
            raise ValueError("CSP regularization must lie in [0, 1)")
        self.config = config
        self.filters_: torch.Tensor | None = None
        self.classes_: torch.Tensor | None = None
        self.fit_sample_identity_sha256_: str | None = None

    def _regularize(self, covariance: torch.Tensor) -> torch.Tensor:
        channels = covariance.shape[-1]
        scale = torch.trace(covariance) / channels
        identity = torch.eye(channels, dtype=covariance.dtype, device=covariance.device)
        return (
            (1.0 - self.config.regularization) * covariance
            + self.config.regularization * scale * identity
        )

    @staticmethod
    def _trial_covariances(trials: torch.Tensor) -> torch.Tensor:
        centered = trials - trials.mean(dim=-1, keepdim=True)
        covariance = centered @ centered.transpose(-1, -2)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        if bool((trace <= 0).any()) or not bool(torch.isfinite(trace).all()):
            raise ValueError("CSP requires finite non-constant trials")
        return covariance / trace[:, None, None]

    @staticmethod
    def _generalized_filters(
        positive: torch.Tensor, negative: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        composite = positive + negative
        eigenvalues, eigenvectors = torch.linalg.eigh(composite)
        floor = torch.finfo(composite.dtype).eps * composite.shape[-1]
        if bool((eigenvalues <= floor).any()):
            raise ValueError("CSP composite covariance is not positive definite")
        whitening = (
            eigenvectors
            @ torch.diag(eigenvalues.rsqrt())
            @ eigenvectors.transpose(-1, -2)
        )
        whitened = whitening @ positive @ whitening
        values, vectors = torch.linalg.eigh(whitened)
        filters = vectors.transpose(-1, -2) @ whitening
        return values, filters

    def fit(
        self, trials: torch.Tensor, labels: torch.Tensor, *, sample_ids: Sequence[str]
    ) -> "TorchCSP":
        if trials.ndim != 3:
            raise ValueError("CSP trials must have shape [N,C,T]")
        labels = labels.to(device=trials.device, dtype=torch.long).reshape(-1)
        if labels.numel() != trials.shape[0]:
            raise ValueError("CSP labels and trials differ in length")
        classes = torch.unique(labels, sorted=True)
        if classes.numel() < 2:
            raise ValueError("CSP requires at least two training classes")
        identity = _validate_fit_identities(sample_ids, trials.shape[0])
        covariance = self._trial_covariances(trials)
        class_covariances = []
        for label in classes:
            selected = covariance[labels == label]
            if selected.shape[0] < 2:
                raise ValueError("CSP requires at least two trials per class")
            class_covariances.append(self._regularize(selected.mean(dim=0)))

        selected_filters = []
        if len(class_covariances) == 2:
            _, filters = self._generalized_filters(
                class_covariances[0], class_covariances[1]
            )
            count = min(self.config.components_per_class, filters.shape[0] // 2)
            selected_filters.extend((filters[:count], filters[-count:]))
        else:
            for index, positive in enumerate(class_covariances):
                negative = torch.stack(
                    [row for other, row in enumerate(class_covariances) if other != index]
                ).mean(dim=0)
                _, filters = self._generalized_filters(positive, negative)
                count = min(self.config.components_per_class, filters.shape[0])
                selected_filters.append(filters[-count:])
        self.filters_ = torch.cat(selected_filters, dim=0).detach().clone()
        self.classes_ = classes.detach().clone()
        self.fit_sample_identity_sha256_ = identity
        return self

    def transform(self, trials: torch.Tensor) -> torch.Tensor:
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted before transform")
        if trials.ndim != 3 or trials.shape[1] != self.filters_.shape[1]:
            raise ValueError("CSP transform trial shape differs from fitted channels")
        filters = self.filters_.to(device=trials.device, dtype=trials.dtype)
        projected = torch.einsum("kc,nct->nkt", filters, trials)
        variance = projected.var(dim=-1, unbiased=False)
        variance = variance / variance.sum(dim=-1, keepdim=True).clamp_min(
            self.config.variance_floor
        )
        return variance.clamp_min(self.config.variance_floor).log()

    def fit_transform(
        self, trials: torch.Tensor, labels: torch.Tensor, *, sample_ids: Sequence[str]
    ) -> torch.Tensor:
        return self.fit(trials, labels, sample_ids=sample_ids).transform(trials)


class NVCPairSelector:
    """Select dynamic NVC pairs by training-only class-separation evidence."""

    def __init__(self, pair_count: int = 32, variance_floor: float = 1e-12) -> None:
        if pair_count <= 0:
            raise ValueError("NVC pair_count must be positive")
        self.pair_count = int(pair_count)
        self.variance_floor = float(variance_floor)
        self.indices_: torch.Tensor | None = None
        self.scores_: torch.Tensor | None = None
        self.fit_sample_identity_sha256_: str | None = None

    def fit(
        self, contributions: torch.Tensor, labels: torch.Tensor, *, sample_ids: Sequence[str]
    ) -> "NVCPairSelector":
        if contributions.ndim != 3:
            raise ValueError("NVC contributions must have shape [N,pairs,time]")
        labels = labels.to(device=contributions.device, dtype=torch.long).reshape(-1)
        if labels.numel() != contributions.shape[0]:
            raise ValueError("NVC selector labels and trials differ in length")
        classes = torch.unique(labels, sorted=True)
        if classes.numel() < 2:
            raise ValueError("NVC pair selection requires at least two classes")
        identity = _validate_fit_identities(sample_ids, contributions.shape[0])
        correlations = contributions.sum(dim=-1)
        overall = correlations.mean(dim=0)
        between = torch.zeros_like(overall)
        within = torch.zeros_like(overall)
        for label in classes:
            selected = correlations[labels == label]
            if selected.shape[0] < 2:
                raise ValueError("NVC pair selection requires two trials per class")
            difference = selected.mean(dim=0) - overall
            between += selected.shape[0] * difference.square()
            within += (selected - selected.mean(dim=0)).square().sum(dim=0)
        scores = between / within.clamp_min(self.variance_floor)
        scores_numpy = scores.detach().cpu().numpy()
        indices_numpy = np.lexsort((np.arange(scores_numpy.size), -scores_numpy))
        count = min(self.pair_count, scores_numpy.size)
        self.indices_ = torch.as_tensor(
            indices_numpy[:count], dtype=torch.long, device=contributions.device
        )
        self.scores_ = scores[self.indices_].detach().clone()
        self.fit_sample_identity_sha256_ = identity
        return self

    def transform(self, contributions: torch.Tensor) -> torch.Tensor:
        if self.indices_ is None:
            raise RuntimeError("NVC pair selector must be fitted before transform")
        indices = self.indices_.to(device=contributions.device)
        return contributions.index_select(1, indices)


class BrainFusionFeaturePipeline:
    """Four-view CSP feature extractor with no state fitted outside its inputs."""

    def __init__(
        self,
        *,
        nvc_config: NVCConfig = NVCConfig(),
        csp_config: CSPConfig = CSPConfig(),
        nvc_pair_count: int = 32,
    ) -> None:
        self.nvc_config = nvc_config
        self.selector = NVCPairSelector(pair_count=nvc_pair_count)
        self.csps = {
            "eeg": TorchCSP(csp_config),
            "hbo": TorchCSP(csp_config),
            "hbr": TorchCSP(csp_config),
            "nvc": TorchCSP(csp_config),
        }
        self.fit_sample_identity_sha256_: str | None = None

    def _nvc(self, eeg: torch.Tensor, hbo: torch.Tensor, hbr: torch.Tensor) -> torch.Tensor:
        _, contributions, _ = brainfusion_nvc_contribution_timeseries(
            eeg, hbo, hbr, self.nvc_config
        )
        return contributions

    def fit(
        self,
        eeg: torch.Tensor,
        hbo: torch.Tensor,
        hbr: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_ids: Sequence[str],
    ) -> "BrainFusionFeaturePipeline":
        identity = _validate_fit_identities(sample_ids, eeg.shape[0])
        if hbo.shape[0] != eeg.shape[0] or hbr.shape[0] != eeg.shape[0]:
            raise ValueError("BrainFusion feature modalities differ in trial count")
        contributions = self._nvc(eeg, hbo, hbr)
        selected_nvc = self.selector.fit(
            contributions, labels, sample_ids=sample_ids
        ).transform(contributions)
        for name, trials in (
            ("eeg", eeg),
            ("hbo", hbo),
            ("hbr", hbr),
            ("nvc", selected_nvc),
        ):
            self.csps[name].fit(trials, labels, sample_ids=sample_ids)
        self.fit_sample_identity_sha256_ = identity
        return self

    def transform(
        self, eeg: torch.Tensor, hbo: torch.Tensor, hbr: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        contributions = self._nvc(eeg, hbo, hbr)
        selected_nvc = self.selector.transform(contributions)
        return {
            "eeg": self.csps["eeg"].transform(eeg),
            "hbo": self.csps["hbo"].transform(hbo),
            "hbr": self.csps["hbr"].transform(hbr),
            "nvc": self.csps["nvc"].transform(selected_nvc),
        }

    def fit_transform(
        self,
        eeg: torch.Tensor,
        hbo: torch.Tensor,
        hbr: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        return self.fit(
            eeg, hbo, hbr, labels, sample_ids=sample_ids
        ).transform(eeg, hbo, hbr)

    def audit_state(self) -> dict[str, Any]:
        if self.fit_sample_identity_sha256_ is None or self.selector.indices_ is None:
            raise RuntimeError("BrainFusion feature pipeline is not fitted")
        return {
            "schema": "brainfusion_fold_local_feature_state_v2",
            "fit_sample_identity_sha256": self.fit_sample_identity_sha256_,
            "nvc_pair_count": int(self.selector.indices_.numel()),
            "nvc_pair_indices": self.selector.indices_.detach().cpu().tolist(),
            "csp_feature_dimensions": {
                name: int(csp.filters_.shape[0]) if csp.filters_ is not None else 0
                for name, csp in self.csps.items()
            },
            "all_fitted_states_share_training_identity": all(
                csp.fit_sample_identity_sha256_ == self.fit_sample_identity_sha256_
                for csp in self.csps.values()
            )
            and self.selector.fit_sample_identity_sha256_
            == self.fit_sample_identity_sha256_,
        }
