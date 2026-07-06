"""Versioned, auditable measurement adapters for physiology-semantic inputs.

The adapter removes a record-level baseline and applies a scale learned only
from training records.  It deliberately does not normalize individual crops:
the same samples therefore receive the same canonical values regardless of
where a downstream crop is taken.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


ADAPTER_SCHEMA = "physiology_measurement_adapter_v1"
_MAD_TO_STD = 1.482602218505602


def _as_time_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"measurement values must have shape [time, channels], got {array.shape}")
    return array


def robust_location_scale(values: np.ndarray, *, epsilon: float = 1e-8) -> tuple[float, float]:
    """Return a finite pooled median and robust standard-deviation estimate."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot estimate robust statistics without finite values")
    location = float(np.median(finite))
    scale = float(_MAD_TO_STD * np.median(np.abs(finite - location)))
    if not np.isfinite(scale) or scale < epsilon:
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        scale = float((q75 - q25) / 1.3489795003921634)
    if not np.isfinite(scale) or scale < epsilon:
        scale = float(np.std(finite))
    return location, max(scale, float(epsilon))


@dataclass(frozen=True)
class MeasurementAdapterSpec:
    dataset: str
    modality: str
    original_semantics: str
    original_unit: str
    canonical_semantics: str
    transform: str
    channel_names: tuple[str, ...]
    shared_scale: float
    fit_subjects: tuple[str, ...]
    schema: str = ADAPTER_SCHEMA
    baseline_rule: str = "full_record_channel_median"
    scale_rule: str = "train_only_pooled_mad"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_names"] = list(self.channel_names)
        payload["fit_subjects"] = list(self.fit_subjects)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasurementAdapterSpec":
        values = dict(payload)
        values["channel_names"] = tuple(str(item) for item in values["channel_names"])
        values["fit_subjects"] = tuple(str(item) for item in values["fit_subjects"])
        return cls(**values)


class PhysiologyMeasurementAdapter:
    """Apply an explicit dataset-specific transform in a shared relative scale."""

    VALID_TRANSFORMS = {"center", "relative_change"}

    def __init__(self, spec: MeasurementAdapterSpec):
        if spec.schema != ADAPTER_SCHEMA:
            raise ValueError(f"unsupported adapter schema {spec.schema!r}")
        if spec.transform not in self.VALID_TRANSFORMS:
            raise ValueError(f"unsupported measurement transform {spec.transform!r}")
        if not np.isfinite(spec.shared_scale) or spec.shared_scale <= 0:
            raise ValueError("shared_scale must be finite and positive")
        self.spec = spec

    @staticmethod
    def record_baseline(values: np.ndarray) -> np.ndarray:
        array = _as_time_channels(values)
        baseline = np.nanmedian(array, axis=0)
        if not np.all(np.isfinite(baseline)):
            raise ValueError("every channel must contain at least one finite baseline sample")
        return baseline

    @classmethod
    def _relative_values(
        cls,
        values: np.ndarray,
        *,
        baseline: np.ndarray,
        transform: str,
    ) -> np.ndarray:
        array = _as_time_channels(values)
        baseline = np.asarray(baseline, dtype=np.float64).reshape(1, -1)
        if baseline.shape[1] != array.shape[1]:
            raise ValueError("baseline channel count does not match measurement")
        centered = array - baseline
        if transform == "center":
            return centered
        if transform == "relative_change":
            finite_baseline = np.abs(baseline[np.isfinite(baseline)])
            reference = float(np.median(finite_baseline)) if finite_baseline.size else 1.0
            floor = max(reference * 1e-3, np.finfo(np.float64).eps)
            return centered / np.maximum(np.abs(baseline), floor)
        raise ValueError(f"unsupported transform {transform!r}")

    @classmethod
    def fit(
        cls,
        records: Iterable[np.ndarray],
        *,
        dataset: str,
        modality: str,
        original_semantics: str,
        original_unit: str,
        canonical_semantics: str,
        transform: str,
        channel_names: Iterable[str],
        fit_subjects: Iterable[str],
    ) -> "PhysiologyMeasurementAdapter":
        records = [_as_time_channels(record) for record in records]
        if not records:
            raise ValueError("at least one training record is required")
        if transform not in cls.VALID_TRANSFORMS:
            raise ValueError(f"unsupported measurement transform {transform!r}")
        channel_names = tuple(str(item) for item in channel_names)
        if any(record.shape[1] != len(channel_names) for record in records):
            raise ValueError("all records must match channel_names")
        relative = [
            cls._relative_values(
                record,
                baseline=cls.record_baseline(record),
                transform=transform,
            )
            for record in records
        ]
        _, scale = robust_location_scale(np.concatenate([item.ravel() for item in relative]))
        spec = MeasurementAdapterSpec(
            dataset=str(dataset),
            modality=str(modality),
            original_semantics=str(original_semantics),
            original_unit=str(original_unit),
            canonical_semantics=str(canonical_semantics),
            transform=str(transform),
            channel_names=channel_names,
            shared_scale=scale,
            fit_subjects=tuple(str(item) for item in fit_subjects),
        )
        return cls(spec)

    def transform(self, values: np.ndarray, *, baseline: np.ndarray | None = None) -> np.ndarray:
        array = _as_time_channels(values)
        if array.shape[1] != len(self.spec.channel_names):
            raise ValueError("measurement channel count does not match adapter")
        if baseline is None:
            baseline = self.record_baseline(array)
        relative = self._relative_values(array, baseline=baseline, transform=self.spec.transform)
        return relative / self.spec.shared_scale

    def inverse_transform(self, canonical: np.ndarray, *, baseline: np.ndarray) -> np.ndarray:
        canonical = _as_time_channels(canonical)
        baseline = np.asarray(baseline, dtype=np.float64).reshape(1, -1)
        relative = canonical * self.spec.shared_scale
        if self.spec.transform == "center":
            return relative + baseline
        finite_baseline = np.abs(baseline[np.isfinite(baseline)])
        reference = float(np.median(finite_baseline)) if finite_baseline.size else 1.0
        floor = max(reference * 1e-3, np.finfo(np.float64).eps)
        return relative * np.maximum(np.abs(baseline), floor) + baseline
