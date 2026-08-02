"""Leakage-safe classical stacking for BrainFusion CSP feature views."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


VIEW_ORDER = ("eeg", "hbo", "hbr", "nvc")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StackingConfig:
    inner_folds: int = 5
    seed: int = 42
    linear_svm_c_values: tuple[float, ...] = (0.1, 1.0)
    rbf_svm_c_values: tuple[float, ...] = (1.0,)
    random_forest_estimators: int = 200
    random_forest_max_depth: int | None = None
    meta_svm_c: float = 1.0


def _estimator_candidates(config: StackingConfig) -> list[tuple[str, BaseEstimator]]:
    candidates: list[tuple[str, BaseEstimator]] = []
    for value in config.linear_svm_c_values:
        candidates.append(
            (
                f"svm_linear_c{value:g}",
                SVC(
                    C=float(value),
                    kernel="linear",
                    class_weight="balanced",
                    decision_function_shape="ovr",
                    random_state=config.seed,
                ),
            )
        )
    for value in config.rbf_svm_c_values:
        candidates.append(
            (
                f"svm_rbf_c{value:g}",
                SVC(
                    C=float(value),
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    decision_function_shape="ovr",
                    random_state=config.seed,
                ),
            )
        )
    candidates.append(
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=int(config.random_forest_estimators),
                max_depth=config.random_forest_max_depth,
                class_weight="balanced",
                random_state=config.seed,
                n_jobs=1,
            ),
        )
    )
    return candidates


def _clone_candidate(name: str, config: StackingConfig) -> BaseEstimator:
    matches = [estimator for candidate, estimator in _estimator_candidates(config) if candidate == name]
    if len(matches) != 1:
        raise KeyError(f"unknown BrainFusion base-estimator candidate: {name}")
    return matches[0]


def _score_matrix(
    estimator: BaseEstimator, features: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    estimator_classes = np.asarray(getattr(estimator, "classes_"))
    if not np.array_equal(estimator_classes, classes):
        raise RuntimeError("inner estimator class set drifted")
    if hasattr(estimator, "predict_proba"):
        values = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    else:
        values = np.asarray(estimator.decision_function(features), dtype=np.float64)
        if values.ndim == 1:
            values = np.column_stack((-values, values))
    if values.shape != (features.shape[0], classes.size):
        raise RuntimeError("base estimator score shape drifted")
    if not np.isfinite(values).all():
        raise RuntimeError("base estimator emitted non-finite scores")
    return values


class FoldLocalStackingClassifier:
    """Select base learners and fit a linear-SVM meta model using train-only OOF scores."""

    def __init__(self, config: StackingConfig = StackingConfig()) -> None:
        if config.inner_folds < 2:
            raise ValueError("stacking inner_folds must be at least two")
        if config.random_forest_estimators <= 0:
            raise ValueError("random_forest_estimators must be positive")
        self.config = config
        self.classes_: np.ndarray | None = None
        self.view_scalers_: dict[str, StandardScaler] = {}
        self.base_estimators_: dict[str, BaseEstimator] = {}
        self.selected_candidates_: dict[str, str] = {}
        self.meta_scaler_: StandardScaler | None = None
        self.meta_estimator_: SVC | None = None
        self.fit_sample_identity_sha256_: str | None = None
        self.audit_: dict[str, Any] | None = None

    @staticmethod
    def _validate_views(features: Mapping[str, np.ndarray]) -> tuple[int, dict[str, np.ndarray]]:
        if set(features) != set(VIEW_ORDER):
            raise ValueError(f"stacking requires exactly these views: {VIEW_ORDER}")
        arrays = {name: np.asarray(features[name], dtype=np.float64) for name in VIEW_ORDER}
        counts = {array.shape[0] for array in arrays.values()}
        if len(counts) != 1 or any(array.ndim != 2 for array in arrays.values()):
            raise ValueError("stacking views must be finite [N,D] matrices with equal N")
        if any(array.shape[1] == 0 or not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("stacking views contain empty or non-finite features")
        return counts.pop(), arrays

    def _splits(
        self, labels: np.ndarray, groups: np.ndarray, sample_ids: tuple[str, ...]
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
        splitter = StratifiedGroupKFold(
            n_splits=self.config.inner_folds, shuffle=True, random_state=self.config.seed
        )
        splits = list(splitter.split(np.zeros(labels.size), labels, groups))
        reports = []
        validation_membership: list[int] = []
        for fold, (train, validation) in enumerate(splits):
            train_groups = set(groups[train].tolist())
            validation_groups = set(groups[validation].tolist())
            if train_groups & validation_groups:
                raise RuntimeError("inner stacking groups overlap")
            if set(labels[train]) != set(self.classes_) or set(labels[validation]) != set(
                self.classes_
            ):
                raise RuntimeError("inner stacking fold lacks a class")
            validation_membership.extend(validation.tolist())
            reports.append(
                {
                    "inner_fold": fold,
                    "train_sample_identity_sha256": _stable_hash(
                        [sample_ids[index] for index in train]
                    ),
                    "validation_sample_identity_sha256": _stable_hash(
                        [sample_ids[index] for index in validation]
                    ),
                    "train_group_count": len(train_groups),
                    "validation_group_count": len(validation_groups),
                    "group_overlap": False,
                }
            )
        if sorted(validation_membership) != list(range(labels.size)):
            raise RuntimeError("inner OOF folds do not cover training samples exactly once")
        return splits, reports

    def _candidate_oof(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        splits: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        candidate_name: str,
    ) -> tuple[np.ndarray, float]:
        if self.classes_ is None:
            raise RuntimeError("stacking classes are not initialized")
        output = np.full((labels.size, self.classes_.size), np.nan, dtype=np.float64)
        for train, validation in splits:
            scaler = StandardScaler().fit(features[train])
            estimator = _clone_candidate(candidate_name, self.config)
            estimator.fit(scaler.transform(features[train]), labels[train])
            output[validation] = _score_matrix(
                estimator, scaler.transform(features[validation]), self.classes_
            )
        if not np.isfinite(output).all():
            raise RuntimeError("candidate OOF scores are incomplete")
        predictions = self.classes_[np.argmax(output, axis=1)]
        score = float(f1_score(labels, predictions, average="macro", labels=self.classes_))
        return output, score

    def fit(
        self,
        features: Mapping[str, np.ndarray],
        labels: Sequence[int] | np.ndarray,
        *,
        groups: Sequence[str],
        sample_ids: Sequence[str],
    ) -> "FoldLocalStackingClassifier":
        count, arrays = self._validate_views(features)
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        group_array = np.asarray([str(value) for value in groups])
        identities = tuple(str(value) for value in sample_ids)
        if target.size != count or group_array.size != count or len(identities) != count:
            raise ValueError("stacking labels, groups, identities, and features differ in length")
        if len(set(identities)) != count:
            raise ValueError("stacking fit sample identities must be unique")
        self.classes_ = np.unique(target)
        if self.classes_.size < 2:
            raise ValueError("stacking requires at least two training classes")
        splits, fold_reports = self._splits(target, group_array, identities)

        selection_reports: dict[str, Any] = {}
        oof_blocks = []
        for view in VIEW_ORDER:
            candidate_scores: dict[str, float] = {}
            candidate_oof: dict[str, np.ndarray] = {}
            for name, _ in _estimator_candidates(self.config):
                output, score = self._candidate_oof(
                    arrays[view], target, splits, candidate_name=name
                )
                candidate_scores[name] = score
                candidate_oof[name] = output
            selected = max(candidate_scores, key=lambda name: (candidate_scores[name], -list(candidate_scores).index(name)))
            self.selected_candidates_[view] = selected
            oof_blocks.append(candidate_oof[selected])
            selection_reports[view] = {
                "candidate_macro_f1": candidate_scores,
                "selected_candidate": selected,
                "selection_scope": "outer_training_inner_group_oof_only",
            }

            scaler = StandardScaler().fit(arrays[view])
            estimator = _clone_candidate(selected, self.config)
            estimator.fit(scaler.transform(arrays[view]), target)
            self.view_scalers_[view] = scaler
            self.base_estimators_[view] = estimator

        meta_features = np.concatenate(oof_blocks, axis=1)
        self.meta_scaler_ = StandardScaler().fit(meta_features)
        self.meta_estimator_ = SVC(
            C=float(self.config.meta_svm_c),
            kernel="linear",
            class_weight="balanced",
            decision_function_shape="ovr",
            random_state=self.config.seed,
        )
        self.meta_estimator_.fit(self.meta_scaler_.transform(meta_features), target)
        self.fit_sample_identity_sha256_ = _stable_hash(identities)
        self.audit_ = {
            "schema": "brainfusion_fold_local_stacking_audit_v2",
            "fit_sample_count": count,
            "fit_sample_identity_sha256": self.fit_sample_identity_sha256_,
            "fit_group_count": int(np.unique(group_array).size),
            "classes": self.classes_.tolist(),
            "inner_folds": fold_reports,
            "inner_validation_covers_training_exactly_once": True,
            "selection": selection_reports,
            "meta_fit_scope": "outer_training_oof_base_scores_only",
            "validation_or_protected_labels_consumed": False,
        }
        return self

    def _meta_features(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.classes_ is None or not self.base_estimators_:
            raise RuntimeError("stacking classifier must be fitted before prediction")
        count, arrays = self._validate_views(features)
        blocks = []
        for view in VIEW_ORDER:
            transformed = self.view_scalers_[view].transform(arrays[view])
            blocks.append(
                _score_matrix(self.base_estimators_[view], transformed, self.classes_)
            )
        output = np.concatenate(blocks, axis=1)
        if output.shape[0] != count:
            raise RuntimeError("stacking meta-feature row count drifted")
        return output

    def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.meta_scaler_ is None or self.meta_estimator_ is None:
            raise RuntimeError("stacking classifier must be fitted before prediction")
        meta = self.meta_scaler_.transform(self._meta_features(features))
        return np.asarray(self.meta_estimator_.predict(meta), dtype=np.int64)

    def decision_function(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.meta_scaler_ is None or self.meta_estimator_ is None:
            raise RuntimeError("stacking classifier must be fitted before prediction")
        meta = self.meta_scaler_.transform(self._meta_features(features))
        values = np.asarray(self.meta_estimator_.decision_function(meta), dtype=np.float64)
        if values.ndim == 1:
            values = np.column_stack((-values, values))
        return values

    def audit_state(self) -> dict[str, Any]:
        if self.audit_ is None:
            raise RuntimeError("stacking classifier is not fitted")
        return json.loads(json.dumps(self.audit_))

    def save(self, path: str | Path) -> Path:
        if self.audit_ is None:
            raise RuntimeError("cannot serialize an unfitted stacking classifier")
        output = Path(path).resolve()
        if "protected" in {part.lower() for part in output.parts}:
            raise PermissionError(f"refusing protected stacking checkpoint path: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema": "brainfusion_fold_local_stacking_checkpoint_v2",
                "fit_sample_identity_sha256": self.fit_sample_identity_sha256_,
                "model": self,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FoldLocalStackingClassifier":
        source = Path(path).resolve()
        if "protected" in {part.lower() for part in source.parts}:
            raise PermissionError(f"refusing protected stacking checkpoint path: {source}")
        payload = joblib.load(source)
        if not isinstance(payload, dict) or payload.get("schema") != (
            "brainfusion_fold_local_stacking_checkpoint_v2"
        ):
            raise ValueError("invalid BrainFusion stacking checkpoint schema")
        model = payload.get("model")
        if not isinstance(model, cls) or payload.get("fit_sample_identity_sha256") != (
            model.fit_sample_identity_sha256_
        ):
            raise ValueError("BrainFusion stacking checkpoint identity drifted")
        return model
