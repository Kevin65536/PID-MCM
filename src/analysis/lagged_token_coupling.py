"""Leakage-safe held-out coupling statistics for soft discrete tokens.

This module contains NumPy-only, side-effect-free helpers for analysing a pair of
posterior token sequences.  The central convention is that a positive lag pairs
``E[t]`` with ``F[t + lag]`` and that a pair is counted only when both token
positions are valid.  Inputs are expected to be trial windows with shape
``[batch, time, code]``; windows are never concatenated across their boundary.

The module intentionally stops at numerical summaries.  It does not read data,
fit a model/runner, or construct/optimise a heatmap.  Train-only token ordering
and categorical proper-score probes are represented by explicit fit objects so
that an evaluation call cannot accidentally refit on held-out observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LAGGED_COUPLING_SCHEMA_VERSION = "lagged_token_coupling_v1"
EPS = 1e-12


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    """Convert an input to float64 while preserving a useful public error."""

    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - NumPy wording varies
        raise ValueError(f"{name} must be numeric") from exc
    return result


def _validate_binary_mask(
    mask: np.ndarray | None,
    shape: tuple[int, int],
    *,
    name: str,
) -> np.ndarray:
    """Validate and return a boolean ``[batch,time]`` mask.

    Numeric masks are accepted only when every value is exactly zero or one;
    this prevents a silently truncated probability/weight mask from changing
    the estimand.
    """

    if mask is None:
        return np.ones(shape, dtype=bool)
    raw = np.asarray(mask)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    if raw.dtype == bool:
        return raw.copy()
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain booleans or zero/one values") from exc
    if np.any(~np.isfinite(numeric)) or np.any(~np.isin(numeric, (0.0, 1.0))):
        raise ValueError(f"{name} must contain only finite zero/one values")
    return numeric.astype(bool)


def _validate_probability_rows(
    probabilities: Any,
    *,
    name: str,
    ndim: int | None = None,
    valid_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Validate non-negative finite rows that sum to one.

    ``valid_rows`` is a one-dimensional row mask for inputs whose final axis is
    the categorical axis.  Rows marked false are ignored and replaced with
    zeros.  This is useful for padded posterior rows, while active rows remain
    strictly validated.
    """

    values = _as_float_array(probabilities, name=name)
    if ndim is not None and values.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {values.ndim}")
    if values.ndim < 1 or values.shape[-1] <= 0:
        raise ValueError(f"{name} must have a non-empty categorical axis")
    if valid_rows is None:
        rows = np.ones(values.shape[:-1], dtype=bool)
    else:
        rows = np.asarray(valid_rows, dtype=bool)
        if rows.shape != values.shape[:-1]:
            raise ValueError(
                f"{name} valid-row mask must have shape {values.shape[:-1]}, "
                f"got {rows.shape}"
            )
    active = values[rows]
    if np.any(~np.isfinite(active)) or np.any(active < 0.0):
        raise ValueError(f"{name} active rows must be finite and non-negative")
    if len(active):
        sums = active.sum(axis=-1)
        if np.any(~np.isclose(sums, 1.0, rtol=1e-7, atol=1e-8)):
            raise ValueError(f"{name} active rows must sum to one")
    # Do not allow NaN from an ignored row to propagate through einsum.
    cleaned = np.zeros_like(values, dtype=np.float64)
    cleaned[rows] = values[rows]
    return cleaned


def _validate_posterior_windows(
    posterior: Any,
    valid_mask: np.ndarray | None,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = _as_float_array(posterior, name=name)
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [batch,time,code]")
    if values.shape[0] <= 0 or values.shape[1] <= 0 or values.shape[2] <= 0:
        raise ValueError(f"{name} must have positive batch, time, and code dimensions")
    mask = _validate_binary_mask(valid_mask, values.shape[:2], name=f"{name}_valid_mask")
    return _validate_probability_rows(values, name=name, ndim=3, valid_rows=mask), mask


def validate_probabilities(probabilities: Any, *, name: str = "probabilities") -> np.ndarray:
    """Public strict validator for a categorical probability matrix/tensor."""

    return _validate_probability_rows(probabilities, name=name)


def _validate_counts_tensor(counts: Any, *, name: str = "cooccurrence") -> np.ndarray:
    values = _as_float_array(counts, name=name)
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [lag,eeg_code,fnirs_code]")
    if values.shape[0] <= 0 or values.shape[1] <= 0 or values.shape[2] <= 0:
        raise ValueError(f"{name} must have positive lag and code dimensions")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return values


def _validate_alpha(alpha: float, *, name: str = "alpha", strictly_positive: bool = True) -> float:
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(value) or (value <= 0.0 if strictly_positive else value < 0.0):
        relation = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be finite and {relation}")
    return value


def _lag_values_without_position_check(lags: Iterable[int] | int) -> np.ndarray:
    """Validate integer lag labels when the source time length is unavailable."""

    if isinstance(lags, (bool, np.bool_)):
        raise ValueError("lags must contain integers")
    if isinstance(lags, (int, np.integer)):
        raw = (int(lags),)
    else:
        try:
            raw = tuple(lags)
        except TypeError as exc:
            raise ValueError("lags must be an integer or iterable of integers") from exc
    if not raw:
        raise ValueError("at least one lag is required")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError("lags must contain integers")
    return np.asarray(raw, dtype=np.int64)


def _validate_lags(lags: Iterable[int] | int, *, positions: int) -> np.ndarray:
    result = _lag_values_without_position_check(lags)
    if positions <= 0 or np.any(np.abs(result) >= positions):
        raise ValueError("each lag must satisfy abs(lag) < time")
    return result


def validate_lags(lags: Iterable[int] | int, *, positions: int) -> np.ndarray:
    """Public lag validator using the ``E[t]`` to ``F[t+lag]`` convention."""

    return _validate_lags(lags, positions=positions)


def _validate_prior(
    prior: Any | None,
    *,
    n_lags: int,
    n_categories: int,
    name: str,
) -> np.ndarray:
    if prior is None:
        return np.full(
            (n_lags, n_categories), 1.0 / n_categories, dtype=np.float64
        )
    values = _as_float_array(prior, name=name)
    if values.shape == (n_categories,):
        values = np.broadcast_to(values[None, :], (n_lags, n_categories)).copy()
    elif values.shape != (n_lags, n_categories):
        raise ValueError(
            f"{name} must have shape [{n_categories}] or "
            f"[{n_lags},{n_categories}], got {values.shape}"
        )
    return _validate_probability_rows(values, name=name, ndim=2)


def _validate_probability_tensor(values: Any, *, name: str) -> np.ndarray:
    result = _validate_probability_rows(values, name=name, ndim=3)
    return result


# ---------------------------------------------------------------------------
# Soft lagged coupling and probability transforms
# ---------------------------------------------------------------------------


def soft_cooccurrence_tensor(
    eeg_posterior: Any,
    fnirs_posterior: Any,
    lags: Iterable[int] | int,
    eeg_valid_mask: np.ndarray | None = None,
    fnirs_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute mask-aware soft co-occurrence counts.

    Parameters
    ----------
    eeg_posterior, fnirs_posterior:
        Categorical posterior windows with shapes ``[batch,time,K_E]`` and
        ``[batch,time,K_F]``.  Active rows must be finite probability vectors.
    lags:
        Integer offsets.  A positive lag pairs ``E[:,t]`` with ``F[:,t+lag]``.
    eeg_valid_mask, fnirs_valid_mask:
        Optional boolean/zero-one masks with shape ``[batch,time]``.

    Returns
    -------
    numpy.ndarray
        Float64 tensor with shape ``[n_lags,K_E,K_F]`` in the exact requested
        lag order.  Its total at a lag equals the number of valid token pairs.
    """

    eeg, eeg_mask = _validate_posterior_windows(
        eeg_posterior, eeg_valid_mask, name="eeg_posterior"
    )
    fnirs, fnirs_mask = _validate_posterior_windows(
        fnirs_posterior, fnirs_valid_mask, name="fnirs_posterior"
    )
    if eeg.shape[:2] != fnirs.shape[:2]:
        raise ValueError("EEG and fNIRS posterior windows must share [batch,time]")
    lag_values = _validate_lags(lags, positions=eeg.shape[1])
    result = np.zeros(
        (len(lag_values), eeg.shape[-1], fnirs.shape[-1]), dtype=np.float64
    )
    positions = eeg.shape[1]
    for lag_index, lag in enumerate(lag_values):
        lag_int = int(lag)
        if lag_int >= 0:
            usable = positions - lag_int
            left = eeg[:, :usable]
            right = fnirs[:, lag_int : lag_int + usable]
            pair_mask = eeg_mask[:, :usable] & fnirs_mask[:, lag_int : lag_int + usable]
        else:
            offset = -lag_int
            usable = positions - offset
            left = eeg[:, offset : offset + usable]
            right = fnirs[:, :usable]
            pair_mask = eeg_mask[:, offset : offset + usable] & fnirs_mask[:, :usable]
        left_flat = left[pair_mask]
        right_flat = right[pair_mask]
        if len(left_flat):
            result[lag_index] = np.einsum(
                "mi,mj->ij", left_flat, right_flat, optimize=True
            )
    return result


def soft_cooccurrence(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`soft_cooccurrence_tensor`."""

    return soft_cooccurrence_tensor(*args, **kwargs)


def conditional_fnirs_given_eeg(
    cooccurrence: Any,
    *,
    alpha: float = 0.5,
    prior: Any | None = None,
) -> np.ndarray:
    """Return Dirichlet-smoothed ``P(F=j | E=i, lag)``.

    ``alpha`` is the pseudo-count per fNIRS cell, matching
    ``(C_ij + alpha) / (sum_j C_ij + K_F * alpha)`` under a uniform prior.
    A non-uniform ``prior`` redistributes the same total concentration
    ``K_F * alpha``.  ``prior`` may be shared by all lags or supplied per lag.
    """

    counts = _validate_counts_tensor(cooccurrence)
    concentration = _validate_alpha(alpha, strictly_positive=False)
    lag_count, eeg_count, fnirs_count = counts.shape
    prior_values = _validate_prior(
        prior,
        n_lags=lag_count,
        n_categories=fnirs_count,
        name="prior",
    )
    smoothed = counts + concentration * fnirs_count * prior_values[:, None, :]
    denominator = smoothed.sum(axis=-1, keepdims=True)
    result = np.divide(
        smoothed,
        denominator,
        out=np.zeros_like(smoothed, dtype=np.float64),
        where=denominator > 0.0,
    )
    empty_rows = denominator[..., 0] <= 0.0
    if np.any(empty_rows):
        broadcast_prior = np.broadcast_to(prior_values[:, None, :], result.shape)
        result[empty_rows] = broadcast_prior[empty_rows]
    return result


def dirichlet_conditional(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`conditional_fnirs_given_eeg`."""

    return conditional_fnirs_given_eeg(*args, **kwargs)


def dirichlet_smoothed_conditional(*args: Any, **kwargs: Any) -> np.ndarray:
    """Descriptive alias for :func:`conditional_fnirs_given_eeg`."""

    return conditional_fnirs_given_eeg(*args, **kwargs)


def fnirs_marginal(
    cooccurrence: Any,
    *,
    alpha: float = 0.0,
    prior: Any | None = None,
) -> np.ndarray:
    """Compute the fNIRS marginal ``P(F=j | lag)`` from co-occurrences.

    With the default ``alpha=0`` this is the empirical marginal.  An empty lag
    is assigned a uniform marginal (or ``prior``) so the returned rows remain
    valid probability vectors.  Positive ``alpha`` applies Dirichlet smoothing
    to the marginal counts.
    """

    counts = _validate_counts_tensor(cooccurrence)
    concentration = _validate_alpha(alpha, strictly_positive=False)
    lag_count, _, fnirs_count = counts.shape
    prior_values = _validate_prior(
        prior,
        n_lags=lag_count,
        n_categories=fnirs_count,
        name="prior",
    )
    marginal_counts = counts.sum(axis=1)
    if concentration > 0.0:
        marginal_counts = (
            marginal_counts + concentration * fnirs_count * prior_values
        )
    totals = marginal_counts.sum(axis=-1, keepdims=True)
    result = np.divide(
        marginal_counts,
        totals,
        out=np.zeros_like(marginal_counts, dtype=np.float64),
        where=totals > 0.0,
    )
    empty = totals[:, 0] <= 0.0
    if np.any(empty):
        result[empty] = prior_values[empty]
    return result


def fnirs_token_marginal(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`fnirs_marginal`."""

    return fnirs_marginal(*args, **kwargs)


def fnirs_marginal_from_cooccurrence(*args: Any, **kwargs: Any) -> np.ndarray:
    """Descriptive alias for :func:`fnirs_marginal`."""

    return fnirs_marginal(*args, **kwargs)


def conditional_log_lift(
    conditional: Any,
    marginal: Any,
) -> np.ndarray:
    """Return ``log(P(F|E,lag) / P(F|lag))`` without heatmap optimisation."""

    conditional_values = _validate_probability_tensor(
        conditional, name="conditional"
    )
    lag_count, _, fnirs_count = conditional_values.shape
    marginal_values = _as_float_array(marginal, name="marginal")
    if marginal_values.shape == (fnirs_count,):
        marginal_values = np.broadcast_to(
            marginal_values[None, :], (lag_count, fnirs_count)
        ).copy()
    elif marginal_values.shape != (lag_count, fnirs_count):
        raise ValueError(
            f"marginal must have shape [{fnirs_count}] or "
            f"[{lag_count},{fnirs_count}], got {marginal_values.shape}"
        )
    marginal_values = _validate_probability_rows(
        marginal_values, name="marginal", ndim=2
    )
    # Epsilon is only relevant for user-supplied zero cells; smoothed
    # conditionals and marginals are strictly positive in normal use.
    return np.log(np.maximum(conditional_values, EPS)) - np.log(
        np.maximum(marginal_values[:, None, :], EPS)
    )


def log_lift(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`conditional_log_lift`."""

    return conditional_log_lift(*args, **kwargs)


def _log_lift_from_counts(
    counts: np.ndarray,
    *,
    alpha: float,
    prior: Any | None,
    marginal_alpha: float,
    marginal_prior: Any | None,
) -> np.ndarray:
    conditional = conditional_fnirs_given_eeg(counts, alpha=alpha, prior=prior)
    marginal = fnirs_marginal(
        counts, alpha=marginal_alpha, prior=marginal_prior
    )
    return conditional_log_lift(conditional, marginal)


def matched_minus_deranged_expected_residual_log_lift(
    matched_cooccurrence: Any,
    deranged_cooccurrences: Any,
    *,
    alpha: float = 0.5,
    prior: Any | None = None,
    marginal_alpha: float = 0.0,
    marginal_prior: Any | None = None,
    deranged_weights: Any | None = None,
) -> np.ndarray:
    """Subtract the expected deranged log-lift from matched log-lift.

    ``matched_cooccurrence`` has shape ``[lag,E,F]``.  ``deranged_cooccurrences``
    may be one such tensor or an ensemble with shape ``[derangement,lag,E,F]``.
    Each derangement is transformed independently and the expected null is an
    arithmetic (or explicitly weighted) mean.  No held-out or protected data
    are accessed here; callers decide which split supplied each tensor.
    """

    matched = _validate_counts_tensor(
        matched_cooccurrence, name="matched_cooccurrence"
    )
    deranged = _as_float_array(
        deranged_cooccurrences, name="deranged_cooccurrences"
    )
    if deranged.ndim == 3:
        deranged = deranged[None, ...]
    if deranged.ndim != 4 or deranged.shape[1:] != matched.shape:
        raise ValueError(
            "deranged_cooccurrences must have shape [derangement,lag,E,F] "
            "matching matched_cooccurrence"
        )
    if deranged.shape[0] <= 0:
        raise ValueError("at least one deranged co-occurrence tensor is required")
    if np.any(~np.isfinite(deranged)) or np.any(deranged < 0.0):
        raise ValueError("deranged_cooccurrences must be finite and non-negative")
    if deranged_weights is None:
        weights = np.full(deranged.shape[0], 1.0 / deranged.shape[0])
    else:
        weights = _as_float_array(deranged_weights, name="deranged_weights")
        if weights.shape != (deranged.shape[0],):
            raise ValueError(
                f"deranged_weights must have shape [{deranged.shape[0]}]"
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("deranged_weights must be finite and non-negative")
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            raise ValueError("deranged_weights must have positive total")
        weights = weights / total_weight

    matched_lift = _log_lift_from_counts(
        matched,
        alpha=alpha,
        prior=prior,
        marginal_alpha=marginal_alpha,
        marginal_prior=marginal_prior,
    )
    null_lifts = np.stack(
        [
            _log_lift_from_counts(
                deranged[index],
                alpha=alpha,
                prior=prior,
                marginal_alpha=marginal_alpha,
                marginal_prior=marginal_prior,
            )
            for index in range(deranged.shape[0])
        ],
        axis=0,
    )
    expected_null = np.tensordot(weights, null_lifts, axes=(0, 0))
    return matched_lift - expected_null


def residual_log_lift(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`matched_minus_deranged_expected_residual_log_lift`."""

    return matched_minus_deranged_expected_residual_log_lift(*args, **kwargs)


def matched_minus_deranged_residual_log_lift(
    *args: Any, **kwargs: Any
) -> np.ndarray:
    """Short alias for the matched-minus-expected-null residual."""

    return matched_minus_deranged_expected_residual_log_lift(*args, **kwargs)


def residual_log_lift_from_lifts(
    matched_log_lift: Any,
    deranged_log_lifts: Any,
    *,
    deranged_weights: Any | None = None,
) -> np.ndarray:
    """Subtract an expected ensemble of already-computed null log-lifts."""

    matched = _as_float_array(matched_log_lift, name="matched_log_lift")
    if matched.ndim != 3 or np.any(~np.isfinite(matched)):
        raise ValueError("matched_log_lift must be finite with shape [lag,E,F]")
    deranged = _as_float_array(deranged_log_lifts, name="deranged_log_lifts")
    if deranged.ndim == 3:
        deranged = deranged[None, ...]
    if deranged.ndim != 4 or deranged.shape[1:] != matched.shape:
        raise ValueError(
            "deranged_log_lifts must have shape [derangement,lag,E,F] "
            "matching matched_log_lift"
        )
    if np.any(~np.isfinite(deranged)) or deranged.shape[0] == 0:
        raise ValueError("deranged_log_lifts must be non-empty and finite")
    if deranged_weights is None:
        weights = np.full(deranged.shape[0], 1.0 / deranged.shape[0])
    else:
        weights = _as_float_array(deranged_weights, name="deranged_weights")
        if weights.shape != (deranged.shape[0],):
            raise ValueError("deranged_weights must match derangement count")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("deranged_weights must be finite and non-negative")
        if weights.sum() <= 0.0:
            raise ValueError("deranged_weights must have positive total")
        weights = weights / weights.sum()
    return matched - np.tensordot(weights, deranged, axes=(0, 0))


# ---------------------------------------------------------------------------
# Train-only row/column ordering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenOrder:
    """A fixed row/column order fitted from one training matrix only."""

    row_order: tuple[int, ...]
    column_order: tuple[int, ...]
    method: str
    source_shape: tuple[int, int]


def _orient_score(score: np.ndarray) -> np.ndarray:
    """Resolve an SVD sign using the largest-magnitude coordinate."""

    values = np.asarray(score, dtype=np.float64).copy()
    if len(values):
        anchor = int(np.argmax(np.abs(values)))
        if values[anchor] < 0.0:
            values *= -1.0
    return values


def _stable_score_order(score: np.ndarray) -> tuple[int, ...]:
    indices = np.arange(len(score), dtype=np.int64)
    return tuple(int(i) for i in np.lexsort((indices, score)))


def _hierarchical_order(features: np.ndarray) -> tuple[int, ...]:
    """Dependency-free average-linkage seriation with deterministic ties."""

    values = np.asarray(features, dtype=np.float64)
    count = values.shape[0]
    if count <= 1:
        return tuple(range(count))
    distances = np.sum(
        np.square(values[:, None, :] - values[None, :, :]), axis=-1
    )
    clusters: list[tuple[int, ...]] = [(index,) for index in range(count)]
    while len(clusters) > 1:
        best: tuple[float, tuple[int, int], int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                pair_distance = float(
                    np.mean(
                        distances[
                            np.ix_(clusters[left], clusters[right])
                        ]
                    )
                )
                key = (pair_distance, (clusters[left][0], clusters[right][0]), left, right)
                if best is None or key < best:
                    best = key
        assert best is not None  # len(clusters) > 1
        _, _, left, right = best
        first, second = clusters[left], clusters[right]
        # Preserve the closer boundary, then use lexical index order as a tie
        # breaker; this creates a stable linear leaf order rather than a tree.
        boundary_ab = distances[first[-1], second[0]]
        boundary_ba = distances[second[-1], first[0]]
        if boundary_ba < boundary_ab or (
            boundary_ba == boundary_ab and second[0] < first[0]
        ):
            merged = second + first
        else:
            merged = first + second
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (left, right)
        ]
        clusters.append(merged)
    return clusters[0]


def fit_token_order(
    train_matrix: Any,
    *,
    method: str = "svd",
) -> TokenOrder:
    """Fit a fixed row/column order from training data only.

    ``train_matrix`` may be ``[E,F]`` or ``[lag,E,F]``; lagged inputs are
    averaged across lag before fitting.  ``method='svd'``/``'spectral'`` uses
    deterministic first-singular-vector scores.  ``method='hierarchical'`` is
    a dependency-free average-linkage seriation.  The returned order can be
    applied to held-out matrices with :func:`apply_token_order` without any
    further fitting.
    """

    values = _as_float_array(train_matrix, name="train_matrix")
    if values.ndim == 2:
        base = values
    elif values.ndim == 3:
        if values.shape[0] <= 0:
            raise ValueError("train_matrix must contain at least one lag")
        base = values.mean(axis=0)
    else:
        raise ValueError("train_matrix must have shape [E,F] or [lag,E,F]")
    if base.shape[0] <= 0 or base.shape[1] <= 0:
        raise ValueError("train_matrix must have positive row and column dimensions")
    if np.any(~np.isfinite(base)):
        raise ValueError("train_matrix must be finite")
    normalized_method = str(method).lower()
    if normalized_method not in {"svd", "spectral", "hierarchical"}:
        raise ValueError("method must be 'svd', 'spectral', or 'hierarchical'")

    if normalized_method == "hierarchical":
        row_order = _hierarchical_order(base)
        column_order = _hierarchical_order(base.T)
    else:
        # Double-centering makes the score respond to interaction structure
        # instead of merely sorting by token occupancy.
        centered = (
            base
            - base.mean(axis=1, keepdims=True)
            - base.mean(axis=0, keepdims=True)
            + base.mean()
        )
        left, singular, right_transposed = np.linalg.svd(
            centered, full_matrices=False
        )
        if len(singular) and singular[0] > EPS:
            row_score = _orient_score(left[:, 0] * singular[0])
            column_score = _orient_score(right_transposed[0] * singular[0])
        else:
            # Degenerate interaction matrices still receive a deterministic
            # order based on weighted column/row centroids.
            row_score = base @ np.arange(base.shape[1], dtype=np.float64)
            column_score = base.T @ np.arange(base.shape[0], dtype=np.float64)
        row_order = _stable_score_order(row_score)
        column_order = _stable_score_order(column_score)
    return TokenOrder(
        row_order=row_order,
        column_order=column_order,
        method=normalized_method,
        source_shape=(int(base.shape[0]), int(base.shape[1])),
    )


def fit_train_only_token_order(*args: Any, **kwargs: Any) -> TokenOrder:
    """Alias making the no-leakage fit boundary explicit."""

    return fit_token_order(*args, **kwargs)


def apply_token_order(matrix: Any, order: TokenOrder) -> np.ndarray:
    """Apply a previously fitted row/column order to ``[E,F]`` or ``[lag,E,F]``."""

    if not isinstance(order, TokenOrder):
        raise ValueError("order must be a TokenOrder returned by fit_token_order")
    values = _as_float_array(matrix, name="matrix")
    if values.ndim not in (2, 3):
        raise ValueError("matrix must have shape [E,F] or [lag,E,F]")
    if values.shape[-2:] != order.source_shape:
        raise ValueError(
            f"matrix code dimensions {values.shape[-2:]} do not match "
            f"fitted order {order.source_shape}"
        )
    if np.any(~np.isfinite(values)):
        raise ValueError("matrix must be finite")
    rows = np.asarray(order.row_order, dtype=np.int64)
    columns = np.asarray(order.column_order, dtype=np.int64)
    if not np.array_equal(np.sort(rows), np.arange(values.shape[-2])):
        raise ValueError("order.row_order must be a permutation of row indices")
    if not np.array_equal(np.sort(columns), np.arange(values.shape[-1])):
        raise ValueError("order.column_order must be a permutation of column indices")
    return np.take(np.take(values, rows, axis=-2), columns, axis=-1)


# ---------------------------------------------------------------------------
# Coupling summaries
# ---------------------------------------------------------------------------


def _as_lagged_values(values: Any, *, name: str) -> tuple[np.ndarray, bool]:
    result = _as_float_array(values, name=name)
    if result.ndim == 2:
        result = result[None, ...]
        was_matrix = True
    elif result.ndim == 3:
        was_matrix = False
    else:
        raise ValueError(f"{name} must have shape [E,F] or [lag,E,F]")
    if result.shape[1] <= 0 or result.shape[2] <= 0:
        raise ValueError(f"{name} must have positive code dimensions")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result, was_matrix


def top_pairs(matrix: Any, *, top_k: int = 1) -> tuple[tuple[int, int], ...]:
    """Return deterministic top ``(EEG token, fNIRS token)`` cells."""

    values = _as_float_array(matrix, name="matrix")
    if values.ndim != 2:
        raise ValueError("matrix must have shape [E,F]")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("matrix must have positive dimensions")
    if np.any(~np.isfinite(values)):
        raise ValueError("matrix must be finite")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    if top_k <= 0 or top_k > values.size:
        raise ValueError("top_k must be between one and the number of cells")
    order = np.argsort(-values.reshape(-1), kind="stable")[:top_k]
    return tuple((int(index // values.shape[1]), int(index % values.shape[1])) for index in order)


def pair_concentration(values: Any, *, top_k: int = 1) -> float | np.ndarray:
    """Fraction of each lag's non-negative mass in its top ``k`` pairs."""

    lagged, was_matrix = _as_lagged_values(values, name="values")
    if np.any(lagged < 0.0):
        raise ValueError("pair concentration values must be non-negative")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    if top_k <= 0 or top_k > lagged.shape[1] * lagged.shape[2]:
        raise ValueError("top_k must be between one and the number of cells")
    flattened = np.sort(lagged.reshape(lagged.shape[0], -1), axis=1)[:, ::-1]
    numerator = flattened[:, :top_k].sum(axis=1)
    denominator = lagged.sum(axis=(1, 2))
    concentration = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )
    return float(concentration[0]) if was_matrix else concentration


def _validate_lag_score_input(scores: Any, lags: Iterable[int] | int) -> tuple[np.ndarray, np.ndarray]:
    values = _as_float_array(scores, name="scores")
    if values.ndim not in (1, 3):
        raise ValueError("scores must have shape [lag] or [lag,E,F]")
    if values.ndim == 3 and (values.shape[1] <= 0 or values.shape[2] <= 0):
        raise ValueError("scores must have positive code dimensions")
    if len(values) <= 0 or np.any(~np.isfinite(values)):
        raise ValueError("scores must contain at least one finite lag")
    lag_values = _lag_values_without_position_check(lags)
    if len(lag_values) != len(values):
        raise ValueError("lags length must match the first scores dimension")
    return values, lag_values


def positive_vs_negative_lag_specificity(
    scores: Any,
    *,
    lags: Iterable[int] | int,
    positive_lags: Iterable[int] | None = None,
    negative_lags: Iterable[int] | None = None,
    reduction: str = "max",
) -> Mapping[str, Any]:
    """Compare a lag score on positive versus negative offsets.

    Tensor scores are reduced per lag using ``max`` (default), ``mean``,
    ``mean_abs`` or ``sum_positive``.  By default positive/negative groups are
    selected by the sign of the supplied lag values; zero is excluded.
    """

    values, lag_values = _validate_lag_score_input(scores, lags)
    reduction_name = str(reduction).lower()
    if values.ndim == 1:
        lag_scores = values
    elif reduction_name == "max":
        lag_scores = values.max(axis=(1, 2))
    elif reduction_name == "mean":
        lag_scores = values.mean(axis=(1, 2))
    elif reduction_name == "mean_abs":
        lag_scores = np.abs(values).mean(axis=(1, 2))
    elif reduction_name == "sum_positive":
        lag_scores = np.maximum(values, 0.0).sum(axis=(1, 2))
    else:
        raise ValueError(
            "reduction must be 'max', 'mean', 'mean_abs', or 'sum_positive'"
        )

    def _group_mask(group: Iterable[int] | None, *, positive: bool) -> np.ndarray:
        if group is None:
            return lag_values > 0 if positive else lag_values < 0
        selected = tuple(group)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in selected
        ):
            raise ValueError("explicit lag groups must contain integers")
        mask = np.isin(lag_values, np.asarray(selected, dtype=np.int64))
        if positive and np.any(lag_values[mask] <= 0):
            raise ValueError("positive_lags must contain only positive lags")
        if not positive and np.any(lag_values[mask] >= 0):
            raise ValueError("negative_lags must contain only negative lags")
        unknown = set(selected).difference(set(int(value) for value in lag_values))
        if unknown:
            raise ValueError(f"explicit lag group contains unavailable lags: {sorted(unknown)}")
        return mask

    positive_mask = _group_mask(positive_lags, positive=True)
    negative_mask = _group_mask(negative_lags, positive=False)
    positive_mean = (
        float(np.mean(lag_scores[positive_mask])) if np.any(positive_mask) else float("nan")
    )
    negative_mean = (
        float(np.mean(lag_scores[negative_mask])) if np.any(negative_mask) else float("nan")
    )
    difference = (
        positive_mean - negative_mean
        if np.isfinite(positive_mean) and np.isfinite(negative_mean)
        else float("nan")
    )
    ratio = (
        positive_mean / abs(negative_mean)
        if np.isfinite(positive_mean)
        and np.isfinite(negative_mean)
        and abs(negative_mean) > EPS
        else float("nan")
    )
    return {
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "positive_minus_negative": difference,
        "positive_to_abs_negative_ratio": ratio,
        "positive_lags": tuple(int(value) for value in lag_values[positive_mask]),
        "negative_lags": tuple(int(value) for value in lag_values[negative_mask]),
        "n_positive_lags": int(np.sum(positive_mask)),
        "n_negative_lags": int(np.sum(negative_mask)),
        "reduction": reduction_name,
        "lag_scores": lag_scores,
    }


def _top_pair_set(matrix: np.ndarray, *, top_k: int) -> set[tuple[int, int]]:
    return set(top_pairs(matrix, top_k=top_k))


def top_pair_jaccard(
    first: Any,
    second: Any,
    *,
    top_k: int = 1,
) -> float:
    """Jaccard similarity of top-pair sets in two matrices.

    For lagged inputs the per-lag Jaccards are averaged, retaining lag identity
    rather than pooling cells from unrelated lags.
    """

    left, left_was_matrix = _as_lagged_values(first, name="first")
    right, right_was_matrix = _as_lagged_values(second, name="second")
    if left.shape != right.shape:
        raise ValueError("first and second must have identical shapes")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    if top_k <= 0 or top_k > left.shape[1] * left.shape[2]:
        raise ValueError("top_k must be between one and the number of cells")
    values = []
    for lag_index in range(left.shape[0]):
        a = _top_pair_set(left[lag_index], top_k=top_k)
        b = _top_pair_set(right[lag_index], top_k=top_k)
        union = a | b
        values.append(float(len(a & b) / len(union)) if union else 1.0)
    return float(np.mean(values))


def top_pair_stability(
    replicates: Any,
    *,
    top_k: int = 1,
) -> Mapping[str, Any]:
    """Summarise mean pairwise top-set Jaccard across replicate matrices.

    Accepted shapes are ``[replicate,E,F]`` and ``[replicate,lag,E,F]``.  The
    result contains one Jaccard per lag and their mean.
    """

    values = _as_float_array(replicates, name="replicates")
    if values.ndim == 3:
        values = values[:, None, :, :]
    elif values.ndim != 4:
        raise ValueError("replicates must have shape [R,E,F] or [R,lag,E,F]")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("replicates must contain at least one replicate and lag")
    if np.any(~np.isfinite(values)):
        raise ValueError("replicates must be finite")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    if top_k <= 0 or top_k > values.shape[-2] * values.shape[-1]:
        raise ValueError("top_k must be between one and the number of cells")
    per_lag = np.empty(values.shape[1], dtype=np.float64)
    if values.shape[0] == 1:
        per_lag.fill(1.0)
    else:
        for lag_index in range(values.shape[1]):
            jaccards = []
            for first_index, second_index in combinations(range(values.shape[0]), 2):
                jaccards.append(
                    top_pair_jaccard(
                        values[first_index, lag_index],
                        values[second_index, lag_index],
                        top_k=top_k,
                    )
                )
            per_lag[lag_index] = float(np.mean(jaccards))
    return {
        "mean_pairwise_jaccard": float(np.mean(per_lag)),
        "per_lag_jaccard": per_lag,
        "replicate_count": int(values.shape[0]),
        "lag_count": int(values.shape[1]),
        "top_k": top_k,
    }


def summarize_lagged_coupling(
    cooccurrence: Any,
    *,
    lags: Iterable[int] | int,
    residual_log_lift: Any | None = None,
    top_k: int = 1,
    stability_replicates: Any | None = None,
) -> Mapping[str, Any]:
    """Return compact numerical summaries without selecting a heatmap."""

    counts = _validate_counts_tensor(cooccurrence)
    lag_values = _lag_values_without_position_check(lags)
    if len(lag_values) != counts.shape[0]:
        raise ValueError("lags length must match cooccurrence lag dimension")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be a positive integer")
    top_k = int(top_k)
    if top_k <= 0 or top_k > counts.shape[1] * counts.shape[2]:
        raise ValueError("top_k must be between one and the number of cells")
    top_by_lag = tuple(
        top_pairs(counts[index], top_k=top_k) for index in range(counts.shape[0])
    )
    summary: dict[str, Any] = {
        "schema_version": LAGGED_COUPLING_SCHEMA_VERSION,
        "lags": tuple(int(value) for value in lag_values),
        "pair_count_per_lag": counts.sum(axis=(1, 2)),
        "pair_concentration_top1": pair_concentration(counts, top_k=1),
        "pair_concentration_top_k": pair_concentration(counts, top_k=top_k),
        "top_pairs_per_lag": top_by_lag,
        "top_k": top_k,
    }
    if residual_log_lift is not None:
        residual = _as_float_array(residual_log_lift, name="residual_log_lift")
        if residual.shape != counts.shape or np.any(~np.isfinite(residual)):
            raise ValueError(
                "residual_log_lift must be finite and match cooccurrence shape"
            )
        summary["residual_peak_per_lag"] = residual.max(axis=(1, 2))
        summary["positive_vs_negative_lag_specificity"] = (
            positive_vs_negative_lag_specificity(residual, lags=lag_values)
        )
    if stability_replicates is not None:
        summary["top_pair_stability"] = top_pair_stability(
            stability_replicates, top_k=top_k
        )
    return summary


# ---------------------------------------------------------------------------
# Boundary-safe q0/q1 design rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaggedCategoricalRows:
    """Flattened within-window rows for one explicitly declared lag."""

    q0_design: np.ndarray
    eeg_posterior: np.ndarray
    fnirs_target: np.ndarray
    subject: np.ndarray
    window_index: np.ndarray
    source_time: np.ndarray
    target_time: np.ndarray
    condition_id: np.ndarray
    lag: int
    history_steps: int

    def __post_init__(self) -> None:
        row_count = len(self.subject)
        if self.q0_design.ndim != 2 or len(self.q0_design) != row_count:
            raise ValueError("q0_design must be [row,feature]")
        if self.eeg_posterior.ndim != 2 or len(self.eeg_posterior) != row_count:
            raise ValueError("eeg_posterior must be [row,eeg_code]")
        if self.fnirs_target.ndim != 2 or len(self.fnirs_target) != row_count:
            raise ValueError("fnirs_target must be [row,fnirs_code]")
        for values in (
            self.window_index,
            self.source_time,
            self.target_time,
            self.condition_id,
        ):
            if np.asarray(values).shape != (row_count,):
                raise ValueError("lagged categorical row metadata must be vectors")
        if row_count <= 0:
            raise ValueError("lagged categorical design contains no valid rows")


def _validate_integer_vector(
    values: Any,
    *,
    length: int,
    name: str,
    upper_exclusive: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer IDs") from exc
    if (
        np.any(~np.isfinite(numeric))
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric < 0)
    ):
        raise ValueError(f"{name} must contain non-negative integer IDs")
    result = numeric.astype(np.int64)
    if upper_exclusive is not None and np.any(result >= int(upper_exclusive)):
        raise ValueError(f"{name} contains an ID outside the declared range")
    return result


def build_lagged_categorical_rows(
    eeg_posterior: Any,
    fnirs_posterior: Any,
    *,
    lag: int,
    subject_ids: Sequence[str],
    condition_ids: Any,
    condition_count: int,
    eeg_valid_mask: np.ndarray | None = None,
    fnirs_valid_mask: np.ndarray | None = None,
    fnirs_history_steps: int = 1,
    include_target_time: bool = True,
    include_condition: bool = True,
) -> LaggedCategoricalRows:
    """Build q0/q1 rows without crossing a trial-window boundary.

    q0 contains the declared number of immediately preceding fNIRS posteriors
    before the target position, followed by optional target-time and condition
    one-hot controls. q1 appends the EEG posterior at the source position via
    :func:`fit_q0_q1`.  A row is admitted only when source, target, and every
    history token are valid.
    """

    eeg, eeg_mask = _validate_posterior_windows(
        eeg_posterior, eeg_valid_mask, name="eeg_posterior"
    )
    fnirs, fnirs_mask = _validate_posterior_windows(
        fnirs_posterior, fnirs_valid_mask, name="fnirs_posterior"
    )
    if eeg.shape[:2] != fnirs.shape[:2]:
        raise ValueError("EEG and fNIRS posterior windows must share [batch,time]")
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
        raise ValueError("lag must be an integer")
    lag = int(lag)
    _validate_lags((lag,), positions=eeg.shape[1])
    if (
        isinstance(fnirs_history_steps, (bool, np.bool_))
        or not isinstance(fnirs_history_steps, (int, np.integer))
        or int(fnirs_history_steps) <= 0
    ):
        raise ValueError("fnirs_history_steps must be a positive integer")
    history_steps = int(fnirs_history_steps)
    if history_steps >= eeg.shape[1]:
        raise ValueError("fnirs_history_steps must be smaller than time")
    if (
        isinstance(condition_count, (bool, np.bool_))
        or not isinstance(condition_count, (int, np.integer))
        or int(condition_count) <= 0
    ):
        raise ValueError("condition_count must be a positive integer")
    condition_count = int(condition_count)
    conditions = _validate_integer_vector(
        condition_ids,
        length=eeg.shape[0],
        name="condition_ids",
        upper_exclusive=condition_count,
    )
    subjects = np.asarray(subject_ids, dtype=str)
    if subjects.shape != (eeg.shape[0],) or np.any(subjects == ""):
        raise ValueError("subject_ids must be one non-empty value per window")

    q0_rows: list[np.ndarray] = []
    eeg_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    row_subject: list[str] = []
    window_rows: list[int] = []
    source_rows: list[int] = []
    target_time_rows: list[int] = []
    condition_rows: list[int] = []
    time_count = eeg.shape[1]
    for window in range(eeg.shape[0]):
        for source_time in range(time_count):
            target_time = source_time + lag
            if target_time < 0 or target_time >= time_count:
                continue
            history_times = tuple(
                range(target_time - history_steps, target_time)
            )
            if not history_times or history_times[0] < 0:
                continue
            if not eeg_mask[window, source_time] or not fnirs_mask[window, target_time]:
                continue
            if not all(fnirs_mask[window, time] for time in history_times):
                continue
            controls = [fnirs[window, list(history_times)].reshape(-1)]
            if include_target_time:
                time_control = np.zeros(time_count, dtype=np.float64)
                time_control[target_time] = 1.0
                controls.append(time_control)
            if include_condition:
                condition_control = np.zeros(condition_count, dtype=np.float64)
                condition_control[conditions[window]] = 1.0
                controls.append(condition_control)
            q0_rows.append(np.concatenate(controls))
            eeg_rows.append(eeg[window, source_time])
            target_rows.append(fnirs[window, target_time])
            row_subject.append(str(subjects[window]))
            window_rows.append(window)
            source_rows.append(source_time)
            target_time_rows.append(target_time)
            condition_rows.append(int(conditions[window]))
    if not q0_rows:
        raise ValueError("no valid lagged categorical rows remain after masks/history")
    return LaggedCategoricalRows(
        q0_design=np.stack(q0_rows),
        eeg_posterior=np.stack(eeg_rows),
        fnirs_target=np.stack(target_rows),
        subject=np.asarray(row_subject, dtype=str),
        window_index=np.asarray(window_rows, dtype=np.int64),
        source_time=np.asarray(source_rows, dtype=np.int64),
        target_time=np.asarray(target_time_rows, dtype=np.int64),
        condition_id=np.asarray(condition_rows, dtype=np.int64),
        lag=lag,
        history_steps=history_steps,
    )


# ---------------------------------------------------------------------------
# Categorical q0/q1 proper-score probes
# ---------------------------------------------------------------------------


def _validate_row_mask(mask: Any | None, n_rows: int, *, name: str) -> np.ndarray:
    if mask is None:
        return np.ones(n_rows, dtype=bool)
    raw = np.asarray(mask)
    if raw.shape not in {(n_rows,), (n_rows, 1)}:
        raise ValueError(f"{name} must have shape [{n_rows}]")
    if raw.shape == (n_rows, 1):
        raw = raw.reshape(n_rows)
    if raw.dtype == bool:
        return raw.copy()
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain booleans or zero/one values") from exc
    if np.any(~np.isfinite(numeric)) or np.any(~np.isin(numeric, (0.0, 1.0))):
        raise ValueError(f"{name} must contain only finite zero/one values")
    return numeric.astype(bool)


def _validate_design_matrix(
    design_matrix: Any | None,
    n_rows: int,
    valid_rows: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    if design_matrix is None:
        return np.empty((n_rows, 0), dtype=np.float64)
    values = _as_float_array(design_matrix, name=name)
    if values.ndim != 2 or values.shape[0] != n_rows:
        raise ValueError(f"{name} must have shape [n_samples,n_features]")
    if np.any(~np.isfinite(values[valid_rows])):
        raise ValueError(f"{name} active rows must be finite")
    cleaned = np.zeros_like(values, dtype=np.float64)
    cleaned[valid_rows] = values[valid_rows]
    return cleaned


def _validate_class_count(n_classes: int | None) -> int | None:
    if n_classes is None:
        return None
    if isinstance(n_classes, (bool, np.bool_)) or not isinstance(n_classes, (int, np.integer)):
        raise ValueError("n_classes must be a positive integer")
    n_classes = int(n_classes)
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    return n_classes


def _prepare_targets(
    targets: Any,
    valid_rows: np.ndarray,
    *,
    n_classes: int | None,
) -> tuple[np.ndarray, int]:
    raw = np.asarray(targets)
    if raw.ndim == 1:
        if len(raw) != len(valid_rows):
            raise ValueError("targets and valid_mask must have the same row count")
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("categorical targets must be integer IDs") from exc
        active = numeric[valid_rows]
        if np.any(~np.isfinite(active)) or np.any(active < 0) or np.any(active != np.floor(active)):
            raise ValueError("active categorical targets must be non-negative integer IDs")
        inferred = int(np.max(active)) + 1 if len(active) else None
        n_classes = _validate_class_count(n_classes)
        if n_classes is None:
            if inferred is None:
                raise ValueError("n_classes is required when no target row is valid")
            n_classes = inferred
        if inferred is not None and inferred > n_classes:
            raise ValueError("target ID exceeds n_classes")
        one_hot = np.zeros((len(raw), n_classes), dtype=np.float64)
        if len(active):
            row_indices = np.flatnonzero(valid_rows)
            one_hot[row_indices, active.astype(np.int64)] = 1.0
        return one_hot, n_classes
    if raw.ndim == 2:
        if raw.shape[0] != len(valid_rows) or raw.shape[1] <= 0:
            raise ValueError("probability targets must have shape [n_samples,n_classes]")
        values = _validate_probability_rows(
            raw, name="targets", ndim=2, valid_rows=valid_rows
        )
        inferred = int(values.shape[1])
        n_classes = _validate_class_count(n_classes)
        if n_classes is not None and n_classes != inferred:
            raise ValueError("target probability width must equal n_classes")
        return values, inferred if n_classes is None else n_classes
    raise ValueError("targets must have shape [n_samples] or [n_samples,n_classes]")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -745.0, 50.0))
    return exponent / np.maximum(exponent.sum(axis=-1, keepdims=True), EPS)


@dataclass(frozen=True)
class CategoricalModel:
    """Frozen multinomial softmax probe fitted on a training split."""

    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    n_classes: int
    n_features: int
    l2: float
    iterations: int
    converged: bool


def fit_categorical_model(
    design_matrix: Any | None,
    targets: Any,
    *,
    n_classes: int | None = None,
    valid_mask: np.ndarray | None = None,
    l2: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-8,
    learning_rate: float | None = None,
) -> CategoricalModel:
    """Fit a categorical softmax probe on training rows only.

    The function never sees evaluation arrays.  Design columns are standardized
    using valid training rows and the fitted statistics are stored in the frozen
    model for a later :func:`evaluate_categorical_model` call.
    """

    target_array = np.asarray(targets)
    if target_array.ndim < 1:
        raise ValueError("targets must have at least one dimension")
    n_rows = target_array.shape[0]
    valid_rows = _validate_row_mask(valid_mask, n_rows, name="valid_mask")
    if not np.any(valid_rows):
        raise ValueError("at least one valid training row is required")
    n_classes = _validate_class_count(n_classes)
    l2_value = _validate_alpha(l2, name="l2", strictly_positive=False)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer")
    max_iter = int(max_iter)
    if isinstance(tol, (bool, np.bool_)) or not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and positive")
    tol = float(tol)
    if learning_rate is not None:
        if isinstance(learning_rate, (bool, np.bool_)) or not np.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        learning_rate = float(learning_rate)

    design = _validate_design_matrix(
        design_matrix, n_rows, valid_rows, name="design_matrix"
    )
    target_matrix, class_count = _prepare_targets(
        targets, valid_rows, n_classes=n_classes
    )
    x_train = design[valid_rows]
    y_train = target_matrix[valid_rows]
    if x_train.shape[0] == 0:
        raise ValueError("at least one valid training row is required")
    if x_train.shape[1]:
        feature_mean = x_train.mean(axis=0)
        feature_scale = x_train.std(axis=0)
        feature_scale = np.where(
            np.isfinite(feature_scale) & (feature_scale > 1e-8), feature_scale, 1.0
        )
        standardized = (x_train - feature_mean) / feature_scale
    else:
        feature_mean = np.empty(0, dtype=np.float64)
        feature_scale = np.empty(0, dtype=np.float64)
        standardized = np.empty((len(x_train), 0), dtype=np.float64)
    x_augmented = np.concatenate(
        [np.ones((len(standardized), 1), dtype=np.float64), standardized], axis=1
    )
    coefficients = np.zeros((x_augmented.shape[1], class_count), dtype=np.float64)
    spectral_norm = float(np.linalg.norm(x_augmented, ord=2) ** 2 / len(x_augmented))
    step = (
        float(learning_rate)
        if learning_rate is not None
        else 1.0 / max(0.25 * spectral_norm + l2_value + 1e-8, 1e-8)
    )
    # Avoid a pathological huge step for all-zero/intercept-only designs.
    step = min(step, 10.0)
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        probabilities = _softmax(x_augmented @ coefficients)
        gradient = (x_augmented.T @ (probabilities - y_train)) / len(x_train)
        if l2_value > 0.0:
            gradient[1:] += l2_value * coefficients[1:]
        updated = coefficients - step * gradient
        iterations = iteration
        if np.max(np.abs(updated - coefficients)) <= tol:
            coefficients = updated
            converged = True
            break
        coefficients = updated
        if np.max(np.abs(gradient)) <= tol:
            converged = True
            break
    return CategoricalModel(
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        n_classes=class_count,
        n_features=int(design.shape[1]),
        l2=l2_value,
        iterations=iterations,
        converged=converged,
    )


def predict_categorical_proba(
    model: CategoricalModel,
    design_matrix: Any | None,
) -> np.ndarray:
    """Predict categorical probabilities using a fitted model only."""

    if not isinstance(model, CategoricalModel):
        raise ValueError("model must be a CategoricalModel")
    if design_matrix is None:
        design = np.empty((1, 0), dtype=np.float64)
        if model.n_features:
            raise ValueError("design_matrix is required for this fitted model")
    else:
        design = _as_float_array(design_matrix, name="design_matrix")
        if design.ndim != 2 or design.shape[1] != model.n_features:
            raise ValueError(
                f"design_matrix must have shape [n_samples,{model.n_features}]"
            )
        if np.any(~np.isfinite(design)):
            raise ValueError("design_matrix must be finite")
    if model.n_features == 0:
        standardized = np.empty((design.shape[0], 0), dtype=np.float64)
    else:
        standardized = (design - model.feature_mean) / model.feature_scale
    x_augmented = np.concatenate(
        [np.ones((design.shape[0], 1), dtype=np.float64), standardized], axis=1
    )
    return _softmax(x_augmented @ model.coefficients)


def categorical_proper_scores(
    probabilities: Any,
    targets: Any,
    *,
    valid_mask: np.ndarray | None = None,
) -> Mapping[str, float | int]:
    """Evaluate multiclass log loss and Brier score without fitting."""

    predicted = _as_float_array(probabilities, name="probabilities")
    if predicted.ndim != 2 or predicted.shape[1] <= 0:
        raise ValueError("probabilities must have shape [n_samples,n_classes]")
    n_rows = predicted.shape[0]
    valid_rows = _validate_row_mask(valid_mask, n_rows, name="valid_mask")
    predicted = _validate_probability_rows(
        predicted, name="probabilities", ndim=2, valid_rows=valid_rows
    )
    target_matrix, class_count = _prepare_targets(
        targets, valid_rows, n_classes=predicted.shape[1]
    )
    if class_count != predicted.shape[1]:
        raise ValueError("target class count must match probabilities")
    selected = valid_rows
    if not np.any(selected):
        return {
            "n_samples": 0,
            "log_loss_nats": float("nan"),
            "brier_score": float("nan"),
        }
    target_selected = target_matrix[selected]
    prediction_selected = predicted[selected]
    log_loss = float(
        -np.mean(
            np.sum(
                target_selected * np.log(np.maximum(prediction_selected, EPS)),
                axis=1,
            )
        )
    )
    brier = float(np.mean(np.sum(np.square(prediction_selected - target_selected), axis=1)))
    return {
        "n_samples": int(np.sum(selected)),
        "log_loss_nats": log_loss,
        "brier_score": brier,
    }


def evaluate_categorical_model(
    model: CategoricalModel,
    design_matrix: Any | None,
    targets: Any,
    *,
    valid_mask: np.ndarray | None = None,
) -> Mapping[str, float | int]:
    """Predict and score a fitted categorical model on held-out rows."""

    target_array = np.asarray(targets)
    if target_array.ndim < 1:
        raise ValueError("targets must have at least one dimension")
    n_rows = target_array.shape[0]
    valid_rows = _validate_row_mask(valid_mask, n_rows, name="valid_mask")
    design = _validate_design_matrix(
        design_matrix, n_rows, valid_rows, name="design_matrix"
    )
    probabilities = predict_categorical_proba(model, design)
    return categorical_proper_scores(
        probabilities, targets, valid_mask=valid_rows
    )


def _assemble_q1_design(
    q0_design: Any | None,
    eeg_posterior: Any,
    n_rows: int,
    valid_rows: np.ndarray,
) -> np.ndarray:
    controls = _validate_design_matrix(
        q0_design, n_rows, valid_rows, name="q0_design"
    )
    eeg = _as_float_array(eeg_posterior, name="eeg_posterior")
    if eeg.ndim != 2 or eeg.shape[0] != n_rows or eeg.shape[1] <= 0:
        raise ValueError("eeg_posterior must have shape [n_samples,n_eeg_codes]")
    eeg = _validate_probability_rows(
        eeg, name="eeg_posterior", ndim=2, valid_rows=valid_rows
    )
    return np.concatenate([controls, eeg], axis=1)


@dataclass(frozen=True)
class Q0Q1Models:
    """Explicitly separated baseline (q0) and EEG-augmented (q1) probes."""

    q0: CategoricalModel
    q1: CategoricalModel


def fit_q0_q1(
    train_targets: Any,
    *,
    q0_design_train: Any | None = None,
    eeg_posterior_train: Any,
    n_classes: int | None = None,
    train_valid_mask: np.ndarray | None = None,
    l2: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-8,
    learning_rate: float | None = None,
) -> Q0Q1Models:
    """Fit q0 controls and q1 controls-plus-EEG probes on training only.

    q0's design can contain fNIRS history, time, and task controls.  q1 uses
    exactly those columns followed by the EEG posterior.  Evaluation data are
    intentionally absent from this function and must be supplied to
    :func:`evaluate_q0_q1` separately.
    """

    target_array = np.asarray(train_targets)
    if target_array.ndim < 1:
        raise ValueError("train_targets must have at least one dimension")
    n_rows = target_array.shape[0]
    valid_rows = _validate_row_mask(
        train_valid_mask, n_rows, name="train_valid_mask"
    )
    q0_design = _validate_design_matrix(
        q0_design_train, n_rows, valid_rows, name="q0_design_train"
    )
    q1_design = _assemble_q1_design(
        q0_design, eeg_posterior_train, n_rows, valid_rows
    )
    q0_model = fit_categorical_model(
        q0_design,
        train_targets,
        n_classes=n_classes,
        valid_mask=valid_rows,
        l2=l2,
        max_iter=max_iter,
        tol=tol,
        learning_rate=learning_rate,
    )
    q1_model = fit_categorical_model(
        q1_design,
        train_targets,
        n_classes=q0_model.n_classes,
        valid_mask=valid_rows,
        l2=l2,
        max_iter=max_iter,
        tol=tol,
        learning_rate=learning_rate,
    )
    return Q0Q1Models(q0=q0_model, q1=q1_model)


def evaluate_q0_q1(
    models: Q0Q1Models,
    eval_targets: Any,
    *,
    q0_design_eval: Any | None = None,
    eeg_posterior_eval: Any,
    eval_valid_mask: np.ndarray | None = None,
) -> Mapping[str, Any]:
    """Evaluate frozen q0/q1 probes and return q1-minus-q0 score increments."""

    if not isinstance(models, Q0Q1Models):
        raise ValueError("models must be a Q0Q1Models returned by fit_q0_q1")
    target_array = np.asarray(eval_targets)
    if target_array.ndim < 1:
        raise ValueError("eval_targets must have at least one dimension")
    n_rows = target_array.shape[0]
    valid_rows = _validate_row_mask(eval_valid_mask, n_rows, name="eval_valid_mask")
    q0_design = _validate_design_matrix(
        q0_design_eval, n_rows, valid_rows, name="q0_design_eval"
    )
    q1_design = _assemble_q1_design(
        q0_design, eeg_posterior_eval, n_rows, valid_rows
    )
    q0_scores = evaluate_categorical_model(
        models.q0, q0_design, eval_targets, valid_mask=valid_rows
    )
    q1_scores = evaluate_categorical_model(
        models.q1, q1_design, eval_targets, valid_mask=valid_rows
    )
    q0_log_loss = float(q0_scores["log_loss_nats"])
    q1_log_loss = float(q1_scores["log_loss_nats"])
    q0_brier = float(q0_scores["brier_score"])
    q1_brier = float(q1_scores["brier_score"])
    return {
        "q0": q0_scores,
        "q1": q1_scores,
        # Positive ``*_gain`` means q1 improves over q0.
        "log_loss_gain_nats": q0_log_loss - q1_log_loss,
        "brier_gain": q0_brier - q1_brier,
        # Explicit q1-minus-q0 increments are convenient for preregistered
        # conventions where lower proper scores are better.
        "delta_log_loss_q1_minus_q0": q1_log_loss - q0_log_loss,
        "delta_brier_q1_minus_q0": q1_brier - q0_brier,
    }


def subject_block_bootstrap(
    subject_values: Any,
    *,
    iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Bootstrap a subject-equal mean after all within-subject aggregation.

    ``subject_values`` may be ``[subject]`` or ``[subject,endpoint]``.  Paired
    comparisons must be differenced before this call, preserving pairing in
    every resample.
    """

    values = _as_float_array(subject_values, name="subject_values")
    scalar = values.ndim == 1
    if scalar:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("subject_values must be [subject] or [subject,endpoint]")
    if np.any(~np.isfinite(values)):
        raise ValueError("subject_values must be finite after within-subject aggregation")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or int(iterations) <= 0
    ):
        raise ValueError("iterations must be a positive integer")
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    sample_indices = rng.integers(
        0, values.shape[0], size=(int(iterations), values.shape[0])
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    point = values.mean(axis=0)
    lower = np.quantile(bootstrap_means, alpha, axis=0)
    upper = np.quantile(bootstrap_means, 1.0 - alpha, axis=0)
    probability_positive = (
        1.0 + np.sum(bootstrap_means > 0.0, axis=0)
    ) / (int(iterations) + 1.0)
    if scalar:
        return {
            "subject_count": int(values.shape[0]),
            "iterations": int(iterations),
            "confidence_level": confidence_level,
            "mean": float(point[0]),
            "ci_lower": float(lower[0]),
            "ci_upper": float(upper[0]),
            "bootstrap_probability_positive": float(probability_positive[0]),
        }
    return {
        "subject_count": int(values.shape[0]),
        "endpoint_count": int(values.shape[1]),
        "iterations": int(iterations),
        "confidence_level": confidence_level,
        "mean": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_probability_positive": probability_positive,
    }


def evaluate_q0_q1_by_subject(
    models: Q0Q1Models,
    rows: LaggedCategoricalRows,
) -> Mapping[str, Any]:
    """Apply frozen q0/q1 models separately to each biological subject."""

    if not isinstance(rows, LaggedCategoricalRows):
        raise ValueError("rows must be returned by build_lagged_categorical_rows")
    identities = tuple(sorted(set(rows.subject.tolist())))
    subject_rows = []
    for identity in identities:
        selected = rows.subject == identity
        scores = evaluate_q0_q1(
            models,
            rows.fnirs_target[selected],
            q0_design_eval=rows.q0_design[selected],
            eeg_posterior_eval=rows.eeg_posterior[selected],
        )
        subject_rows.append(
            {
                "subject": identity,
                "row_count": int(np.sum(selected)),
                "log_loss_gain_nats": float(scores["log_loss_gain_nats"]),
                "brier_gain": float(scores["brier_gain"]),
                "q0_log_loss_nats": float(scores["q0"]["log_loss_nats"]),
                "q1_log_loss_nats": float(scores["q1"]["log_loss_nats"]),
                "q0_brier_score": float(scores["q0"]["brier_score"]),
                "q1_brier_score": float(scores["q1"]["brier_score"]),
            }
        )
    return {
        "lag": int(rows.lag),
        "history_steps": int(rows.history_steps),
        "subject_count": len(subject_rows),
        "subject_rows": subject_rows,
        "subject_equal_log_loss_gain_nats": float(
            np.mean([row["log_loss_gain_nats"] for row in subject_rows])
        ),
        "subject_equal_brier_gain": float(
            np.mean([row["brier_gain"] for row in subject_rows])
        ),
    }


def fit_categorical_proper_score_models(*args: Any, **kwargs: Any) -> Q0Q1Models:
    """Alias for :func:`fit_q0_q1`."""

    return fit_q0_q1(*args, **kwargs)


def evaluate_categorical_proper_score_increment(
    *args: Any, **kwargs: Any
) -> Mapping[str, Any]:
    """Alias for :func:`evaluate_q0_q1`."""

    return evaluate_q0_q1(*args, **kwargs)


def fit_q0_q1_probe(*args: Any, **kwargs: Any) -> Q0Q1Models:
    """Alias for :func:`fit_q0_q1` emphasizing the probe boundary."""

    return fit_q0_q1(*args, **kwargs)


def evaluate_q0_q1_probe(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Alias for :func:`evaluate_q0_q1` emphasizing apply-only evaluation."""

    return evaluate_q0_q1(*args, **kwargs)


__all__ = [
    "EPS",
    "LAGGED_COUPLING_SCHEMA_VERSION",
    "CategoricalModel",
    "LaggedCategoricalRows",
    "Q0Q1Models",
    "TokenOrder",
    "build_lagged_categorical_rows",
    "apply_token_order",
    "categorical_proper_scores",
    "conditional_fnirs_given_eeg",
    "conditional_log_lift",
    "dirichlet_conditional",
    "dirichlet_smoothed_conditional",
    "evaluate_categorical_model",
    "evaluate_categorical_proper_score_increment",
    "evaluate_q0_q1",
    "evaluate_q0_q1_by_subject",
    "evaluate_q0_q1_probe",
    "fit_categorical_model",
    "fit_categorical_proper_score_models",
    "fit_q0_q1",
    "fit_q0_q1_probe",
    "fit_token_order",
    "fit_train_only_token_order",
    "fnirs_marginal",
    "fnirs_marginal_from_cooccurrence",
    "fnirs_token_marginal",
    "log_lift",
    "matched_minus_deranged_expected_residual_log_lift",
    "matched_minus_deranged_residual_log_lift",
    "pair_concentration",
    "positive_vs_negative_lag_specificity",
    "predict_categorical_proba",
    "residual_log_lift",
    "residual_log_lift_from_lifts",
    "soft_cooccurrence",
    "soft_cooccurrence_tensor",
    "subject_block_bootstrap",
    "summarize_lagged_coupling",
    "top_pair_jaccard",
    "top_pair_stability",
    "top_pairs",
    "validate_lags",
    "validate_probabilities",
]
