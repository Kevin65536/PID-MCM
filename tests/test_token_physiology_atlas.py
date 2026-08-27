import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import analyze_token_physiology_atlas as atlas_cli
from experiments.scripts.export_physiology_semantic_tokens import (
    compact_export_to_assignments,
)
from src.analysis.token_physiology_atlas import (
    ATLAS_SCHEMA_VERSION,
    build_token_physiology_atlas,
    load_token_export,
    prepare_measurement_feature_caches,
)


def _atlas_config() -> dict:
    return {
        "schema_version": "physiology_token_atlas_config_v1",
        "analysis": {
            "splits": ["train", "val"],
            "seed": 17,
        },
        "input": {
            "unit": "canonical_robust_sd",
            "sample_rate_hz": {"eeg": 200.0, "fnirs": 10.0},
        },
        "features": {
            "minimum_valid_fraction": 1.0,
            "eeg": {
                "reference_band_hz": [1.0, 45.0],
                "bands_hz": {"alpha": [8.0, 13.0]},
                "psd_window": "hann_symmetric",
                "psd_detrend": "constant",
            },
            "fnirs": {
                "local_morphology": True,
                "band_power": False,
            },
        },
        "profiles": {
            "min_count": 1,
            "min_subjects": 1,
            "rare_count": 1,
            "bootstrap": {
                "iterations": 0,
                "confidence": 0.95,
                "unit": "subject",
            },
        },
        "associations": {
            "state_fields": ["label"],
            "max_metadata_categories": 16,
        },
        "sequence": {
            "lags": [-1, 0, 1],
            "patch_duration_s": 2.0,
            "null": {"permutations": 0},
        },
        "stability": {
            "profile_type": "hard",
            "min_feature_overlap": 2,
        },
        "figures": {
            "formats": ["png"],
            "dpi": 72,
            "top_tokens": 3,
        },
    }


def _sample_order_sha256(sample_ids: np.ndarray) -> str:
    return hashlib.sha256(
        "\n".join(sample_ids.astype(str).tolist()).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_synthetic_v3_export(
    directory: Path,
    *,
    split: str,
    sample_count: int,
    seed: int,
) -> Path:
    """Write a small, fully numeric v3 export accepted with allow_pickle=False."""

    rng = np.random.default_rng(seed)
    token_count = 3
    codebook_size = 3
    latent_size = 4
    hard_ids = (
        np.arange(sample_count)[:, None] + np.arange(token_count)[None, :]
    ) % codebook_size
    valid_mask = np.ones((sample_count, token_count), dtype=bool)

    posterior = np.full(
        (sample_count, token_count, codebook_size),
        0.05,
        dtype=np.float32,
    )
    rows = np.arange(sample_count)[:, None]
    positions = np.arange(token_count)[None, :]
    posterior[rows, positions, hard_ids] = 0.90
    posterior /= posterior.sum(axis=-1, keepdims=True)

    eeg_time = np.arange(400, dtype=np.float64) / 200.0
    eeg_patches = np.empty(
        (sample_count, token_count, 6, 400), dtype=np.float32
    )
    token_frequencies = np.asarray([3.0, 10.0, 20.0])
    for sample in range(sample_count):
        for position in range(token_count):
            token_id = hard_ids[sample, position]
            for channel in range(6):
                phase = 0.15 * sample + 0.11 * channel
                signal = (
                    (1.0 + 0.08 * channel)
                    * np.sin(
                        2.0
                        * np.pi
                        * token_frequencies[token_id]
                        * eeg_time
                        + phase
                    )
                    + 0.15
                    * np.sin(2.0 * np.pi * 10.0 * eeg_time + 0.3 * position)
                    + 0.03 * sample * eeg_time
                    + rng.normal(0.0, 0.025, size=eeg_time.shape)
                )
                eeg_patches[sample, position, channel] = signal

    fnirs_time = np.arange(20, dtype=np.float64) / 10.0
    fnirs_patches = np.empty(
        (sample_count, token_count, 2, 20), dtype=np.float32
    )
    token_slopes = np.asarray([-0.35, 0.05, 0.40])
    for sample in range(sample_count):
        for position in range(token_count):
            token_id = hard_ids[sample, position]
            slope = token_slopes[token_id] + 0.012 * sample
            hbo = (
                slope * fnirs_time
                + 0.12 * np.square(fnirs_time - 0.9)
                + 0.025 * position
                + rng.normal(0.0, 0.012, size=fnirs_time.shape)
            )
            hbr = (
                -0.55 * slope * fnirs_time
                + 0.04 * np.sin(np.pi * fnirs_time)
                - 0.01 * sample
                + rng.normal(0.0, 0.012, size=fnirs_time.shape)
            )
            fnirs_patches[sample, position, 0] = hbo
            fnirs_patches[sample, position, 1] = hbr

    lookup = np.asarray(
        [
            [1.0, 0.0, 0.2, -0.1],
            [0.0, 1.0, -0.2, 0.3],
            [-0.5, 0.2, 1.0, 0.1],
        ],
        dtype=np.float32,
    )
    semantic_latent = (
        lookup[hard_ids]
        + rng.normal(
            0.0,
            0.08,
            size=(sample_count, token_count, latent_size),
        )
    ).astype(np.float32)
    codebook_embedding = (
        lookup[hard_ids]
        + rng.normal(
            0.0,
            0.01,
            size=(sample_count, token_count, latent_size),
        )
    ).astype(np.float32)
    expected_embedding = (posterior @ lookup).astype(np.float32)
    residual = np.stack(
        (
            hard_ids.astype(np.float32) / max(codebook_size - 1, 1),
            np.broadcast_to(
                np.arange(token_count, dtype=np.float32)[None, :],
                hard_ids.shape,
            ),
        ),
        axis=-1,
    )

    sample_ids = np.asarray(
        [f"{split}_sample_{index:02d}" for index in range(sample_count)],
        dtype=np.str_,
    )
    subject_ids = np.asarray(
        [
            f"{split}_subject_{index // 2:02d}"
            for index in range(sample_count)
        ],
        dtype=np.str_,
    )
    selected_channels = np.tile(
        np.asarray(["F3", "F4", "C3", "C4", "P3", "P4"], dtype=np.str_),
        (sample_count, 1),
    )
    entropy = -np.sum(
        posterior * np.log(np.clip(posterior, 1e-12, None)), axis=-1
    ).astype(np.float32)
    sorted_posterior = np.sort(posterior, axis=-1)
    margin = (sorted_posterior[..., -1] - sorted_posterior[..., -2]).astype(
        np.float32
    )
    latent_code_l2 = np.linalg.norm(
        semantic_latent - codebook_embedding, axis=-1
    ).astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "sample_id": sample_ids,
        "subject_key": subject_ids,
        "dataset_id": np.repeat(
            np.asarray(["synthetic"], dtype=np.str_), sample_count
        ),
        "task_namespace": np.asarray(
            [
                "task/a" if index % 2 == 0 else "task/b"
                for index in range(sample_count)
            ],
            dtype=np.str_,
        ),
        "record_id": np.asarray(
            [f"record_{index // 2}" for index in range(sample_count)],
            dtype=np.str_,
        ),
        "dependency_group_id": subject_ids.copy(),
        "anchor": np.arange(sample_count, dtype=np.int64) * 400,
        "label": np.arange(sample_count, dtype=np.int64) % 2,
        "selected_eeg_channels": selected_channels,
    }
    for modality, patches in (
        ("eeg", eeg_patches),
        ("fnirs", fnirs_patches),
    ):
        modality_jitter = (
            0.0 if modality == "eeg" else 0.015
        )
        arrays.update(
            {
                f"{modality}_hard_ids": hard_ids.astype(np.int64),
                f"{modality}_posterior": posterior.astype(np.float32),
                f"{modality}_token_valid_mask": valid_mask.copy(),
                f"{modality}_patches": patches,
                f"{modality}_semantic_latent": (
                    semantic_latent + modality_jitter
                ).astype(np.float32),
                f"{modality}_codebook_embedding": (
                    codebook_embedding + modality_jitter
                ).astype(np.float32),
                f"{modality}_codebook": (
                    lookup + modality_jitter
                ).astype(np.float32),
                f"{modality}_expected_embedding": (
                    expected_embedding + modality_jitter
                ).astype(np.float32),
                f"{modality}_residual": residual.astype(np.float32),
                f"{modality}_posterior_entropy": entropy.copy(),
                f"{modality}_posterior_top1_top2_margin": margin.copy(),
                f"{modality}_latent_code_l2": latent_code_l2.copy(),
                f"{modality}_expected_reconstruction_mse": (
                    0.04 + 0.01 * hard_ids + modality_jitter
                ).astype(np.float32),
                f"{modality}_hard_reconstruction_mse": (
                    0.06 + 0.012 * hard_ids + modality_jitter
                ).astype(np.float32),
            }
        )

    path = directory / f"{split}.npz"
    np.savez_compressed(path, **arrays)
    manifest = {
        "schema": "physiology_semantic_export_v3",
        "split": split,
        "count": sample_count,
        "sample_key_array": "sample_id",
        "sample_order_sha256": _sample_order_sha256(sample_ids),
        "checkpoint_sha256": "synthetic-checkpoint-sha256",
        "include_patches": True,
        "include_assignment_diagnostics": True,
        "include_reconstruction_diagnostics": True,
        "array_schema": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cache_statuses(summary: dict) -> set[str]:
    return {
        modality_summary["measurement_cache"]["status"]
        for split_summary in summary["modalities"].values()
        for modality_summary in split_summary.values()
    }


def test_build_no_plots_writes_standard_atlas_and_reuses_measurement_cache(
    tmp_path,
):
    train = _write_synthetic_v3_export(
        tmp_path, split="train", sample_count=6, seed=10
    )
    validation = _write_synthetic_v3_export(
        tmp_path, split="val", sample_count=4, seed=20
    )
    exports = {"train": train, "val": validation}
    cache_dir = tmp_path / "shared_measurement_cache"
    first_output = tmp_path / "atlas_first"

    result = build_token_physiology_atlas(
        exports,
        first_output,
        config=_atlas_config(),
        measurement_cache_dir=cache_dir,
        bootstrap_iterations=0,
        coupling_permutations=0,
        plots=False,
    )

    assert result == first_output.resolve()
    expected_artifacts = {
        "manifest.json",
        "summary.json",
        "stability.json",
        "information_ledger.json",
        "sequence_summary.json",
        "token_analysis_manifests.json",
        "tables/token_support.csv",
        "tables/token_profiles.csv",
        "tables/token_feature_distributions.csv",
        "tables/token_channel_feature_distributions.csv",
        "tables/hard_soft_profile_differences.csv",
        "tables/state_associations.csv",
        "tables/metadata_associations.csv",
        "tables/assignment_diagnostics.csv",
        "tables/cross_modal_lags.csv",
        "tables/token_exemplars.jsonl",
        "arrays/train_sequence_counts.npz",
        "arrays/val_sequence_counts.npz",
    }
    assert expected_artifacts <= {
        str(path.relative_to(first_output))
        for path in first_output.rglob("*")
        if path.is_file()
    }
    assert not (first_output / "figures").exists()

    summary = json.loads(
        (first_output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema"] == ATLAS_SCHEMA_VERSION
    assert summary["status"] == "complete"
    assert summary["splits"] == ["train", "val"]
    assert summary["protected_test_opened"] is False
    assert _cache_statuses(summary) == {"miss"}

    manifest = json.loads(
        (first_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == ATLAS_SCHEMA_VERSION
    assert manifest["completed"] is True
    assert manifest["raw_patches_copied_to_atlas"] is False
    inventory = {row["path"] for row in manifest["artifacts"]}
    assert expected_artifacts - {"manifest.json"} <= inventory
    ledger = json.loads(
        (first_output / "information_ledger.json").read_text(encoding="utf-8")
    )
    for modality in ("eeg", "fnirs"):
        representation_names = set(
            ledger["modalities"][modality]["representations"]
        )
        assert {
            "continuous_latent",
            "hard_one_hot",
            "posterior",
            "codebook_embedding",
            "expected_embedding",
            "residual",
            "continuous_semantic_plus_residual",
        } <= representation_names

    profile_rows = _read_csv(
        first_output / "tables" / "token_profiles.csv"
    )
    eeg_features = {
        row["feature_name"]
        for row in profile_rows
        if row["modality"] == "eeg"
    }
    fnirs_features = {
        row["feature_name"]
        for row in profile_rows
        if row["modality"] == "fnirs"
    }
    assert any("log_relative_power_alpha" in name for name in eeg_features)
    alpha_profile = next(
        row
        for row in profile_rows
        if row["modality"] == "eeg"
        and row["feature_name"] == "channel_mean/log_relative_power_alpha"
    )
    assert alpha_profile["feature_unit"] == "log_fraction"
    assert fnirs_features
    assert all("power" not in name.lower() for name in fnirs_features)
    distribution_rows = _read_csv(
        first_output / "tables" / "token_feature_distributions.csv"
    )
    alpha_distributions = [
        row
        for row in distribution_rows
        if row["modality"] == "eeg"
        and "log_relative_power_alpha" in row["feature_name"]
        and int(row["finite_patch_count"]) > 0
    ]
    assert alpha_distributions
    assert all(row["patch_q25"] and row["patch_q75"] for row in alpha_distributions)
    channel_rows = _read_csv(
        first_output / "tables" / "token_channel_feature_distributions.csv"
    )
    assert any(
        row["modality"] == "eeg"
        and row["channel_name"] == "F3"
        and row["feature_name"] == "log_absolute_power_alpha"
        for row in channel_rows
    )
    assert all(
        "power" not in row["feature_name"].lower()
        for row in channel_rows
        if row["modality"] == "fnirs"
    )
    hard_soft_rows = _read_csv(
        first_output / "tables" / "hard_soft_profile_differences.csv"
    )
    assert hard_soft_rows
    assert all(
        row["interpretation"].startswith("assignment-boundary")
        for row in hard_soft_rows
    )

    with np.load(
        first_output / "arrays" / "train_sequence_counts.npz",
        allow_pickle=False,
    ) as sequence:
        assert sequence["eeg_transition_counts"].shape == (3, 3)
        assert sequence["fnirs_transition_counts"].shape == (3, 3)
        assert sequence["cross_modal_lag_counts"].shape == (3, 3, 3)
    sequence_summary = json.loads(
        (first_output / "sequence_summary.json").read_text(encoding="utf-8")
    )
    eeg_sequence_row = sequence_summary["train"]["eeg"]["token_rows"][0]
    assert eeg_sequence_row["dwell_mean_seconds"] == pytest.approx(
        eeg_sequence_row["run_length_mean_tokens"] * 2.0
    )
    assert (
        sequence_summary["train"]["eeg"]["dwell_definition"]
        == "contiguous within-window run length multiplied by patch duration"
    )

    cache_archives = sorted(cache_dir.glob("*.npz"))
    assert len(cache_archives) == 4
    assert all(
        path.with_suffix(".manifest.json").is_file()
        for path in cache_archives
    )

    second_output = tmp_path / "atlas_second"
    build_token_physiology_atlas(
        exports,
        second_output,
        config=_atlas_config(),
        measurement_cache_dir=cache_dir,
        bootstrap_iterations=0,
        coupling_permutations=0,
        plots=False,
    )
    second_summary = json.loads(
        (second_output / "summary.json").read_text(encoding="utf-8")
    )
    assert _cache_statuses(second_summary) == {"hit"}
    assert second_summary["protected_test_opened"] is False
    with pytest.raises(FileExistsError, match="not empty"):
        build_token_physiology_atlas(
            exports,
            first_output,
            config=_atlas_config(),
            measurement_cache_dir=cache_dir,
            bootstrap_iterations=0,
            coupling_permutations=0,
            plots=False,
        )


def test_figure_sidecars_record_selection_color_scale_and_marker_semantics(
    tmp_path,
):
    train = _write_synthetic_v3_export(
        tmp_path, split="train", sample_count=6, seed=25
    )
    output = tmp_path / "atlas_with_figures"

    build_token_physiology_atlas(
        {"train": train},
        output,
        config=_atlas_config(),
        measurement_cache_dir=tmp_path / "figure_measurement_cache",
        bootstrap_iterations=0,
        coupling_permutations=0,
        plots=True,
        information_ledger=False,
    )

    figure_dir = output / "figures"
    heatmap_manifest = json.loads(
        (figure_dir / "train_eeg_phenotype_heatmap.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    heatmap_spec = heatmap_manifest["provenance"]["visualization"]
    assert heatmap_spec["kind"] == "hard_token_phenotype_heatmap"
    assert len(heatmap_spec["selected_token_ids"]) == 3
    assert heatmap_spec["color_scale"]["center"] == 0.0
    assert (
        heatmap_spec["color_scale"]["vmin"]
        == -heatmap_spec["color_scale"]["vmax"]
    )

    codebook_stem = (
        figure_dir
        / "train_eeg_codebook_channel_mean_log_relative_power_alpha"
    )
    codebook_manifest = json.loads(
        codebook_stem.with_suffix(".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    codebook_spec = codebook_manifest["provenance"]["visualization"]
    assert codebook_spec["projection"] == "centered unscaled PCA via SVD"
    assert codebook_spec["embedding_shape"] == [3, 4]
    assert "% variance" in codebook_manifest["figure"]["axes"][0]["xlabel"]
    assert not list(figure_dir.glob("*.alt.txt"))


def test_protected_test_export_is_rejected_without_explicit_authorization(
    tmp_path,
):
    test_export = _write_synthetic_v3_export(
        tmp_path, split="test", sample_count=4, seed=30
    )

    with pytest.raises(ValueError, match="sealed"):
        load_token_export(test_export)
    with pytest.raises(ValueError, match="allow_test=True"):
        build_token_physiology_atlas(
            {"test": test_export},
            tmp_path / "atlas_test",
            config=_atlas_config(),
            measurement_cache_dir=tmp_path / "cache",
            bootstrap_iterations=0,
            coupling_permutations=0,
            plots=False,
        )


def test_static_codebook_preserves_inactive_tokens_without_full_posterior(
    tmp_path,
):
    export = _write_synthetic_v3_export(
        tmp_path, split="train", sample_count=4, seed=35
    )
    with np.load(export, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for modality in ("eeg", "fnirs"):
        arrays.pop(f"{modality}_posterior")
        arrays[f"{modality}_codebook"] = np.vstack(
            (
                arrays[f"{modality}_codebook"],
                np.asarray([[0.2, -0.4, 0.6, -0.8]], dtype=np.float32),
            )
        )
    np.savez_compressed(export, **arrays)
    manifest_path = export.with_suffix(export.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["array_schema"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "atlas_inactive_static_code"
    build_token_physiology_atlas(
        {"train": export},
        output,
        config=_atlas_config(),
        measurement_cache_dir=tmp_path / "inactive_cache",
        bootstrap_iterations=0,
        coupling_permutations=0,
        plots=False,
        information_ledger=False,
    )

    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    for modality in ("eeg", "fnirs"):
        modality_summary = summary["modalities"]["train"][modality]
        assert modality_summary["codebook_size"] == 4
        assert modality_summary["active_token_count"] == 3
        assert modality_summary["soft_profile_available"] is False
    support = _read_csv(output / "tables" / "token_support.csv")
    inactive = [
        row
        for row in support
        if int(row["token_id"]) == 3
    ]
    assert len(inactive) == 2
    assert all(row["inactive"] == "True" for row in inactive)


def test_compact_assignment_export_references_verified_measurement_caches(
    tmp_path,
):
    export = _write_synthetic_v3_export(
        tmp_path, split="train", sample_count=6, seed=40
    )
    config = _atlas_config()
    measurement_cache_dir = tmp_path / "compact_measurements"
    original_size = export.stat().st_size
    original_sha256 = _file_sha256(export)

    references = prepare_measurement_feature_caches(
        export,
        config=config,
        measurement_cache_dir=measurement_cache_dir,
        expected_split="train",
    )

    assert set(references) == {"eeg", "fnirs"}
    for modality, reference in references.items():
        cache_path = Path(reference["path"])
        cache_manifest_path = cache_path.with_suffix(".manifest.json")
        cache_manifest = json.loads(
            cache_manifest_path.read_text(encoding="utf-8")
        )
        assert cache_path.is_file()
        assert reference["npz_sha256"] == _file_sha256(cache_path)
        assert cache_manifest["npz_sha256"] == reference["npz_sha256"]
        assert (
            cache_manifest["measurement_cache_key"]
            == reference["measurement_cache_key"]
        )
        assert (
            cache_manifest["feature_spec_hash"]
            == reference["feature_spec_hash"]
        )
        assert cache_manifest["modality"] == modality
        assert (
            cache_manifest["source_sample_order_sha256"]
            == reference["source_sample_order_sha256"]
        )

    assert compact_export_to_assignments(
        export,
        measurement_caches=references,
    ) == export.resolve()

    compact_size = export.stat().st_size
    compact_sha256 = _file_sha256(export)
    assert compact_size < original_size
    assert compact_sha256 != original_sha256

    compact_payload, compact_manifest = load_token_export(
        export,
        expected_split="train",
    )
    assert "eeg_patches" not in compact_payload
    assert "fnirs_patches" not in compact_payload
    assert compact_manifest["include_patches"] is False
    assert compact_manifest["raw_patches_stored"] is False
    assert (
        compact_manifest["compacted_after_measurement_feature_extraction"]
        is True
    )
    assert compact_manifest["precompaction_sha256"] == original_sha256
    assert compact_manifest["measurement_caches"] == references
    assert "eeg_patches" not in compact_manifest["arrays"]
    assert "fnirs_patches" not in compact_manifest["arrays"]
    for modality, reference in compact_manifest["measurement_caches"].items():
        cache_path = Path(reference["path"])
        assert reference["npz_sha256"] == _file_sha256(cache_path)
        assert (
            reference["source_sample_order_sha256"]
            == compact_manifest["sample_order_sha256"]
        )
        assert modality in {"eeg", "fnirs"}

    output = tmp_path / "atlas_from_compact"
    build_token_physiology_atlas(
        {"train": export},
        output,
        config=config,
        measurement_cache_dir=measurement_cache_dir,
        bootstrap_iterations=0,
        coupling_permutations=0,
        plots=False,
        information_ledger=False,
    )

    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert _cache_statuses(summary) == {"hit"}
    for modality in ("eeg", "fnirs"):
        cache_summary = summary["modalities"]["train"][modality][
            "measurement_cache"
        ]
        assert cache_summary["source"] == "assignment_manifest_reference"
        assert (
            cache_summary["measurement_cache_key"]
            == references[modality]["measurement_cache_key"]
        )
    atlas_manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        atlas_manifest["input_exports"]["train"]["sha256"]
        == compact_sha256
    )
    assert (
        atlas_manifest["input_exports"]["train"]["manifest"][
            "precompaction_sha256"
        ]
        == original_sha256
    )


@pytest.mark.parametrize(
    ("fault", "error_match"),
    [
        ("sample_ids", "sample IDs do not align"),
        ("token_mask", "token mask does not align"),
        ("channel_identity", "channel identities do not align"),
        ("reference_hash", "measurement-cache reference failed"),
    ],
)
def test_compaction_fails_closed_without_modifying_raw_export(
    tmp_path,
    fault,
    error_match,
):
    export = _write_synthetic_v3_export(
        tmp_path, split="train", sample_count=6, seed=50
    )
    manifest_path = export.with_suffix(export.suffix + ".manifest.json")
    export_sha256_before = _file_sha256(export)
    manifest_sha256_before = _file_sha256(manifest_path)
    export_size_before = export.stat().st_size
    references = prepare_measurement_feature_caches(
        export,
        config=_atlas_config(),
        measurement_cache_dir=tmp_path / "measurements",
        expected_split="train",
    )
    references = json.loads(json.dumps(references))

    if fault == "reference_hash":
        references["eeg"]["npz_sha256"] = "0" * 64
    else:
        cache_path = Path(references["eeg"]["path"])
        cache_manifest_path = cache_path.with_suffix(".manifest.json")
        with np.load(cache_path, allow_pickle=False) as archive:
            cache_payload = {
                name: archive[name].copy() for name in archive.files
            }
        if fault == "sample_ids":
            cache_payload["sample_ids"] = cache_payload["sample_ids"][::-1]
        elif fault == "token_mask":
            cache_payload["token_valid_mask"][0, 0] = (
                not cache_payload["token_valid_mask"][0, 0]
            )
        elif fault == "channel_identity":
            cache_payload["selected_eeg_channels"][0, 0] = "XX"
        else:  # pragma: no cover - parametrization is exhaustive
            raise AssertionError(f"unknown fault {fault}")
        np.savez_compressed(cache_path, **cache_payload)
        tampered_cache_sha256 = _file_sha256(cache_path)
        cache_manifest = json.loads(
            cache_manifest_path.read_text(encoding="utf-8")
        )
        cache_manifest["npz_sha256"] = tampered_cache_sha256
        cache_manifest_path.write_text(
            json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        references["eeg"]["npz_sha256"] = tampered_cache_sha256

    with pytest.raises(ValueError, match=error_match):
        compact_export_to_assignments(
            export,
            measurement_caches=references,
        )

    assert _file_sha256(export) == export_sha256_before
    assert _file_sha256(manifest_path) == manifest_sha256_before
    assert export.stat().st_size == export_size_before
    with np.load(export, allow_pickle=False) as archive:
        assert "eeg_patches" in archive.files
        assert "fnirs_patches" in archive.files
    unchanged_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    assert unchanged_manifest["include_patches"] is True
    assert "precompaction_sha256" not in unchanged_manifest
    assert "measurement_caches" not in unchanged_manifest


def test_assignment_cache_identity_rejects_scope_config_and_npz_drift(
    tmp_path,
):
    export = _write_synthetic_v3_export(
        tmp_path, split="train", sample_count=6, seed=60
    )
    manifest_path = export.with_suffix(export.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "config_sha256": "config-a",
            "analysis_view_contract_sha256": "view-a",
            "max_batches": 1,
            "replay_scope": "first_1_batches",
            "npz_sha256": _file_sha256(export),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    common = {
        "path": export,
        "split": "train",
        "checkpoint_sha256": "synthetic-checkpoint-sha256",
        "analysis_view_contract_sha256": "view-a",
        "atlas_config": _atlas_config(),
        "measurement_cache_dir": tmp_path / "measurements",
        "required_extra_fields": (),
        "allow_test": False,
    }

    assert atlas_cli._assignment_cache_valid(
        config_sha256="config-a",
        max_batches=1,
        **common,
    )
    assert not atlas_cli._assignment_cache_valid(
        config_sha256="config-a",
        max_batches=None,
        **common,
    )
    assert not atlas_cli._assignment_cache_valid(
        config_sha256="config-b",
        max_batches=1,
        **common,
    )
    assert not atlas_cli._assignment_cache_valid(
        config_sha256="config-a",
        max_batches=1,
        **{**common, "analysis_view_contract_sha256": "view-b"},
    )

    with export.open("ab") as handle:
        handle.write(b"tampered-assignment-cache")
    assert not atlas_cli._assignment_cache_valid(
        config_sha256="config-a",
        max_batches=1,
        **common,
    )


def test_checkpoint_cache_paths_separate_scope_model_and_state_contracts(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    config_hash = {"value": "a" * 64}
    emitted_outputs = []

    monkeypatch.setattr(atlas_cli, "_sha256", lambda path: "c" * 64)
    monkeypatch.setattr(
        atlas_cli,
        "_resolved_model_config",
        lambda checkpoint_path, model_config_path: (
            {"model": "synthetic"},
            config_hash["value"],
        ),
    )
    monkeypatch.setattr(
        atlas_cli,
        "_assignment_cache_valid",
        lambda *args, **kwargs: False,
    )

    def fake_export(args):
        emitted_outputs.append(Path(args.output))
        return Path(args.output)

    monkeypatch.setattr(atlas_cli, "run_export", fake_export)
    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        model_config=None,
        max_batches=1,
        allow_test=False,
        force=False,
        device="cpu",
    )
    cache_root = tmp_path / "cache"
    smoke = atlas_cli._exports_from_checkpoint(
        args,
        splits=("train",),
        cache_root=cache_root,
        atlas_config=_atlas_config(),
    )["train"]

    args.max_batches = None
    full = atlas_cli._exports_from_checkpoint(
        args,
        splits=("train",),
        cache_root=cache_root,
        atlas_config=_atlas_config(),
    )["train"]

    args.max_batches = 1
    config_hash["value"] = "b" * 64
    different_config = atlas_cli._exports_from_checkpoint(
        args,
        splits=("train",),
        cache_root=cache_root,
        atlas_config=_atlas_config(),
    )["train"]

    state_config = _atlas_config()
    state_config["associations"]["state_fields"] = ["label", "arousal_state"]
    different_state_contract = atlas_cli._exports_from_checkpoint(
        args,
        splits=("train",),
        cache_root=cache_root,
        atlas_config=state_config,
    )["train"]

    assert len(
        {smoke, full, different_config, different_state_contract}
    ) == 4
    assert "first_1_batches" in smoke.parts
    assert "full_split" in full.parts
    assert ("a" * 24) in smoke.parts
    assert ("b" * 24) in different_config.parts
    assert emitted_outputs == [
        smoke,
        full,
        different_config,
        different_state_contract,
    ]


def test_cli_parses_export_mapping_and_forwards_no_plot_overrides(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "atlas.yaml"
    config_path.write_text(
        yaml.safe_dump(_atlas_config(), sort_keys=False),
        encoding="utf-8",
    )
    train = tmp_path / "synthetic_train.npz"
    validation = tmp_path / "synthetic_val.npz"
    output = tmp_path / "cli_atlas"
    cache_root = tmp_path / "cli_cache"
    args = atlas_cli.parse_args(
        [
            "--atlas-config",
            str(config_path),
            "--export",
            f"train={train}",
            "--export",
            f"val={validation}",
            "--output-dir",
            str(output),
            "--measurement-cache-dir",
            str(cache_root),
            "--splits",
            "train,val",
            "--bootstrap-iterations",
            "0",
            "--coupling-permutations",
            "0",
            "--formats",
            "png,svg",
            "--no-plots",
        ]
    )
    assert args.export == [("train", train), ("val", validation)]
    assert args.splits == ("train", "val")
    assert args.formats == ("png", "svg")
    assert args.no_plots is True

    captured = {}

    def fake_build(exports, output_dir, **kwargs):
        captured["exports"] = exports
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        return Path(output_dir).resolve()

    monkeypatch.setattr(
        atlas_cli, "build_token_physiology_atlas", fake_build
    )
    result = atlas_cli.run(args)

    assert result == output.resolve()
    assert captured["exports"] == {
        "train": train.resolve(),
        "val": validation.resolve(),
    }
    assert captured["output_dir"] == output.resolve()
    assert captured["kwargs"]["measurement_cache_dir"] == (
        cache_root / "measurements"
    ).resolve()
    assert captured["kwargs"]["bootstrap_iterations"] == 0
    assert captured["kwargs"]["coupling_permutations"] == 0
    assert captured["kwargs"]["formats"] == ("png", "svg")
    assert captured["kwargs"]["plots"] is False
    assert captured["kwargs"]["information_ledger"] is False
    assert captured["kwargs"]["allow_test"] is False
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["splits"] == ["train", "val"]
    assert cli_report["protected_test_opened"] is False
