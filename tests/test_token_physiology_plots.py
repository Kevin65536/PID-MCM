import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.token_physiology_plots import (
    INSUFFICIENT_SUPPORT_COLOR,
    MISSING_COLOR,
    plot_codebook_embedding_colored,
    plot_token_feature_heatmap,
    plot_token_feature_profile_ci,
    plot_token_support,
    save_figure_atomic,
)


def test_heatmap_centres_shared_scale_and_marks_nan_and_support():
    profiles = np.array(
        [
            [-2.0, 1.0, np.nan],
            [0.2, -0.3, 0.1],
            [4.0, 2.0, -1.0],
        ]
    )
    fig, ax = plot_token_feature_heatmap(
        profiles,
        token_ids=[7, 11, 19],
        feature_names=["alpha", "beta", "RMS"],
        support_flags=[True, False, True],
        value_label="Within-subject z-score",
    )
    try:
        assert fig.get_layout_engine() is not None
        assert ax.get_xlabel() == "Feature"
        assert ax.get_ylabel() == "Token ID"
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "alpha",
            "beta",
            "RMS",
        ]
        image = ax.images[0]
        assert image.norm.vcenter == 0.0
        assert image.norm.vmin == -image.norm.vmax
        assert image.norm.vmax == pytest.approx(4.0)
        np.testing.assert_allclose(
            image.cmap.get_bad(),
            mcolors.to_rgba(MISSING_COLOR),
        )
        assert any(text.get_text() == "\u00d7" for text in ax.texts)
        support_patches = [
            patch
            for patch in ax.patches
            if patch.get_hatch() == "///"
            and mcolors.to_hex(patch.get_facecolor()).upper()
            == INSUFFICIENT_SUPPORT_COLOR.upper()
        ]
        assert len(support_patches) == profiles.shape[1]
    finally:
        plt.close(fig)


def test_support_plot_has_zero_baseline_threshold_and_non_colour_cue():
    fig, ax = plot_token_support(
        token_ids=[0, 1, 2],
        support_values=[100, 4, np.nan],
        minimum_support=10,
    )
    try:
        assert ax.get_xlabel() == "Token ID"
        assert ax.get_ylabel() == "Assigned patch count"
        assert ax.get_ylim()[0] == 0.0
        assert any(bar.get_hatch() == "///" for bar in ax.patches)
        assert any(bar.get_hatch() == "xx" for bar in ax.patches)
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "Insufficient support" in legend_labels
        assert "Missing support value" in legend_labels
        assert "Minimum support = 10" in legend_labels
    finally:
        plt.close(fig)


def test_profile_ci_names_interval_and_suppresses_unsupported_interval():
    fig, ax = plot_token_feature_profile_ci(
        token_ids=[3, 5, 8],
        estimates=[0.2, -0.5, np.nan],
        interval_lower=[0.1, -0.8, np.nan],
        interval_upper=[0.4, -0.2, np.nan],
        support_flags=[True, False, True],
        feature_name="Alpha power",
        units="log(µV²)",
        estimator_label="Median",
        interval_label="95% subject-bootstrap confidence interval",
    )
    try:
        assert ax.get_xlabel() == "Token ID"
        assert ax.get_ylabel() == "Alpha power (log(µV²))"
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert (
            "Median; 95% subject-bootstrap confidence interval"
            in labels
        )
        assert any("interval suppressed" in label for label in labels)
        assert any("Missing estimate or interval" in text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


def test_codebook_embedding_uses_labels_colorbar_and_marker_redundancy():
    embedding = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ]
    )
    fig, ax = plot_codebook_embedding_colored(
        embedding,
        feature_values=[-2.0, 0.0, 3.0, 1.0],
        token_ids=[0, 1, 2, 3],
        support_flags=[True, True, True, False],
        feature_name="Beta enrichment",
        units="z",
        center=0.0,
    )
    try:
        assert ax.get_xlabel() == "Codebook PCA component 1 (65.6% variance)"
        assert ax.get_ylabel() == "Codebook PCA component 2 (30.8% variance)"
        assert fig.axes[-1].get_ylabel() == "Beta enrichment (z)"
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any(label.startswith("Below reference") for label in labels)
        assert any(label.startswith("Above reference") for label in labels)
        assert "Insufficient token support" in labels
    finally:
        plt.close(fig)


def test_atomic_export_writes_formats_manifest_and_refuses_overwrite(
    tmp_path: Path,
):
    fig, ax = plt.subplots(layout="constrained")
    ax.plot([0, 1], [0, 1], marker="o")
    ax.set(xlabel="Time (s)", ylabel="Amplitude (µV)", title="Example")
    stem = tmp_path / "token_profile"
    try:
        artifacts = save_figure_atomic(
            fig,
            stem,
            formats=("png", "pdf", "svg"),
            dpi=120,
            provenance={
                "source": "synthetic test data",
                "uncertainty": "none; raw observations",
            },
        )
        assert {path.suffix for path in artifacts.figure_paths} == {
            ".png",
            ".pdf",
            ".svg",
        }
        assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.figure_paths)
        assert not list(tmp_path.glob("*.alt.txt"))
        assert artifacts.manifest_path is not None
        manifest = json.loads(
            artifacts.manifest_path.read_text(encoding="utf-8")
        )
        assert manifest["schema"] == "token_physiology_figure_manifest_v2"
        assert manifest["figure"]["axes"][0]["xlabel"] == "Time (s)"
        assert len(manifest["export"]["outputs"]) == 3
        assert all(
            len(output["sha256"]) == 64
            for output in manifest["export"]["outputs"]
        )

        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            save_figure_atomic(
                fig,
                stem,
                formats=("png", "pdf", "svg"),
            )
    finally:
        plt.close(fig)
