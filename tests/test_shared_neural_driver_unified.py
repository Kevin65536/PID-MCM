import numpy as np

from experiments.evaluate_shared_neural_driver_unified import (
    Trial,
    _select_active_hbo,
    _waveform_metrics,
)


def _trial(fnirs: np.ndarray) -> Trial:
    return Trial(
        condition_id="test",
        dataset_id="test",
        subject="s1",
        record_id="r1",
        event_index=0,
        eeg=np.zeros((4000, 2)),
        fnirs=fnirs,
        fnirs_channel_names=("a_HbO", "a_HbR", "b_HbO", "b_HbR"),
        fnirs_roles=("HbO", "HbR", "HbO", "HbR"),
        eeg_artifact_fraction=0.0,
    )


def test_waveform_metrics_exposes_scale_and_shape_separately():
    truth = np.sin(np.linspace(0, 4 * np.pi, 200))
    estimate = 0.2 * truth + 3.0
    metrics = _waveform_metrics(truth, estimate, baseline_n=50)
    assert np.isclose(metrics["amplitude_ratio"], 0.2)
    assert np.isclose(metrics["variance_ratio"], 0.04)
    assert np.isclose(metrics["pcc"], 1.0)
    assert np.isclose(metrics["affine_oracle_r2"], 1.0)
    assert metrics["r2"] < 0.0


def test_active_channel_selection_only_returns_hbo_channels():
    time = np.arange(200) / 10.0
    strong = np.exp(-0.5 * ((time - 8.0) / 2.0) ** 2)
    fnirs = np.column_stack((0.1 * strong, np.zeros(200), 2.0 * strong, np.zeros(200)))
    indices, names, scores = _select_active_hbo(
        [_trial(fnirs), _trial(fnirs * 1.01), _trial(fnirs * 0.99)],
        baseline_duration_s=5.0,
        task_duration_s=10.0,
        count=1,
    )
    assert indices.tolist() == [2]
    assert names == ("b_HbO",)
    assert scores.shape == (2,)

