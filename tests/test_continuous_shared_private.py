import inspect
from pathlib import Path

import torch
import yaml
import pandas as pd

from experiments.analyze_continuous_shared_private_latent import (
    seed_average_subjects,
    simultaneous_max_stat_intervals,
)
from experiments.run_continuous_shared_private_latent import (
    ENDPOINTS,
    make_derangement,
    validate_config,
)
from src.tokenizers.continuous_shared_private import (
    ContinuousSharedPrivateModel,
)


def _model() -> ContinuousSharedPrivateModel:
    return ContinuousSharedPrivateModel(
        shared_dim=16,
        eeg_private_dim=16,
        fnirs_private_dim=8,
        encoder_depth=1,
        encoder_num_heads=4,
        encoder_feedforward_dim=32,
        trajectory_decoder_hidden_dim=24,
        raw_decoder_hidden_dim=24,
        dropout=0.0,
    )


def _inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(batch, 6, 4000), torch.randn(batch, 2, 200)


def _nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in module.parameters()
    )


def test_model_shapes_and_continuous_surface():
    torch.manual_seed(1)
    model = _model().eval()
    eeg, fnirs = _inputs()
    with torch.no_grad():
        output = model(eeg, fnirs)

    assert output["eeg_shared"].shape == (2, 10, 16)
    assert output["fnirs_shared"].shape == (2, 10, 16)
    assert output["eeg_private"].shape == (2, 10, 16)
    assert output["fnirs_private"].shape == (2, 10, 8)
    assert output["eeg_driver"].shape == (2, 10, 20)
    assert output["fnirs_driver"].shape == (2, 10, 20)
    assert output["eeg_raw"].shape == eeg.shape
    assert output["fnirs_raw"].shape == fnirs.shape
    forbidden = ("codebook", "quantizer", "assignment", "commitment")
    surface = " ".join(
        [name for name, _ in model.named_modules()]
        + [name for name, _ in model.named_parameters()]
        + list(output)
    ).lower()
    assert not any(token in surface for token in forbidden)


def test_raw_loss_gradient_isolated_from_shared_path():
    torch.manual_seed(2)
    model = _model().train()
    eeg, fnirs = _inputs(batch=1)
    output = model(eeg, fnirs)
    (output["eeg_raw"].square().mean() + output["fnirs_raw"].square().mean()).backward()

    assert not _nonzero_grad(model.eeg_shared_encoder)
    assert not _nonzero_grad(model.fnirs_shared_encoder)
    assert not _nonzero_grad(model.trajectory_decoder)
    assert _nonzero_grad(model.eeg_private_encoder)
    assert _nonzero_grad(model.fnirs_private_encoder)
    assert _nonzero_grad(model.eeg_raw_decoder)
    assert _nonzero_grad(model.fnirs_raw_decoder)


def test_common_trajectory_decoder_and_cross_modal_swap_contract():
    torch.manual_seed(3)
    model = _model().eval()
    eeg, fnirs = _inputs()
    with torch.no_grad():
        output = model(eeg, fnirs)
        eeg_from_fnirs = model.decode_raw(
            "eeg", output["fnirs_shared"], output["eeg_private"]
        )
        fnirs_from_eeg = model.decode_raw(
            "fnirs", output["eeg_shared"], output["fnirs_private"]
        )

    assert tuple(inspect.signature(model.trajectory_decoder.forward).parameters) == (
        "latent",
    )
    assert eeg_from_fnirs.shape == eeg.shape
    assert fnirs_from_eeg.shape == fnirs.shape


def test_invalid_tokens_are_zero_and_do_not_contaminate_valid_tokens():
    torch.manual_seed(4)
    model = _model().eval()
    eeg, fnirs = _inputs(batch=1)
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[:, 4] = False
    changed = eeg.clone()
    changed[..., 1600:2000] = float("nan")
    with torch.no_grad():
        baseline = model(eeg, fnirs, eeg_token_valid_mask=mask)
        result = model(changed, fnirs, eeg_token_valid_mask=mask)

    valid = mask[0]
    torch.testing.assert_close(
        baseline["eeg_shared"][0, valid], result["eeg_shared"][0, valid], atol=0, rtol=0
    )
    torch.testing.assert_close(
        baseline["eeg_private"][0, valid], result["eeg_private"][0, valid], atol=0, rtol=0
    )
    assert torch.equal(result["eeg_shared"][0, 4], torch.zeros(16))
    assert torch.equal(result["eeg_private"][0, 4], torch.zeros(16))
    assert torch.equal(result["eeg_raw"][0, :, 1600:2000], torch.zeros(6, 400))


def test_derangement_preserves_subject_condition_and_is_nonidentity():
    subjects = ["s1", "s1", "s1", "s1", "s2", "s2"]
    conditions = ["a", "a", "b", "b", "a", "a"]
    sample_ids = [f"sample-{index}" for index in range(6)]
    donor = make_derangement(subjects, conditions, sample_ids, seed=17)

    assert all(index != int(donor[index]) for index in range(len(donor)))
    assert all(subjects[index] == subjects[int(donor[index])] for index in range(len(donor)))
    assert all(conditions[index] == conditions[int(donor[index])] for index in range(len(donor)))
    assert torch.equal(
        torch.as_tensor(donor),
        torch.as_tensor(make_derangement(subjects, conditions, sample_ids, seed=17)),
    )


def test_registered_config_has_exact_16_cell_family_and_no_vq():
    path = Path(
        "experiments/configs/physiology_semantic_tokenizer/continuous_shared_private_latent.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)

    assert len(config["tasks"]) * len(ENDPOINTS) == 16
    assert config["objective"]["vector_quantization"] is False
    assert config["source"]["protected_open"] is False


def _synthetic_family(swap_value: float) -> pd.DataFrame:
    rows = []
    for task_index, task in enumerate(
        ("mental_arithmetic", "motor_imagery", "word_generation", "n_back")
    ):
        dataset = "single" if task_index < 2 else "simultaneous"
        prefix = "s" if dataset == "single" else "v"
        for endpoint in ENDPOINTS:
            center = 0.30 if "target" in endpoint else swap_value
            for subject_index in range(5):
                for seed in (1, 2, 3):
                    rows.append(
                        {
                            "task_id": task,
                            "dataset_id": dataset,
                            "seed": seed,
                            "subject": f"{prefix}{subject_index}",
                            "endpoint": endpoint,
                            "value": center + 0.005 * (subject_index - 2),
                        }
                    )
    return pd.DataFrame(rows)


def test_seed_average_precedes_subject_max_stat_and_common_factor_passes():
    averaged = seed_average_subjects(_synthetic_family(swap_value=0.20))
    intervals, draws, labels = simultaneous_max_stat_intervals(
        averaged, iterations=500, confidence_level=0.95, seed=8
    )

    assert len(averaged) == 4 * 4 * 5
    assert len(intervals) == 16
    assert draws.shape == (500, 16)
    assert len(labels) == 16
    assert intervals.strict_cell_pass.all()


def test_independent_modalities_fixture_fails_strict_family():
    averaged = seed_average_subjects(_synthetic_family(swap_value=0.0))
    intervals, _, _ = simultaneous_max_stat_intervals(
        averaged, iterations=500, confidence_level=0.95, seed=9
    )

    swap = intervals[intervals.endpoint.str.contains("swap")]
    assert not swap.strict_cell_pass.any()
    assert not intervals.strict_cell_pass.all()
