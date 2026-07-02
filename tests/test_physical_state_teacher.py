import torch

from src.teachers.physical_state_teacher import PhysicalStateTeacher


def _teacher_batch(batch_size=2):
    state_time = torch.arange(200, dtype=torch.float32) / 10.0
    eeg_time = torch.arange(4000, dtype=torch.float32) / 200.0
    state = torch.stack(
        [
            2.0 * state_time + 1.0,
            -state_time,
            torch.full_like(state_time, 3.0),
            0.5 * state_time,
            state_time,
        ],
        dim=-1,
    ).repeat(batch_size, 1, 1)
    driver = (4.0 * eeg_time + 2.0).view(1, -1, 1).repeat(batch_size, 1, 1)
    mask = torch.ones(batch_size, 200, dtype=torch.bool)
    mask[:, :100] = False
    return {
        "state_mean": state,
        "state_var": torch.full_like(state, 0.25),
        "neural_driver_eeg_rate": driver,
        "neural_driver_var_eeg_rate": torch.full_like(driver, 0.5),
        "teacher_valid_mask": mask,
    }


def test_teacher_pools_constant_and_ramp_targets():
    adapter = PhysicalStateTeacher()
    output = adapter(_teacher_batch())
    assert output.full_summary.shape == (2, 10, 15)
    assert output.eeg_target.shape == (2, 10, 6)
    assert output.fnirs_target.shape == (2, 10, 9)
    # Statistic-major full summary: slopes start at index 5.
    assert torch.allclose(output.full_summary[:, :, 5], torch.full((2, 10), 2.0), atol=1e-5)
    assert torch.allclose(output.full_summary[:, :, 6], torch.full((2, 10), -1.0), atol=1e-5)
    assert torch.allclose(output.eeg_target[:, :, 1], torch.full((2, 10), 4.0), atol=1e-5)


def test_teacher_contracts_mask_to_complete_patches():
    adapter = PhysicalStateTeacher()
    batch = _teacher_batch(batch_size=1)
    batch["teacher_valid_mask"][0, 125] = False
    output = adapter(batch)
    expected = torch.tensor([[False, False, False, False, False, True, False, True, True, True]])
    assert torch.equal(output.valid_mask, expected)


def test_teacher_outputs_are_detached_and_uncertainty_positive():
    adapter = PhysicalStateTeacher()
    batch = _teacher_batch(batch_size=1)
    batch["state_mean"].requires_grad_(True)
    output = adapter(batch)
    assert not output.full_summary.requires_grad
    assert not output.eeg_target.requires_grad
    assert torch.all(output.full_uncertainty > 0)
    assert torch.all(output.eeg_uncertainty > 0)
    assert torch.all(output.fnirs_uncertainty > 0)
