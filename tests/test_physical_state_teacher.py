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


def test_teacher_separates_local_cache_validity_from_causal_context_validity():
    adapter = PhysicalStateTeacher()
    batch = _teacher_batch(batch_size=1)
    batch["cache_valid_mask"] = torch.ones(1, 200, dtype=torch.bool)
    batch["causal_valid_mask"] = batch["teacher_valid_mask"].clone()
    output = adapter(batch)

    assert torch.all(output.valid_mask)
    assert torch.equal(
        output.context_valid_mask,
        torch.tensor([[False, False, False, False, False, True, True, True, True, True]]),
    )


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


def test_teacher_keeps_entry_masks_independent_and_versioned():
    adapter = PhysicalStateTeacher(
        target_family="physiology_shaped_multimodal_consensus_proxy",
        target_version="adaptive_teacher_v3",
    )
    batch = _teacher_batch(batch_size=1)
    batch["cache_valid_mask"] = torch.ones(1, 200, dtype=torch.bool)
    batch["eeg_prototype_valid_mask"] = torch.ones(1, 200, dtype=torch.bool)
    batch["eeg_prototype_valid_mask"][0, 40] = False
    batch["fnirs_coupling_valid_mask"] = torch.ones(1, 200, dtype=torch.bool)
    batch["fnirs_coupling_valid_mask"][0, :20] = False

    output = adapter(batch)

    assert output.target_family == "physiology_shaped_multimodal_consensus_proxy"
    assert output.target_version == "adaptive_teacher_v3"
    assert output.entry_masks["eeg"]["local"].all()
    assert not output.entry_masks["eeg"]["prototype"][0, 2]
    assert not output.entry_masks["fnirs"]["coupling"][0, 0]
    assert output.entry_masks["fnirs"]["local"].all()


def test_teacher_accepts_direct_patch_targets_and_preserves_entry_masks():
    batch = {
        "eeg_target": torch.randn(2, 10, 6, requires_grad=True),
        "eeg_uncertainty": torch.ones(2, 10, 6),
        "fnirs_target": torch.randn(2, 10, 9),
        "fnirs_uncertainty": torch.ones(2, 10, 9),
    }
    for modality in ("eeg", "fnirs"):
        for entry in ("local", "prototype", "context", "coupling"):
            batch[f"{modality}_{entry}_valid_mask"] = torch.zeros(2, 10, dtype=torch.bool)
    batch["eeg_local_valid_mask"][:, 2:] = True
    batch["fnirs_local_valid_mask"][:, 3:] = True
    batch["eeg_prototype_valid_mask"][:, 1:] = True

    output = PhysicalStateTeacher(
        target_family="adaptive_multimodal_consensus_proxy",
        target_version="adaptive_ssm_gauge_corrected_patch_v1",
    )(batch)

    assert output.eeg_target.shape == (2, 10, 6)
    assert output.fnirs_target.shape == (2, 10, 9)
    assert not output.eeg_target.requires_grad
    assert torch.equal(output.valid_mask, batch["eeg_local_valid_mask"] & batch["fnirs_local_valid_mask"])
    assert torch.equal(output.entry_masks["eeg"]["prototype"], batch["eeg_prototype_valid_mask"])
