# Training Launch Standard

This repository keeps exactly one supported training launcher:

```bash
bash experiments/scripts/launch_training_nohup.sh --task TASK [task args]
```

The launcher is responsible for selecting the training entrypoint and, by default, detaching it with `nohup`.
Use `--foreground` only for short interactive debugging runs.

Direct execution of `train_*.py` entrypoints is intentionally rejected; train tasks must enter through the launcher.

## Supported Tasks

### source-observation-tokenizer

Script: `experiments/scripts/train_source_observation_tokenizer.py`

Supported task-specific arguments:

- `--config PATH`
- `--resume PATH`
- `--run-name NAME`
- `--skip-post-analysis`

Default post-analysis writes the gate scorecard plus codebook usage, reconstruction, and token pattern visualizations under the run `analysis/` directory.
Croce local highWL configs set `experiment.run_group`, so their outputs are written under `experiments/runs/source_observation/croce_local/highwl_v1/<run_name>/`.

Example:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task source-observation-tokenizer \
  --config source_observation/croce_local/highwl_base.yaml \
  --run-name s2_croce_local_highwl_base
```

### tokenizer

Script: `experiments/scripts/train_tokenizer.py`

Supported task-specific arguments:

- `--config PATH`
- `--resume PATH`

Example:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task tokenizer \
  --config phase0plus/eeg_labram_vqnsp.yaml
```

### source-observation-token-export

Script: `experiments/scripts/export_source_observation_tokens.py`

This is the current first-stage downstream entry point for the Croce source/observation tokenizer line. It freezes a completed tokenizer checkpoint and exports 2s token sequences under `experiments/runs/downstream/source_observation_tokens/`.

Supported task-specific arguments:

- `--config PATH`
- `--tokenizer-run-dir PATH`
- `--checkpoint PATH`
- `--run-name NAME`
- `--splits train,val,test`
- `--max-batches N`
- `--max-samples N`

Example:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task source-observation-token-export \
  --foreground \
  --config downstream/source_observation_token_export_coupling0.yaml \
  --max-batches 1
```

The previous raw-signal downstream classifier script is archived at `experiments/scripts/archive/downstream_legacy_20260611/train_downstream_legacy_raw_signal.py`; its old phase1a configs are archived under `experiments/configs/archive/phase1a_legacy_downstream_20260611/`.

### foundation-interface

Script: `experiments/scripts/train_foundation_interface.py`

Supported task-specific arguments:

- `--config PATH`

Example:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task foundation-interface \
  --foreground \
  --config experiments/configs/phase1a/foundation_multimodal_interface.yaml
```

## Analysis Script Placement

- Standardized post-training analysis implementations live in `src/visualization`.
- Standardized manual reruns use `experiments/scripts/analyze_alignment.py`.
- `experiments/scripts/probe/` is reserved for exploratory, non-standardized probe experiments that are not part of the default training workflow.
