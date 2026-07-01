# Training Launch Standard

> **Status (2026-07-01):** The launcher remains the supported process entrypoint, but the physiology-semantic training task is not implemented yet. The source/observation and downstream tasks below are compatibility workflows for archived lineages and must not be treated as target-architecture commands.

This repository keeps exactly one supported training launcher:

```bash
bash experiments/scripts/launch_training_nohup.sh --task TASK [task args]
```

The launcher is responsible for selecting the training entrypoint and, by default, detaching it with `nohup`.
Use `--foreground` only for short interactive debugging runs.

Direct execution of `train_*.py` entrypoints is intentionally rejected; train tasks must enter through the launcher.

## 🎯 Target task status

`physiology-semantic-tokenizer` is reserved but not registered. It must not be added to the launcher until the P1 tensor/cache contract, P2 quantizer tests, dry-run manifest, and active output-root assertions pass. Its future outputs belong below `experiments/runs/physiology_semantic_tokenizer/<suite>/`.

## 🧰 Supported tasks

### source-observation-tokenizer — compatibility only

Script: `experiments/scripts/train_source_observation_tokenizer.py`

Supported task-specific arguments:

- `--config PATH`
- `--resume PATH`
- `--run-name NAME`
- `--skip-post-analysis`

Default post-analysis writes the gate scorecard plus codebook usage, reconstruction, and token pattern visualizations under the run `analysis/` directory.
Existing Croce local configs retain their historical `experiment.run_group`. Their completed outputs were moved to `experiments/archive/pre_physiology_semantic_20260701/runs/source_observation/`. Do not launch them as new-design experiments.

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

This is the compatibility downstream entry point for the Croce source/observation tokenizer line. Its completed outputs are archived under `experiments/archive/pre_physiology_semantic_20260701/runs/downstream/`.

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

## 📊 Analysis script placement

- Standardized post-training analysis implementations live in `src/visualization`.
- Standardized manual reruns use `experiments/scripts/analyze_alignment.py`.
- `experiments/scripts/probe/` is reserved for exploratory, non-standardized probe experiments that are not part of the default training workflow.
