# Copilot Agent Task Runbook

## Purpose

This document defines the recommended way to run long Copilot coding agent tasks for this repository.

The intended mode is:

1. Create a background Copilot coding agent task
2. Let Copilot create its own branch and pull request
3. Use durable workflows for long experiments when possible
4. Track progress through agent-task logs and workflow logs

## Prerequisites

Before starting a task, confirm the following:

1. GitHub CLI version is at least 2.80
2. Copilot coding agent is enabled for the repository
3. `.github/workflows/copilot-setup-steps.yml` is present on the default branch
4. If long experiments must run on your server, a self-hosted runner exists with a single stable label such as `neural-token-gpu`
5. If Copilot-triggered workflows should run without manual approval, repository settings have been updated accordingly

## Recommended Task Scope

For this repository, agent tasks should use [docs/NEXT_STAGE_ALIGNMENT_PLAN.md](docs/NEXT_STAGE_ALIGNMENT_PLAN.md) as the main source of truth.

The safest first automated task is:

1. Create a new branch from `main`
2. Implement warm-start support in shared training
3. Add alignment warmup scheduling
4. Add lag-aware validation or alignment scaffolding
5. Add or update configs under `experiments/configs/phase0plus/`
6. Trigger one durable experiment workflow
7. Analyze produced run artifacts
8. Iterate once based on observed results

## Launch Command Template

Use the GitHub CLI entry point:

```bash
gh agent-task create \
  --repo Kevin65536/PID-MCM \
  --base main \
  --follow \
  "Read docs/NEXT_STAGE_ALIGNMENT_PLAN.md and .github/copilot-instructions.md. Create a new branch from main. Implement the first stage of the alignment plan with minimal, additive changes: add warm-start support to shared training, add alignment warmup controls, add lag-aware validation outputs, create any new configs needed under experiments/configs/phase0plus, and use durable workflows instead of fragile foreground shells for long-running experiments. Start one experiment, monitor its logs and artifacts, analyze the result, then make one follow-up iteration if justified. Keep all changes reviewable in a pull request and report output directories, workflow runs, and residual risks clearly." 
```

## Tracking Commands

List recent sessions:

```bash
gh agent-task list --repo Kevin65536/PID-MCM
```

View one session:

```bash
gh agent-task view TASK_ID --repo Kevin65536/PID-MCM
```

Stream live logs:

```bash
gh agent-task view TASK_ID --repo Kevin65536/PID-MCM --log --follow
```

## Workflow Dispatch Example

If the agent needs a durable run path for a long experiment, it can use the included workflow:

```bash
gh workflow run alignment-experiment.yml \
  --repo Kevin65536/PID-MCM \
  -f runner_label=neural-token-gpu \
  -f script_path=experiments/scripts/train_shared_tokenizer.py \
  -f config_path=experiments/configs/phase0plus/shared_labram_vqnsp_eeg_fnirs_10s_2s.yaml \
  -f extra_args="" \
  -f artifact_path=experiments/runs
```

If no self-hosted runner label is ready yet, replace `neural-token-gpu` with `ubuntu-latest` for lightweight checks only.

## Practical Notes

1. The agent session itself is independent of your current SSH session once created.
2. Long-running model training should be delegated to workflows or other durable backends, not to a foreground shell.
3. Session metadata is appended to `logs/copilot/` through repository hooks.
4. The agent should be steered through the GitHub agents UI or `gh agent-task` log inspection if it drifts from the plan.