# Repository Instructions For Copilot Coding Agent

## Primary Context

- Use [docs/NEXT_STAGE_ALIGNMENT_PLAN.md](../docs/NEXT_STAGE_ALIGNMENT_PLAN.md) as the main planning document for EEG-fNIRS alignment work.
- Preserve the current single-modality baselines and treat them as references, not disposable experiments.
- Prefer additive changes over destructive rewrites. Add new configs and workflows instead of mutating historical experiment records.

## Repository-Specific Expectations

- Do not modify raw data files under `data/`.
- Do not delete or overwrite finished runs under `experiments/runs/`.
- Prefer using existing training and probe entry points under `experiments/scripts/`.
- When editing alignment training, start from `experiments/scripts/train_shared_tokenizer.py` and `src/tokenizers/shared_labram_vqnsp.py` unless the task explicitly requires a different route.
- When proposing or implementing new alignment experiments, add configs under `experiments/configs/phase0plus/` or a clearly named subdirectory.

## Self-hosted Runner Context

- Copilot should run on the self-hosted runner labeled `neural-token-gpu`.
- The canonical local repository root on that runner is `/home/uais5/hkw/neural_token`.
- The preferred Python interpreter on that runner is `/home/uais5/hkw/neural_token/.venv/bin/python` when present.
- Non-Git assets such as `data/`, `experiments/configs/`, `experiments/runs/`, and `experiments/probe_results/` are available locally on the runner and may be symlinked into the GitHub Actions workspace by setup scripts.
- When a workflow or shell command can use the local virtual environment or local asset directories, prefer that over rebuilding from scratch.

## Long-Running Work

- For long experiments, prefer GitHub Actions workflows or other durable execution paths over keeping an interactive shell open.
- If a workflow exists for experiment execution, use it instead of a fragile foreground shell session.
- Always record where results are written, including run directory, artifact name, and summary file paths.

## Validation

- Validate code changes with the smallest relevant command first.
- Prefer using existing probe scripts for alignment evaluation before creating new one-off analysis code.
- When summarizing results, distinguish clearly between reconstruction quality, codebook health, and cross-modal alignment quality.

## Reporting

- In pull request descriptions and session updates, report:
  - branch name
  - workflow runs started
  - experiment configs used
  - output directories
  - remaining risks or blockers