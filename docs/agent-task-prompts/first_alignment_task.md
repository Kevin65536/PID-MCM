Read docs/NEXT_STAGE_ALIGNMENT_PLAN.md, docs/AGENT_TASK_RUNBOOK.md, and .github/copilot-instructions.md before making changes.

Work in the repository Kevin65536/PID-MCM from base branch main. Create a new branch and keep all changes reviewable in a pull request.

Primary objective:
Implement the first execution stage of the EEG-fNIRS alignment plan with minimal, additive changes and then run one durable experiment on the self-hosted runner neural-token-gpu.

Required scope:
1. Add warm-start support to experiments/scripts/train_shared_tokenizer.py so EEG and fNIRS branches can optionally initialize from validated single-modality checkpoints.
2. Add alignment warmup scheduling so alignment losses can start weak and ramp up instead of dominating from epoch 1.
3. Add lag-aware validation outputs or alignment scaffolding grounded in the current finding that delayed coupling is more meaningful than strict synchronous matching.
4. Add or update experiment configs under experiments/configs/phase0plus/ without overwriting historical baselines.
5. Prefer durable workflows over fragile foreground shells for long-running experiments.
6. Launch one experiment workflow using the local self-hosted runner and monitor it through completion or until there is enough evidence to determine the next change.
7. Analyze the produced outputs, summarize reconstruction quality, codebook health, and cross-modal alignment behavior separately, then make one follow-up iteration if the evidence clearly justifies it.

Repository constraints:
1. Do not modify raw data under data/.
2. Do not delete or overwrite finished runs under experiments/runs/.
3. Prefer additive configs and scripts over mutating historical experiment records.
4. Use /home/uais5/hkw/neural_token/.venv/bin/python when a Python interpreter is needed on the self-hosted runner.
5. Non-Git local assets are available on the self-hosted runner and may be bridged into the workspace by repository setup scripts.

Reporting requirements in the PR and session updates:
1. Branch name.
2. Workflow runs started.
3. Experiment configs used.
4. Output directories.
5. Remaining risks or blockers.