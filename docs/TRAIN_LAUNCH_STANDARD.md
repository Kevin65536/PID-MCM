# Training launch standard

_Active entrypoint and archived-workflow boundary as of 2026-07-02_

---

## 🎯 Active status

The repository keeps one active launcher:

```bash
bash experiments/scripts/launch_training_nohup.sh \
  --task physiology-semantic-tokenizer
```

The task name is reserved, but execution is intentionally blocked until the P1 cache/tensor contract, P2 quantizer tests, dry-run manifest, and active output-root assertions pass. The launcher must not dispatch a source/observation, generic tokenizer, coupling-suite, or old downstream task.

## 🗂️ Historical workflows

Pre-redesign launchers and entrypoints are frozen under [`experiments/scripts/archive/pre_physiology_semantic_20260701/`](../experiments/scripts/archive/pre_physiology_semantic_20260701/README.md). Their configurations live under [`experiments/configs/archive/pre_physiology_semantic_20260701/`](../experiments/configs/archive/pre_physiology_semantic_20260701/README.md).

Historical execution is an explicit compatibility action. It is not a supported way to create new-design evidence, and archived suite launchers must never be copied back into the active script root.

## 🛡️ Registration gate

The active task can be enabled only when all of the following hold:

- the target entrypoint imports no module from `src.compatibility`;
- its configs resolve only from `experiments/configs/physiology_semantic_tokenizer/`;
- every run path is below `experiments/runs/physiology_semantic_tokenizer/`;
- a dry run writes the manifest and tensor-shape contract without training;
- default pytest collection excludes archived compatibility tests while active contract tests pass.

_Last updated: 2026-07-02_
