# Legacy Raw-Signal Downstream Pipeline

This directory preserves the previous downstream classifier entry point:

- `train_downstream_legacy_raw_signal.py`

It was archived on 2026-06-11 because it trains classifiers by loading raw physiological windows and applying older single-modality tokenizers online. That contract does not match the current Croce source/observation tokenizer workflow, where downstream and foundation-model experiments should consume exported 2s token sequences.

Its classifier modules now live under `src/classifiers_legacy/`.

Use `experiments/scripts/export_source_observation_tokens.py` for the current first-stage downstream tokenization workflow.
