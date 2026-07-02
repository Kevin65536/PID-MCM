# Pre-physiology-semantic compatibility package

_Frozen source/observation code boundary, 2026-07-02_

---

## 🗂️ Scope

This package preserves the model and analysis contract used by experiments completed before the physiology-semantic redesign:

- `source_observation_tokenizer.py`: four-codebook source/observation tokenizer and embedded coupling mechanisms;
- `cross_modal_fusion.py`: tokenizer-stage EEG/fNIRS exchange;
- `multimodal_tokenizer_losses.py`: coupling-shaping and source/observation losses;
- `visualization/`: scorecards and tensor projections tied to that output schema;
- `classifiers/`: retired raw-signal and tokenizer-classifier downstream models;
- `elp_encoder.py`: unused early ELP prototype retained for provenance.

## 🛡️ Import contract

Active code must not import this package. Historical code registers checkpoint-compatible model names explicitly:

```python
from src.compatibility.pre_physiology_semantic_20260701 import register_legacy_tokenizers

register_legacy_tokenizers()
```

Importing `src.tokenizers` alone does not expose or register the archived model.

_Last updated: 2026-07-02_
