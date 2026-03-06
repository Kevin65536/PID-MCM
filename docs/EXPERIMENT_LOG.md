# Experiment Log

> 实验记录文档，按时间倒序记录每次实验的配置、结果和结论。

## ⚠️ Lessons Learned from Pre-Experiments

**Archived Results:** Previous experiment runs and logs have been archived to `docs/ARCHIVED_PRE_EXPERIMENTS.md` and `experiments/runs/archive/pre_experiments`.

**Key Bottleneck:**
The downstream Motor Imagery (MI) classification task suffered from a severe lack of cross-subject generalization (hovering around ~50% accuracy, essentially chance level for binary classification), despite performing reasonably well on within-subject tests. 

**Root Causes & Observations:**
1. Tokenizers (e.g., VQ-VAE, FSQ, LaBraM VQNSP) tend to encode subject-specific identity features rather than generalized semantic MI features.
2. The extreme inter-subject variability in EEG/fNIRS signals makes standard training overfit to the training subjects.

**Strategies for Future First Stage Experiments:**
- Explore advanced domain adaptation or alignment techniques to remove subject-specific features.
- Consider utilizing larger, more diverse datasets.
- Implement stronger data augmentation strategies specifically aimed at cross-subject invariance.
- Re-evaluate the tokenizer training objective to encourage learning generalized representations instead of perfect reconstruction, which may be forcing the model to remember subject identity.

---

## Experiment Index

| Date | ID | Phase | Description | Status |
|------|----|-------|-------------|--------|
| - | - | - | - | - |

---

## [Template] EXP-XXX: [Title] (YYYY-MM-DD)

### Objective
[What is the goal of this experiment?]

### Configuration
[Key differences from baseline, file paths to config, model parameters, etc.]

### Results
[Tables, metrics, confusion matrices, etc.]

### Analysis
[Why did we get these results? Deep dive into the data.]

### Conclusion
[Final takeaway and next steps.]
