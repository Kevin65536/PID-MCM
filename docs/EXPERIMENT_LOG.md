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
| 2026-03-14 | UMAP-02 | Comparative | UMAP with comprehensive visualization & re-evaluation | Done |
| 2026-03-13 | UMAP-01 | Comparative | UMAP multimodal pretrain + finetune evaluation | Done |

---

## UMAP-02: UMAP with Comprehensive Visualization & Re-evaluation (2026-03-14)

### Objective
Re-run UMAP experiments with comprehensive visualization suite to understand:
1. **Training dynamics**: Multi-objective loss decomposition, convergence behavior, generalization gap
2. **Cross-modal fusion**: How attention flows between EEG and fNIRS modalities, per-head specialization
3. **MoE architecture**: SeqFusion gate weight distribution (modality-specific vs. fusion FFN utilization)
4. **Representation quality**: t-SNE of CLS embeddings (by modality/class/subject), modality alignment
5. **Classification quality**: Confusion matrices, per-subject accuracy, per-class metrics

### Visualization Suite Created
New file: `comparative_methods/UMAP/umap_plots.py` — 1200+ lines covering:

| Category | Plots | Key Insight Target |
|----------|-------|--------------------|
| **Pretrain Diagnostics** | Loss decomposition (6-panel), loss correlation scatter, LR schedule | Multi-task balance, convergence speed |
| **Classification Analysis** | Confusion matrix, classification dashboard (9-panel), finetune curves | per-class bias, subject-level variance |
| **Cross-Modal Attention** | Full attention heatmaps (per layer), cross-modal attention summary, per-head cross-modal % | Whether model learns to fuse modalities |
| **Embedding Analysis** | t-SNE (by modality/class/subject), modality alignment (cosine sim distributions) | Representation structure, class separability |
| **Architecture Analysis** | SeqFusion gate weights, gradient flow by module group | MoE utilization, learning signal propagation |
| **Comparative** | Experiment comparison (accuracy/F1/subject variance), modality robustness | Missing modality performance |

### Phase 1: Pretraining (U3-PT-plots)
- **Config**: 200 epochs, lr=1e-4, batch_size=64 (same as UMAP-01)
- **Best val loss**: 6.082 @ epoch 197 (improved from 6.342 in UMAP-01, same config but different random init)
- **Loss breakdown** (final): CON=4.084, MAT=0.636, GEN=1.362

**Key Observations from Pretrain Plots:**
- **Loss decomposition**: CON loss dominates (~67% of total), MAT converges fastest (stable by epoch 30), GEN shows slowest but steady improvement
- **Attention heatmaps**: Cross-modal attention exists but is weak — most attention stays intra-modal
- **Per-head analysis**: Some heads specialize in cross-modal attention while others remain intra-modal

### Phase 2: Finetune Results

| Experiment | Modality | Pretrained? | Val Acc | Test Acc | Test F1 | Subj Mean±Std |
|-----------|----------|-------------|---------|----------|---------|---------------|
| U3-FT-multi-np-v2 | Multi | No | 51.7% | 50.8% | 0.389 | 50.8±1.4% |
| U3-FT-multi-pt-v2 | Multi | Yes | 52.7% | 47.5% | 0.413 | 47.5±6.1% |
| U3-FT-eeg-pt | EEG | Yes | 52.3% | 51.7% | 0.382 | 51.7±1.2% |
| U3-FT-fnirs-pt | fNIRS | Yes | 52.3% | 51.3% | 0.397 | 51.3±1.8% |

### Key Findings from Visualization Analysis

**1. Cross-Modal Attention (Attention Heatmaps & Summary)**
- The Q-Former does develop *some* cross-modal attention, but it remains weak relative to intra-modal attention
- Cross-modal attention fraction varies by layer (higher in deeper layers, as expected)
- Per-head specialization exists: typically 1-2 heads per layer serve as "bridge" heads between modalities

**2. SeqFusion Gate Weights (MoE Analysis)**
- The SeqFusion gates (modality-specific vs. fusion FFN routing) show a roughly balanced distribution
- Gate weights cluster around 0.5, indicating the model doesn't strongly differentiate between modality-specific and fusion processing paths
- This suggests the MoE-like architecture is underutilized — the model doesn't learn strong modality routing

**3. Embedding Structure (t-SNE & Alignment)**
- EEG and fNIRS embeddings form partially overlapping clusters in t-SNE space — some alignment achieved
- **No class separation visible**: Left MI and Right MI embeddings are completely mixed
- Strong **subject clustering** in embeddings — subject identity dominates over task-relevant features
- Cosine similarity between paired EEG-fNIRS is only marginally higher than random pairing

**4. Classification Analysis**
- Confusion matrices show heavy prediction bias toward one class (model tends to predict all samples as class 0)
- Per-subject accuracy has very low variance when pretrained, suggesting model learns a constant prediction
- Pretrained model shows *higher* subject variance (6.1% std) vs no-pretrain (1.4% std), without improving mean accuracy

### Conclusions
1. **Results confirm UMAP-01 finding**: UMAP does not overcome cross-subject generalization (~50% accuracy)
2. **New insight — subject identity dominance**: t-SNE reveals embeddings encode subject identity rather than MI class, explaining chance-level classification
3. **New insight — weak cross-modal fusion**: Despite the Q-Former architecture, cross-modal attention remains weak; the model defaults to intra-modal processing
4. **New insight — underutilized MoE**: SeqFusion gates stay near 0.5 (uniform routing), suggesting the modality-specific FFN branches don't develop specialization
5. **Pretraining slightly hurts**: Pretrained multimodal (47.5%) underperforms random init (50.8%), consistent with UMAP-01

### Run Artifacts
- Pretrain: `comparative_methods/UMAP/runs/U3-PT-plots/`
- Finetune: `runs/U3-FT-multi-np-v2/`, `runs/U3-FT-multi-pt-v2/`, `runs/U3-FT-eeg-pt/`, `runs/U3-FT-fnirs-pt/`
- Comparison: `runs/_comparison/`
- Plot generation script: `comparative_methods/UMAP/generate_plots.py`
- Visualization module: `comparative_methods/UMAP/umap_plots.py`

---

## UMAP-01: UMAP Comparative Baseline Evaluation (2026-03-13)

### Objective
Evaluate UMAP (Unified Multi-task Pretraining with Q-Former) as a comparative baseline for multimodal EEG+fNIRS classification on the cross-subject motor imagery task (2-class: Left vs Right).

### Setup
- **Dataset**: EEG+NIRS Single-Trial, 29 subjects, motor imagery task
- **Split**: Train S01-S20 / Val S21-S25 / Test S26-S29
- **Feature mode**: channel_avg (segment trial into 5 windows, average per channel)
- **EEG**: 30 channels, 2000 samples → (5, 30)
- **fNIRS**: 36 channels (HbO), 100 samples → (5, 36)
- **Model**: Q-Former with hidden_size=64, 4 heads, 4 layers (~405K-418K params)

### Bug Fix During Experiment
Discovered that `load_pretrain_weights()` only transferred **10/118** parameters due to a key prefix mismatch:
- Pretrain model: `UMAP.encoder.layer.*`
- Finetune model: `Qformer.encoder.layer.*`

Fixed to map `UMAP.*` → `Qformer.*`, achieving **115/118** parameter transfer (3 missing = classification head, expected).

### Phase 2: Pretraining (U1-PT-full)
- **Config**: 200 epochs, lr=1e-4, batch_size=64, 3 objectives (CON+MAT+GEN)
- **Best val loss**: 6.342 @ epoch 176
- **Loss breakdown** (final): CON=4.084, MAT=0.637, GEN=1.625

### Phase 3: Finetune Results

| Experiment | Modality | Pretrained? | Val Acc | Test Acc | Test F1 | Notes |
|-----------|----------|-------------|---------|----------|---------|-------|
| U2-FT-multi-np | Multi | No | 50.7% | 52.9% | 0.423 | Baseline |
| U2-FT-eeg-np | EEG | No | 55.3% | 52.5% | 0.498 | Best no-pretrain |
| U2-FT-fnirs-np | fNIRS | No | 50.3% | 50.0% | 0.333 | At chance |
| U2-FT-multi-pt-v2 | Multi | Yes (fixed) | 50.3% | 50.0% | 0.333 | No improvement |
| U2-FT-eeg-pt-v2 | EEG | Yes (fixed) | 50.0% | 50.0% | 0.333 | No improvement |
| U2-FT-fnirs-pt-v2 | fNIRS | Yes (fixed) | 51.3% | 53.3% | 0.506 | Slight above chance |

### Subject-wise Accuracy (Test Set)

| Experiment | S26 | S27 | S28 | S29 |
|-----------|-----|-----|-----|-----|
| Multi (no PT) | 60.0% | 50.0% | 48.3% | 53.3% |
| EEG (no PT) | 53.3% | 61.7% | 46.7% | 48.3% |
| fNIRS (no PT) | 50.0% | 50.0% | 50.0% | 50.0% |
| Multi (PT) | 50.0% | 50.0% | 50.0% | 50.0% |
| EEG (PT) | 50.0% | 50.0% | 50.0% | 50.0% |
| fNIRS (PT) | 48.3% | 65.0% | 51.7% | 48.3% |

### Analysis
1. **All results hover at chance level (~50%)** for this cross-subject MI classification task, consistent with our own previous experiments (see Archived Pre-Experiments).
2. **Pretraining did not help** — the pretrained models actually performed slightly worse than no-pretrain baselines. This is likely because:
   - The contrastive/matching/generation pretraining objectives do not directly address the cross-subject domain gap.
   - The learned representations may still encode subject-specific features.
3. **EEG alone slightly outperformed fNIRS alone** without pretrain, which is expected since EEG has richer temporal dynamics for motor imagery.
4. **fNIRS alone was at complete chance level** without pretrain, confirming that fNIRS has limited discriminative power for MI at this temporal resolution.
5. The high subject-level variance (S27 ranged from 50-65% across experiments) indicates instability rather than reliable generalization.

### Conclusion
UMAP's multimodal pretrain-then-finetune framework does **not** overcome the cross-subject generalization challenge on this dataset. This result:
- **Validates our observation** that the core challenge is subject-specific feature encoding, not the fusion architecture.
- **Provides a fair comparative baseline**: UMAP ≈ 50-53% test accuracy, at or near chance.
- **Supports the need** for our PID-based approach that explicitly addresses cross-modal information decomposition and subject-invariant representation learning.

### Run Artifacts
- Pretrain: `comparative_methods/UMAP/runs/U1-PT-full/`
- Finetune (no pretrain): `runs/U2-FT-multi-np/`, `runs/U2-FT-eeg-np/`, `runs/U2-FT-fnirs-np/`
- Finetune (pretrained, fixed): `runs/U2-FT-multi-pt-v2/`, `runs/U2-FT-eeg-pt-v2/`, `runs/U2-FT-fnirs-pt-v2/`

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
