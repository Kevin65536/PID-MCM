# UMAP Experiment Design — Comparative Evaluation

## Objective

Evaluate UMAP as a comparative baseline for our PID-MCM multimodal framework.
Key questions:

1. **Does UMAP's multimodal pretraining improve downstream classification?**
2. **How well does UMAP handle missing modalities (EEG-only, fNIRS-only)?**
3. **How does UMAP compare to our PID-based approach?**

## Experiment Matrix

### Phase 1: Functional Validation (Smoke Test)

| ID | Experiment | Config | Purpose |
|----|-----------|--------|---------|
| U0-smoke | Pretrain (3 epochs) | `pretrain.yaml --epochs 3` | Verify pipeline runs end-to-end |

```bash
python train_umap.py pretrain --config configs/pretrain.yaml \
    --epochs 3 --batch_size 32 --run_name U0-smoke
```

### Phase 2: Pretraining

| ID | Experiment | Epochs | Ablation | Purpose |
|----|-----------|--------|----------|---------|
| U1-PT-full | Full pretrain (CON+MAT+GEN) | 200 | all | Baseline pretraining |
| U1-PT-con | CON only | 200 | con | Ablation: contrastive only |
| U1-PT-con-mat | CON+MAT | 200 | con,mat | Ablation: no generation |
| U1-PT-con-gen | CON+GEN | 200 | con,gen | Ablation: no matching |

```bash
# Full pretrain
python train_umap.py pretrain --config configs/pretrain.yaml --run_name U1-PT-full

# Ablation: contrastive only
python train_umap.py pretrain --config configs/pretrain.yaml --run_name U1-PT-con
# (set ablation_tasks: "con" in config or add CLI support)
```

### Phase 3: Finetuning — Classification

| ID | Pretrained? | Modality | n_class | Purpose |
|----|-------------|----------|---------|---------|
| U2-FT-multi-pt | U1-PT-full | multi | 2 | Multimodal with pretrain |
| U2-FT-multi-np | No | multi | 2 | Multimodal without pretrain |
| U2-FT-eeg-pt | U1-PT-full | eeg | 2 | EEG-only with pretrain |
| U2-FT-eeg-np | No | eeg | 2 | EEG-only without pretrain |
| U2-FT-fnirs-pt | U1-PT-full | eye | 2 | fNIRS-only with pretrain |
| U2-FT-fnirs-np | No | eye | 2 | fNIRS-only without pretrain |

```bash
# Multimodal finetuning with pretrained weights
python train_umap.py finetune --config configs/finetune.yaml \
    --pretrain_ckpt runs/U1-PT-full/checkpoints/best_checkpoint.pth \
    --modality multi --run_name U2-FT-multi-pt

# EEG-only finetuning, no pretrain
python train_umap.py finetune --config configs/finetune.yaml \
    --modality eeg --run_name U2-FT-eeg-np

# fNIRS-only with pretrain
python train_umap.py finetune --config configs/finetune.yaml \
    --pretrain_ckpt runs/U1-PT-full/checkpoints/best_checkpoint.pth \
    --modality eye --run_name U2-FT-fnirs-pt
```

### Phase 4: Feature Mode Comparison

| ID | Feature Mode | Pretrained? | Modality | Purpose |
|----|-------------|-------------|----------|---------|
| U3-BP-PT | band_power | Yes | multi | Band power features + pretrain |
| U3-BP-NP | band_power | No | multi | Band power features, no pretrain |
| U3-CA-PT | channel_avg | Yes | multi | Channel avg + pretrain (= U2-FT-multi-pt) |

```bash
python train_umap.py pretrain --config configs/pretrain.yaml \
    --feature_mode band_power --run_name U3-BP-PT-pretrain
python train_umap.py finetune --config configs/finetune.yaml \
    --feature_mode band_power \
    --pretrain_ckpt runs/U3-BP-PT-pretrain/checkpoints/best_checkpoint.pth \
    --run_name U3-BP-PT
```

## Metrics

| Metric | Phase | Description |
|--------|-------|-------------|
| Total pretrain loss | Phase 2 | CON + MAT + GEN loss convergence |
| Per-task loss | Phase 2 | Individual loss component tracking |
| Test accuracy | Phase 3 | Classification accuracy (chance = 50%) |
| F1 macro | Phase 3 | Balanced metric for potential class imbalance |
| Subject-wise accuracy | Phase 3 | Generalization across subjects |
| Pretrain vs No-pretrain Δ | Phase 3 | Improvement from pretraining |
| Missing modality Δ | Phase 3 | Degradation from missing modality |

## Expected Results & Baselines

| Method | Expected Accuracy | Notes |
|--------|-------------------|-------|
| Chance | 50% | 2-class MI |
| EEG-only (no pretrain) | 55-65% | Cross-subject MI is challenging |
| fNIRS-only (no pretrain) | 50-55% | fNIRS alone is weak for MI |
| Multi (no pretrain) | 60-70% | Fusion should help |
| Multi (pretrained) | 65-75% | UMAP's pretraining advantage |
| EEG-only (pretrained) | 60-70% | Missing modality with pretrain |
| fNIRS-only (pretrained) | 52-60% | Exploits cross-modal knowledge |

## Comparison with Our Framework

Results from UMAP experiments will be compared against:
- **Phase 0 / 0+**: Our tokenizer-based representations
- **Phase 1A**: Our classification baselines
- **PID-MCM (full)**: Our proposed framework

Key comparison axes:
1. **Classification accuracy** (same data split, same task)
2. **Missing modality robustness** (how much degrades when one modality is absent)
3. **Parameter efficiency** (UMAP params vs. PID-MCM params)
4. **Representation quality** (probe experiments on learned embeddings)
