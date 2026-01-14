# PID-MCM Theory: Explicit Latent Partitioning Framework

> **Version**: 2.0  
> **Last Updated**: 2025-12-22  
> **Related**: See `IMPLEMENTATION_PLAN.md` for experiment details

---

## 1. Overview

This document describes the theoretical framework for **PID-guided Multimodal Pretraining** using **Explicit Latent Partitioning (ELP)**. The core idea is to learn separate latent representations for different information types defined by Partial Information Decomposition (PID).

**Key Innovation**: Instead of computing PID values (scalars), we extract PID-aligned **latent vectors** that can be used for downstream tasks.

---

## 2. PID Fundamentals

### 2.1 The Decomposition

For two source variables $X_1, X_2$ and target $Y$, PID decomposes total mutual information:

$$I(X_1, X_2; Y) = R + U_1 + U_2 + S$$

| Component | Symbol | Definition |
|:----------|:-------|:-----------|
| Redundancy | $R$ | Information both $X_1$ and $X_2$ provide about $Y$ |
| Unique 1 | $U_1$ | Information only $X_1$ provides |
| Unique 2 | $U_2$ | Information only $X_2$ provides |
| Synergy | $S$ | Information requiring **both** $X_1$ and $X_2$ together |

### 2.2 From Scalars to Vectors

**Classical PID**: $\text{Data} \xrightarrow{\text{estimation}} \{R, U_1, U_2, S\} \in \mathbb{R}^4$

**Our Approach**: $\text{Data} \xrightarrow{\text{neural network}} \{z_r, z_{u_1}, z_{u_2}, z_s\} \in \mathbb{R}^{d \times 4}$

We use geometric constraints as **proxies** for information-theoretic quantities.

---

## 3. ELP Architecture

### 3.1 Query Token Design

The encoder produces a **set** of vectors, not a single representation:

$$Z = \{ z_r, z_{u\_eeg}, z_{u\_fnirs}, z_s \}$$

Each token is a learnable parameter appended to input (like DETR or Perceiver).

### 3.2 Geometric Proxies

| PID Component | Information Definition | Geometric Proxy |
|:--------------|:----------------------|:----------------|
| Redundancy $R$ | $I(S_1; T) \cap I(S_2; T)$ | $z_r^{EEG} \approx z_r^{fNIRS}$ (alignment) |
| Unique $U$ | $I(S_1; T \| S_2)$ | $z_u \perp z_r$ (orthogonality) |
| Synergy $S$ | Joint info - individual sum | $z_s$ encodes residual (see §4.3) |

### 3.3 Architecture Diagram

```
Input: [EEG tokens] + [fNIRS tokens] + [Query tokens: Zr, Zu_e, Zu_f, Zs]
                              ↓
                    Transformer Encoder
                              ↓
              Latent Set: {z_r, z_u_eeg, z_u_fnirs, z_s}
                              ↓
                   Reconstruction Decoder
```

---

## 4. Loss Functions

### 4.1 Alignment Loss (for $z_r$)

Forces redundancy token to capture only shared information.

| Variant | Formula | Notes |
|:--------|:--------|:------|
| **A1: MSE (Baseline)** | $\|z_r^A - z_r^B\|^2$ | Simple, risk of collapse |
| A2: InfoNCE | $-\log \frac{\exp(\text{sim}(z_r^A, z_r^B)/\tau)}{\sum_k \exp(\text{sim}(z_r^A, z_r^{(k)})/\tau)}$ | Prevents collapse |
| A3: MSE + Variance | $\|z_r^A - z_r^B\|^2 + \max(0, \gamma - \text{Std}(z_r))$ | VICReg-style regularization |

### 4.2 Orthogonality Loss (for disjointness)

Ensures tokens encode different information.

| Variant | Formula | Notes |
|:--------|:--------|:------|
| **B1: Cosine (Baseline)** | $\sum_{i<j} |\cos(z_i, z_j)|$ | 6 pairs |
| B2: Squared Cosine | $\sum_{i<j} \cos^2(z_i, z_j)$ | Softer penalty |
| B3: Covariance | Off-diagonal $\text{Cov}(Z)$ penalty | Batch-level |

### 4.3 Synergy Loss (for $z_s$)

**Critical**: This is the most challenging component. We propose multiple approaches:

| Variant | Concept | Formula |
|:--------|:--------|:--------|
| **C1: Masking Diff (Baseline)** | $z_s$ changes when modality missing | $-\|z_s^{joint} - z_s^{masked}\|^2$ |
| **C4: Residual Reconstruction** | $z_s$ encodes what R+U cannot explain | $\|\text{Dec}_s(z_s) - \text{sg}(X - \text{Dec}(z_r + z_u))\|^2$ |
| **C5: Unpredictability** | $z_s$ cannot be predicted from single modality | $-\|z_s - \text{Pred}_{X_1}(z_r, z_{u_1})\|^2$ (adversarial) |

#### C4 Detail: Residual Reconstruction Synergy

```python
# Step 1: Base reconstruction (without synergy)
recon_base = Decoder(z_r + z_u_eeg + z_u_fnirs)
residual = X - recon_base  # What R+U cannot explain

# Step 2: Synergy reconstructs residual
recon_syn = Decoder_syn(z_s)
L_syn = MSE(recon_syn, stop_gradient(residual))
```

**Why it works**: By construction, $z_s$ learns **only** the leftover information.

#### C5 Detail: Cross-Modal Unpredictability

```python
# Adversarial predictor tries to predict z_s from single modality
z_s_pred_eeg = Predictor_eeg(z_r, z_u_eeg)
z_s_pred_fnirs = Predictor_fnirs(z_r, z_u_fnirs)

# Main model maximizes prediction error
L_syn = -||z_s - z_s_pred_eeg||^2 - ||z_s - z_s_pred_fnirs||^2
```

**Why it works**: Ensures $z_s$ contains info that single modalities cannot provide.

### 4.4 Reconstruction Loss

$$\mathcal{L}_{rec} = \|X - \text{Dec}(z_r + z_u + z_s)\|^2$$

### 4.5 Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{orth} + \lambda_3 \mathcal{L}_{syn}$$

---

## 5. Training Strategy

### 5.1 Mixed-Batch Training

| Proportion | Masking Pattern | Active Constraints |
|:-----------|:----------------|:-------------------|
| 25% | Cross-Modal (mask 80% EEG, keep fNIRS) | $\mathcal{L}_{align}$ |
| 25% | Uni-Modal (mask 50% EEG, drop fNIRS) | $\mathcal{L}_{orth}$ |
| 50% | Joint (mask 50% both) | $\mathcal{L}_{syn}$ |

### 5.2 Stop-Gradient Options

| Option | Description |
|:-------|:------------|
| **D1: None (Baseline)** | All tokens receive all gradients |
| D2: Phase-Specific | stop_grad($z_r$) during unique learning |
| D3: Full Routing | Each loss only updates target token |

---

## 6. Theoretical Limitations

### 6.1 Known Assumptions

1. **Linear Gaussian Approximation**: Vector PID correspondence is rigorous only under Gaussian assumption
2. **No Explicit Target $Y$**: In pretraining, we use reconstruction as proxy for task-relevant information
3. **Geometric ≠ Information-Theoretic**: Orthogonality is necessary but not sufficient for independence

### 6.2 Mitigation Strategies

- Use HSIC (not just cosine) for independence verification
- Validate with synthetic data where ground truth is known
- Design domain-specific constraints (e.g., frequency priors for EEG-fNIRS)

---

## 7. References

1. Williams & Beer (2010). Nonnegative decomposition of multivariate information.
2. Liang et al. (2023). FactorCL: Factorized Contrastive Learning.
3. Dufumier et al. (2024). CoMM: What to Align in Multimodal Contrastive Learning?
4. Xin et al. (2025). I²MoE: Interpretable Multimodal Interaction-aware Mixture of Experts.

---

*See `references/` directory for detailed analysis of related methods.*
