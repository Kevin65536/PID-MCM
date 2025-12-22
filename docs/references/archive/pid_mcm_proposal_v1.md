# PID-MCM: Latent Partitioning for Disjoint Feature Extraction

## 1. The Core Problem: Disjoint Information Atoms

According to PID theory, the total mutual information decomposes as:
$$I(S_1, S_2; T) = R + U_1 + U_2 + S$$

where these components are **mutually exclusive** (non-overlapping):
$$R \cap U_{EEG} = \emptyset, \quad R \cap S = \emptyset, \quad U_{EEG} \cap S = \emptyset$$

In EEG-fNIRS context:
*   $R$ (Redundancy): Low-frequency envelopes, hemodynamic correlates that both modalities capture.
*   $U_{EEG}$ (Unique): High-frequency phase dynamics, millisecond-level timing (fNIRS is too slow).
*   $U_{fNIRS}$ (Unique): Slow metabolic baseline shifts, deep oxygenation changes.
*   $S$ (Synergy): Neurovascular coupling delays/phase-shifts — information that only emerges from the *relationship* between modalities.

### Why Single-Vector Encoding Fails

If we use a single unstructured latent vector $Z$ for all masking tasks, the encoder suffers from **Semantic Shift**:
*   In Cross-Modal task: $Z$ must represent $R$
*   In Uni-Modal task: $Z$ must represent $R + U$
*   In Joint task: $Z$ must represent $R + U + S$

This forces the weights to either:
1. **Catastrophic Forgetting**: Later tasks overwrite earlier learned representations
2. **Semantic Blurring**: All components get mixed, losing interpretability

---

## 2. The Solution: Explicit Latent Partitioning (ELP)

Instead of a single `[CLS]` token, we introduce **Dedicated Query Tokens** that partition the latent space. The encoder outputs a **set** of vectors, not just one.

**Latent Structure:**
$$ Z = \{ z_r, z_{u\_eeg}, z_{u\_fnirs}, z_s \} $$

*   $z_r$: **Redundancy Token**. Captures shared information.
*   $z_{u\_eeg}$: **EEG Unique Token**. Captures EEG-specific residuals.
*   $z_{u\_fnirs}$: **fNIRS Unique Token**. Captures fNIRS-specific residuals.
*   $z_s$: **Synergy Token**. Captures interaction effects.

These tokens are learnable parameters (like in DETR or Perceiver) appended to the input sequence.

### Theoretical Justification: Geometric Proxy for Information Theory

The key insight is that **geometric constraints in latent space can serve as proxies for information-theoretic quantities**:

| PID Component | Information-Theoretic Definition | Geometric Proxy |
|:-------------|:--------------------------------|:---------------|
| Redundancy $R$ | $I(S_1; T) \cap I(S_2; T)$ | $z_r^{EEG} \approx z_r^{fNIRS}$ (Alignment) |
| Unique $U$ | $I(S_1; T \| S_2)$ | $z_u \perp z_r$ (Orthogonality as Residual) |
| Synergy $S$ | $I(S_1, S_2; T) - I(S_1; T) - I(S_2; T) + R$ | $z_s$ changes when any modality is masked |

This avoids the computational difficulty of directly estimating mutual information.

---

## 3. Training Strategy: "Routing" via Masking

We still use the "Mixture of Masking" strategy, but now we **route** the gradients to specific tokens. This ensures that $z_r$ *only* learns redundancy, and $z_u$ *only* learns the residual.

### **Phase 1: Redundancy Learning (The "Alignment" Constraint)**
*   **Input**:
    *   View A: `{Masked EEG, Full fNIRS}`
    *   View B: `{Full EEG, Masked fNIRS}`
*   **Operation**:
    *   The encoder produces $z_r^A$ (from fNIRS context) and $z_r^B$ (from EEG context).
*   **Constraint**: **Latent Alignment**
    *   $\mathcal{L}_{align} = || z_r^A - z_r^B ||^2$
    *   *Why?* This forces $z_r$ to capture **only** the information present in *both* modalities (the intersection). If $z_r$ tried to capture high-freq EEG ($U$), it would fail to match $z_r$ from fNIRS.

**Information-Theoretic Interpretation:**
$$z_r \approx \arg\max_z \left[ I(z; S_{EEG}) \cap I(z; S_{fNIRS}) \right]$$

The alignment constraint ensures $z_r$ only encodes information accessible from **both** modalities.

### **Phase 2: Unique Learning (The "Residual" Constraint)**
*   **Input**: `{Masked EEG, Zero fNIRS}` (Uni-modal context).
*   **Operation**:
    *   Encoder produces $z_r$ and $z_{u\_eeg}$.
*   **Prediction**:
    *   Reconstruct EEG using **Sum/Concat** of tokens: $\text{Dec}(z_r + z_{u\_eeg})$.
*   **Constraint**:
    *   Since $z_r$ is already constrained (by Phase 1) to be "low-freq/shared", $z_{u\_eeg}$ is forced to capture the **residual** (high-freq details) to minimize reconstruction error.
    *   **Orthogonality Constraint** (Recommended): $\mathcal{L}_{orth} = |\text{CosSim}(z_r, z_{u\_eeg})|$

**Information-Theoretic Interpretation:**
$$z_{u\_eeg} = \text{Residual}(S_{EEG}, z_r) \approx I(S_{EEG}; T | S_{fNIRS})$$

The reconstruction loss naturally pushes $z_u$ toward modality-specific information.

### **Phase 3: Synergy Learning (The "Integration" Constraint)**
*   **Input**: `{Random Mask EEG, Random Mask fNIRS}` (Joint context).
*   **Operation**:
    *   Encoder produces all tokens, including $z_s$.
*   **Prediction**:
    *   Reconstruct everything using $\text{Dec}(z_r + z_{u\_eeg} + z_{u\_fnirs} + z_s)$.
*   **Constraint**:
    *   $z_s$ captures whatever cannot be explained by the sum of parts ($R+U$). This is the "interaction" term.

**Synergy Verification Constraint** (Recommended):
$$\mathcal{L}_{syn} = -|| z_s^{joint} - z_s^{masked\_one} ||^2$$

This ensures $z_s$ **changes significantly** when one modality is missing — the signature of true synergy (information that requires both sources).

---

## 4. Architecture Diagram (Conceptual)

```mermaid
graph TD
    Input[Input: EEG + fNIRS + Query Tokens] --> Transformer[Transformer Encoder]
    Transformer --> Z_set[Latent Set: {Zr, Zu_e, Zu_f, Zs}]
    
    Z_set -->|Select Zr| Head_Align[Alignment Head]
    Head_Align --> Loss_Align(L_align: Zr_eeg ≈ Zr_fnirs)
    
    Z_set -->|Select Zr + Zu| Head_Rec[Reconstruction Head]
    Head_Rec --> Loss_Rec(L_rec: Predict Masked Signal)
    
    Z_set -->|All Tokens| Head_Orth[Orthogonality Head]
    Head_Orth --> Loss_Orth(L_orth: Zr ⊥ Zu ⊥ Zs)
    
    subgraph "Gradient Routing"
    Loss_Align -.->|Updates| Zr[Zr: Redundancy]
    Loss_Rec -.->|Updates| Zu[Zu: Unique]
    Loss_Orth -.->|Enforces| Disjoint[Disjointness]
    end
```

---

## 5. Training Configuration

### Joint Training (Recommended)
Instead of sequential phases, use **mixed-batch training**:

| Batch Proportion | Masking Pattern | Active Constraints |
|:---------------:|:----------------|:-------------------|
| 25% | Cross-Modal (mask 80% EEG, keep fNIRS) | $\mathcal{L}_{align}$ + $\mathcal{L}_{rec}$ |
| 25% | Uni-Modal (mask 50% EEG, drop fNIRS) | $\mathcal{L}_{rec}$ + $\mathcal{L}_{orth}$ |
| 50% | Joint (mask 50% both) | $\mathcal{L}_{rec}$ + $\mathcal{L}_{syn}$ |

**Total Loss:**
$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{orth} + \lambda_3 \mathcal{L}_{syn}$$

### Optional: Stop-Gradient for Stability
In Unique/Synergy learning, use `stop_gradient(z_r)` to prevent the reconstruction loss from modifying the already-constrained redundancy token.

---

## 6. Theoretical Comparison with Other Methods

| Method | Decomposition Location | Constraint Type | Disjointness Guarantee |
|:-------|:----------------------|:----------------|:----------------------|
| I²MoE | Architecture (4 Expert Networks) | Behavioral (Perturbation) | Weak (shared backbone) |
| Original MCM | Data (Masking Patterns) | Implicit (None) | No |
| **ELP (Ours)** | Latent Space (Query Tokens) | Geometric (Align + Orth) | **Strong** |

### Why ELP is Theoretically Superior

1. **Direct Structural Correspondence**: The latent partition $\{z_r, z_u, z_s\}$ directly mirrors PID's additive decomposition $I = R + U + S$.

2. **Orthogonality = Disjointness**: By enforcing $z_r \perp z_u \perp z_s$, we ensure:
   $$||Z_{total}||^2 = ||z_r||^2 + ||z_u||^2 + ||z_s||^2$$
   This is the Pythagorean decomposition — geometric analog of information additivity.

3. **Avoids Catastrophic Forgetting**: Different tokens store different information; learning $z_s$ doesn't overwrite $z_r$.

---

## 7. Summary of Advantages

1.  **Solves Disjointness**: $R$, $U$, $S$ are stored in separate vectors. No "overwriting" or "averaging".
2.  **Single Encoder Efficiency**: One Transformer backbone. Decomposition happens via **Attention** (different query tokens attend to different features).
3.  **Geometric PID Proxy**: No need to compute Mutual Information. Alignment/Orthogonality constraints serve as tractable proxies.
4.  **Interpretable Representations**: Each token has clear semantic meaning, enabling downstream analysis (e.g., "which tasks rely more on synergy?").

---

## 8. Potential Extensions

### Multi-Scale Unique Tokens
If EEG's unique information is multi-dimensional (different frequency bands):
$$z_{u\_eeg} = \{z_{u\_\delta}, z_{u\_\theta}, z_{u\_\alpha}, z_{u\_\beta}, z_{u\_\gamma}\}$$

### Downstream Task Routing
For classification, dynamically weight the contribution of each token:
$$\hat{y} = f(w_r z_r + w_u z_u + w_s z_s)$$

The weights $w$ reveal which PID component is most relevant for the task.
