# Shared codebook structure report

> for code reference, see src/tokenizers/shared_labram_vqnsp.py and experiments/scripts/train_shared_tokenizer.py at main branch as of 2026-3-26

## Alignment method

- **Shared codebook**: EEG and fNIRS has independent encoder and decoder, but they share a same vector quantizer with a same codebook. EEG and fNIRS latent representations are concatenated before quantization, then split back into separate modalities after quantization. 

```python
z_joint = torch.cat([z_eeg, z_fnirs], dim=0)
z_q_joint, indices_joint, quant_info = self.quantize(z_joint)
```

- **Explicit Alignment Losses**: Two alignment losses are designed to encourage the shared codebook to learn a common representation space for both modalities:
    1. **Latent Alignment Loss**: Use AlignmentLoss to directly calculate the similarity between the EEG and fNIRS latent representations before quantization.
    2. **Assignment Alignment Loss**: Use the codebook assignment indices to calculate a loss that encourages similar inputs from both modalities to be assigned to the same or nearby codebook entries.

- Dynamic Lag Handling:
    - Lag Candidates: The model maintains a set of candidate lags (e.g., 0,1,2,3 time steps) to account for potential temporal misalignment between EEG and fNIRS.
    - Lagged time slice: fNIRS inputs are shifted by each candidate lag.
    - Alignment Selection: in `_compute_alignment_losses`, the model computes alignment losses for each candidate lag and selects the one with the lowest loss as the optimal alignment for that training step.

## Data flow visualization

```mermaid
graph TD
    %% 定义样式
    classDef eeg fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef fnirs fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#000;
    classDef shared fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef align fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000,stroke-dasharray: 5 5;
    classDef loss fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000,stroke-dasharray: 5 5;

    %% 1. 数据输入层
    subgraph Input["1. 原始输入层 (Input Data)"]
        E_in["EEG Raw Data\n[B, C, T]"]:::eeg
        F_in["fNIRS Raw Data\n[B, C, T]"]:::fnirs
    end

    %% 2. 数据预处理与目标计算
    subgraph Preprocessing ["2. 序列切块与频域目标 (Patching & FFT Targets)"]
        E_patch["EEG Patches"]:::eeg
        F_patch["fNIRS Patches"]:::fnirs
        E_tgt["EEG Target (Amp/Phase)"]:::loss
        F_tgt["fNIRS Target (Amp/Phase)"]:::loss

        E_in --> E_patch --> E_tgt
        F_in --> F_patch --> F_tgt
    end

    %% 3. 独立编码器层
    subgraph Encoders ["3. 独立编码阶段 (Independent Encoders)"]
        E_embed["EEG Patch Embed (含 FFT 特征)"]:::eeg
        E_enc["EEG Transformer Encoder"]:::eeg
        E_proj["EEG Encode Proj"]:::eeg
        Z_eeg["Z_eeg (连续潜在特征)"]:::eeg

        F_embed["fNIRS Patch Embed (含 FFT 特征)"]:::fnirs
        F_enc["fNIRS Transformer Encoder"]:::fnirs
        F_proj["fNIRS Encode Proj"]:::fnirs
        Z_fnirs["Z_fnirs (连续潜在特征)"]:::fnirs

        E_in --> E_embed --> E_enc --> E_proj --> Z_eeg
        F_in --> F_embed --> F_enc --> F_proj --> Z_fnirs
    end

    %% 4. 共享码本与量化层
    subgraph SharedVQ ["4. 共享量化空间 (Shared Quantization Space)"]
        Concat["拼接 (torch.cat)"]:::shared
        VQ["Shared NormEMA Vector Quantizer\n(共享码本)"]:::shared
        Split["分离 (torch.split)"]:::shared
        
        ZQ_eeg["Z_q_eeg (离散量化特征)"]:::eeg
        ZQ_fnirs["Z_q_fnirs (离散量化特征)"]:::fnirs
        Indices["Codebook Indices (码本索引)"]:::shared

        Z_eeg --> Concat
        Z_fnirs --> Concat
        Concat --> VQ
        VQ --> Split
        VQ -.-> Indices
        Split --> ZQ_eeg
        Split --> ZQ_fnirs
    end

    %% 5. 对齐计算模块（侧边分支）
    subgraph Alignment ["动态对齐模块 (Alignment & Loss Computation)"]
        E_log["EEG Logits\n(分配概率)"]:::align
        F_log["fNIRS Logits\n(分配概率)"]:::align
        LagSearch["寻找最佳动态延迟 (Dynamic Lag Selection)"]:::align
        AlignLoss["Alignment Losses\n(Latent + Assignment KL)"]:::loss

        Z_eeg -.-> E_log
        Z_fnirs -.-> F_log
        Z_eeg -.-> LagSearch
        Z_fnirs -.-> LagSearch
        E_log -.-> LagSearch
        F_log -.-> LagSearch
        LagSearch -.-> AlignLoss
    end

    %% 6. 独立解码器层
    subgraph Decoders ["5. 独立解码阶段 (Independent Decoders)"]
        E_dec_in["EEG Decode Input Proj"]:::eeg
        E_dec["EEG Transformer Decoder"]:::eeg
        E_heads["EEG Amp/Phase Heads"]:::eeg
        E_pred_AP["EEG Pred Amp/Phase"]:::eeg

        F_dec_in["fNIRS Decode Input Proj"]:::fnirs
        F_dec["fNIRS Transformer Decoder"]:::fnirs
        F_heads["fNIRS Amp/Phase Heads"]:::fnirs
        F_pred_AP["fNIRS Pred Amp/Phase"]:::fnirs

        ZQ_eeg --> E_dec_in --> E_dec --> E_heads --> E_pred_AP
        ZQ_fnirs --> F_dec_in --> F_dec --> F_heads --> F_pred_AP
    end

    %% 7. 时域重建与损失
    subgraph Output ["6. 时域重建与损失计算 (Reconstruction & Losses)"]
        E_rec_time["EEG Time Reconstruction\n(逆FFT)"]:::eeg
        F_rec_time["fNIRS Time Reconstruction\n(逆FFT)"]:::fnirs
        Loss_E["EEG Loss\n(Amp + Phase + Time)"]:::loss
        Loss_F["fNIRS Loss\n(Amp + Phase + Time)"]:::loss
        Loss_Total["Total Loss\n(+ VQ_loss + Alignment)"]:::loss

        E_pred_AP -->|Reconstruct Time| E_rec_time
        F_pred_AP -->|Reconstruct Time| F_rec_time

        E_tgt -.-> Loss_E
        E_pred_AP -.-> Loss_E
        E_in -.-> Loss_E
        E_rec_time -.-> Loss_E

        F_tgt -.-> Loss_F
        F_pred_AP -.-> Loss_F
        F_in -.-> Loss_F
        F_rec_time -.-> Loss_F

        Loss_E --> Loss_Total
        Loss_F --> Loss_Total
        AlignLoss --> Loss_Total
    end
```
