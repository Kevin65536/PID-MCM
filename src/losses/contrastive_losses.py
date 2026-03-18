"""
Contrastive losses for cross-modal EEG-fNIRS token alignment.

These losses support the shared-codebook training paradigm where EEG and fNIRS
tokenizers are trained jointly with a single shared codebook, and contrastive
alignment explicitly pulls same-trial EEG and fNIRS representations together.

Key loss classes:
- InfoNCELoss: InfoNCE / NT-Xent loss for batch-level cross-modal alignment.
- TokenAlignmentLoss: Token-level mean-pooled cross-modal alignment.
- IndexMatchingLoss: Soft loss encouraging same codebook index assignment.
- SharedCodebookLoss: Combined loss for shared-codebook training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss for cross-modal alignment.

    Given a batch of N paired (EEG, fNIRS) samples, treats pair (i, i) as
    positive and all cross-modal pairs (i, j≠i) as negatives.

    Both directions are computed and averaged:
      L = 0.5 * (L_{EEG→fNIRS} + L_{fNIRS→EEG})

    Reference: "Representation Learning with Contrastive Predictive Coding"
               (Oord et al., 2018); SimCLR (Chen et al., 2020).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_fnirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute bidirectional InfoNCE loss.

        Args:
            z_eeg:   [B, D] L2-normalised EEG representations.
            z_fnirs: [B, D] L2-normalised fNIRS representations.

        Returns:
            loss:  scalar InfoNCE loss.
            info:  dict with per-direction losses and top-1 retrieval accuracy.
        """
        # L2-normalise so dot-product equals cosine similarity
        z_eeg = F.normalize(z_eeg, p=2, dim=-1)
        z_fnirs = F.normalize(z_fnirs, p=2, dim=-1)

        B = z_eeg.shape[0]
        labels = torch.arange(B, device=z_eeg.device)

        # Similarity matrices [B, B]
        logits_ef = (z_eeg @ z_fnirs.T) / self.temperature  # EEG → fNIRS
        logits_fe = (z_fnirs @ z_eeg.T) / self.temperature  # fNIRS → EEG

        loss_ef = F.cross_entropy(logits_ef, labels)
        loss_fe = F.cross_entropy(logits_fe, labels)
        loss = 0.5 * (loss_ef + loss_fe)

        # Top-1 retrieval accuracy (diagnostic)
        with torch.no_grad():
            acc_ef = (logits_ef.argmax(dim=1) == labels).float().mean()
            acc_fe = (logits_fe.argmax(dim=1) == labels).float().mean()
            acc = 0.5 * (acc_ef + acc_fe)

        return loss, {
            'loss_ef': loss_ef,
            'loss_fe': loss_fe,
            'retrieval_acc': acc,
        }


class TokenAlignmentLoss(nn.Module):
    """
    Token-level contrastive alignment loss.

    Computes InfoNCE over mean-pooled token representations.  This is suitable
    when EEG and fNIRS produce the same number of tokens per window (e.g. both
    produce 4 tokens for a 4 s window).

    If the modalities produce different token counts, mean-pooling collapses the
    sequence dimension to a single [B, D] vector before computing InfoNCE.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.infonce = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_fnirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z_eeg:   [B, N_eeg, D]  pre-quantisation EEG token embeddings.
            z_fnirs: [B, N_fnirs, D] pre-quantisation fNIRS token embeddings.

        Returns:
            loss, info dict.
        """
        # Mean-pool over token dimension -> [B, D]
        z_eeg_pooled = z_eeg.mean(dim=1)
        z_fnirs_pooled = z_fnirs.mean(dim=1)
        return self.infonce(z_eeg_pooled, z_fnirs_pooled)


class IndexMatchingLoss(nn.Module):
    """
    Soft index-matching loss that encourages EEG and fNIRS tokens to use the
    same codebook entries for the same trial.

    Because codebook indices are discrete and non-differentiable, this loss
    operates on the pre-quantisation similarities rather than on indices
    directly.  For each token position t, it computes a soft cross-entropy
    between the probability distribution the EEG encoder induces over codebook
    entries and the hard assignment produced by the fNIRS encoder (and vice
    versa).

    This provides a gradient signal that pushes each modality's continuous
    embeddings toward the codebook entry chosen by the other modality.

    Args:
        temperature: Softmax temperature for computing soft assignments.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_fnirs: torch.Tensor,
        codebook: torch.Tensor,
        indices_eeg: Optional[torch.Tensor] = None,
        indices_fnirs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z_eeg:      [B, N, D] continuous EEG encoder output (L2-normalised).
            z_fnirs:    [B, N, D] continuous fNIRS encoder output (L2-normalised).
            codebook:   [K, D] shared codebook weight (L2-normalised).
            indices_eeg:   Optional [B, N] hard EEG codebook assignments.
            indices_fnirs: Optional [B, N] hard fNIRS codebook assignments.

        Returns:
            loss, info dict.
        """
        B, N, D = z_eeg.shape
        K = codebook.shape[0]

        z_eeg_flat = F.normalize(z_eeg.view(B * N, D), p=2, dim=-1)
        z_fnirs_flat = F.normalize(z_fnirs.view(B * N, D), p=2, dim=-1)
        cb = F.normalize(codebook, p=2, dim=-1)

        # Soft assignment probabilities [B*N, K]
        logits_eeg = (z_eeg_flat @ cb.T) / self.temperature
        logits_fnirs = (z_fnirs_flat @ cb.T) / self.temperature
        probs_eeg = F.softmax(logits_eeg, dim=-1)
        probs_fnirs = F.softmax(logits_fnirs, dim=-1)

        # Cross-entropy of one modality's soft distribution against the hard
        # assignment of the other modality (if provided) or its soft distribution.
        if indices_fnirs is not None:
            targets_for_eeg = indices_fnirs.view(B * N)
            loss_eeg = F.cross_entropy(logits_eeg, targets_for_eeg)
        else:
            # KL divergence with stop-gradient on fNIRS side
            loss_eeg = F.kl_div(
                F.log_softmax(logits_eeg, dim=-1),
                probs_fnirs.detach(),
                reduction='batchmean',
            )

        if indices_eeg is not None:
            targets_for_fnirs = indices_eeg.view(B * N)
            loss_fnirs = F.cross_entropy(logits_fnirs, targets_for_fnirs)
        else:
            loss_fnirs = F.kl_div(
                F.log_softmax(logits_fnirs, dim=-1),
                probs_eeg.detach(),
                reduction='batchmean',
            )

        loss = 0.5 * (loss_eeg + loss_fnirs)

        # Diagnostic: how often do EEG and fNIRS hard-assign to the same code?
        with torch.no_grad():
            hard_eeg = logits_eeg.argmax(dim=-1)
            hard_fnirs = logits_fnirs.argmax(dim=-1)
            index_match_rate = (hard_eeg == hard_fnirs).float().mean()

        return loss, {
            'loss_eeg': loss_eeg,
            'loss_fnirs': loss_fnirs,
            'index_match_rate': index_match_rate,
        }


class SharedCodebookLoss(nn.Module):
    """
    Combined loss for joint EEG-fNIRS shared-codebook training.

    Components:
      1. Reconstruction MSE for both modalities.
      2. VQ commitment loss (from the shared quantiser).
      3. Cross-modal contrastive (InfoNCE) loss on mean-pooled tokens.
      4. Optional index-matching loss for token-level code agreement.

    Args:
        contrastive_weight:    Weight for the InfoNCE cross-modal loss (λ_c).
        index_match_weight:    Weight for the index-matching loss (λ_idx).
        temperature:           Temperature for InfoNCE.
        index_match_temperature: Temperature for IndexMatchingLoss.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        index_match_weight: float = 0.5,
        temperature: float = 0.1,
        index_match_temperature: float = 0.05,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.index_match_weight = index_match_weight

        self.token_align = TokenAlignmentLoss(temperature=temperature)
        self.index_match = IndexMatchingLoss(temperature=index_match_temperature)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        x_eeg: torch.Tensor,
        x_fnirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the combined loss.

        Args:
            outputs: Output dict from SharedCodebookTokenizer.forward().
                     Expected keys: 'x_rec_eeg', 'x_rec_fnirs', 'z_eeg', 'z_fnirs',
                     'vq_loss', 'codebook_weight', and optionally 'indices_eeg',
                     'indices_fnirs'.
            x_eeg:   [B, C_eeg, T_eeg] input EEG signal.
            x_fnirs: [B, C_fnirs, T_fnirs] input fNIRS signal.

        Returns:
            total_loss: scalar.
            losses:     dict with all component losses for logging.
        """
        losses: Dict[str, torch.Tensor] = {}

        # 1. Reconstruction losses
        losses['rec_eeg'] = F.mse_loss(outputs['x_rec_eeg'], x_eeg)
        losses['rec_fnirs'] = F.mse_loss(outputs['x_rec_fnirs'], x_fnirs)
        losses['rec'] = losses['rec_eeg'] + losses['rec_fnirs']

        # 2. VQ commitment loss
        losses['vq'] = outputs.get('vq_loss', torch.zeros(1, device=x_eeg.device))

        # 3. Cross-modal contrastive (InfoNCE) loss
        contrastive_loss, ct_info = self.token_align(
            outputs['z_eeg'], outputs['z_fnirs']
        )
        losses['contrastive'] = contrastive_loss
        losses['retrieval_acc'] = ct_info['retrieval_acc']

        # 4. Optional index-matching loss (token-level)
        if self.index_match_weight > 0 and 'codebook_weight' in outputs:
            idx_loss, idx_info = self.index_match(
                outputs['z_eeg'],
                outputs['z_fnirs'],
                outputs['codebook_weight'],
                indices_eeg=outputs.get('indices_eeg'),
                indices_fnirs=outputs.get('indices_fnirs'),
            )
            losses['index_match'] = idx_loss
            losses['index_match_rate'] = idx_info['index_match_rate']
        else:
            losses['index_match'] = torch.zeros(1, device=x_eeg.device)
            losses['index_match_rate'] = torch.zeros(1, device=x_eeg.device)

        # Total weighted loss
        total = (
            losses['rec']
            + losses['vq']
            + self.contrastive_weight * losses['contrastive']
            + self.index_match_weight * losses['index_match']
        )
        losses['total'] = total

        return total, losses
