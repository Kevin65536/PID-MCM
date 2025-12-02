"""
PID-specific loss functions for the ELP framework.

Reference: docs/pid_mcm_proposal.md (Section 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    Forces redundancy tokens from different views to align.
    
    L_align = ||z_r^A - z_r^B||^2
    
    This ensures z_r captures only information accessible from BOTH modalities.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, z_r_view_a, z_r_view_b):
        """
        Args:
            z_r_view_a: [Batch, Dim] - Redundancy token from view A
            z_r_view_b: [Batch, Dim] - Redundancy token from view B
        """
        return F.mse_loss(z_r_view_a, z_r_view_b)


class OrthogonalityLoss(nn.Module):
    """
    Enforces disjointness between latent components.
    
    L_orth = |cos(z_r, z_u)| + |cos(z_r, z_s)| + |cos(z_u, z_s)|
    
    Ensures information atoms don't overlap (geometric proxy for disjointness).
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, z_r, z_u_eeg, z_u_fnirs, z_s):
        """
        Args:
            z_r, z_u_eeg, z_u_fnirs, z_s: [Batch, Dim]
        """
        def cosine_sim(a, b):
            return F.cosine_similarity(a, b, dim=1).abs().mean()
        
        loss = 0.0
        # All pairwise orthogonality
        loss += cosine_sim(z_r, z_u_eeg)
        loss += cosine_sim(z_r, z_u_fnirs)
        loss += cosine_sim(z_r, z_s)
        loss += cosine_sim(z_u_eeg, z_u_fnirs)
        loss += cosine_sim(z_u_eeg, z_s)
        loss += cosine_sim(z_u_fnirs, z_s)
        
        return loss / 6.0  # Average


class SynergyLoss(nn.Module):
    """
    Ensures synergy token changes when modalities are masked.
    
    L_syn = -||z_s^{joint} - z_s^{masked}||^2
    
    Negative sign: We WANT them to be different (synergy requires both modalities).
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, z_s_joint, z_s_masked):
        """
        Args:
            z_s_joint: [Batch, Dim] - Synergy from full input
            z_s_masked: [Batch, Dim] - Synergy when one modality is masked
        """
        # We want to MAXIMIZE the difference
        diff = F.mse_loss(z_s_joint, z_s_masked)
        return -diff  # Negative: gradient descent will increase difference


class ReconstructionLoss(nn.Module):
    """
    Standard MSE reconstruction loss.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [Batch, SeqLen, Dim]
            target: [Batch, SeqLen, Dim]
            mask: Optional [Batch, SeqLen] - only compute loss on masked positions
        """
        if mask is not None:
            # Expand mask to match feature dimension
            mask = mask.unsqueeze(-1).expand_as(pred)
            loss = F.mse_loss(pred * mask, target * mask, reduction='sum')
            return loss / mask.sum()
        else:
            return F.mse_loss(pred, target)


class PIDTotalLoss(nn.Module):
    """
    Combined loss for ELP training.
    
    L_total = L_rec + λ1*L_align + λ2*L_orth + λ3*L_syn
    """
    def __init__(self, lambda_align=0.5, lambda_orth=0.3, lambda_syn=0.2):
        super().__init__()
        
        self.lambda_align = lambda_align
        self.lambda_orth = lambda_orth
        self.lambda_syn = lambda_syn
        
        self.reconstruction_loss = ReconstructionLoss()
        self.alignment_loss = AlignmentLoss()
        self.orthogonality_loss = OrthogonalityLoss()
        self.synergy_loss = SynergyLoss()
        
    def forward(self, outputs, targets, masking_mode='joint'):
        """
        Args:
            outputs: Dict from model forward pass
            targets: Dict with ground truth signals
            masking_mode: 'cross', 'uni', or 'joint'
        """
        losses = {}
        
        # Reconstruction always active
        losses['rec'] = self.reconstruction_loss(
            outputs['reconstructed'], 
            targets['signals']
        )
        
        # Alignment active in cross-modal mode
        if masking_mode == 'cross' and 'z_r_view_a' in outputs:
            losses['align'] = self.alignment_loss(
                outputs['z_r_view_a'], 
                outputs['z_r_view_b']
            )
        else:
            losses['align'] = torch.tensor(0.0, device=outputs['z_r'].device)
        
        # Orthogonality active in uni-modal and joint modes
        if masking_mode in ['uni', 'joint']:
            losses['orth'] = self.orthogonality_loss(
                outputs['z_r'],
                outputs['z_u_eeg'],
                outputs['z_u_fnirs'],
                outputs['z_s']
            )
        else:
            losses['orth'] = torch.tensor(0.0, device=outputs['z_r'].device)
        
        # Synergy active in joint mode
        if masking_mode == 'joint' and 'z_s_joint' in outputs:
            losses['syn'] = self.synergy_loss(
                outputs['z_s_joint'],
                outputs['z_s_masked']
            )
        else:
            losses['syn'] = torch.tensor(0.0, device=outputs['z_r'].device)
        
        # Total weighted loss
        total = (losses['rec'] + 
                self.lambda_align * losses['align'] +
                self.lambda_orth * losses['orth'] +
                self.lambda_syn * losses['syn'])
        
        losses['total'] = total
        return losses
