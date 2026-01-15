"""
End-to-end classifier that combines tokenizer and classifier.

Supports:
1. Single modality (EEG or fNIRS)
2. Multi-modality fusion (early, late, or hybrid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, Tuple, Any

from .simple_classifier import TokenClassifierHead


class EndToEndClassifier(nn.Module):
    """
    End-to-end pipeline: Raw Signal → Tokenizer → Classifier
    
    The tokenizer can be pre-trained and frozen, or jointly fine-tuned.
    """
    
    def __init__(
        self,
        tokenizer: nn.Module,
        num_classes: int = 2,
        pool_type: str = 'mean',
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        freeze_tokenizer: bool = True,
        use_pre_quantized: bool = False,
    ):
        """
        Args:
            tokenizer: Pre-trained tokenizer module
            num_classes: Number of output classes
            pool_type: Pooling strategy
            hidden_dims: Hidden layer dimensions for classifier
            dropout: Dropout probability
            freeze_tokenizer: Whether to freeze tokenizer parameters
            use_pre_quantized: If True, use encoder output (z) instead of quantized (z_q)
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.freeze_tokenizer = freeze_tokenizer
        self.use_pre_quantized = use_pre_quantized
        
        if freeze_tokenizer:
            for param in tokenizer.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from tokenizer
        self.embedding_dim = tokenizer.latent_dim
        
        # Classification head
        self.classifier = TokenClassifierHead(
            input_dim=self.embedding_dim,
            num_classes=num_classes,
            pool_type=pool_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_tokenizer_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Raw signal [B, T] or [B, C, T]
            return_tokenizer_outputs: Whether to include tokenizer outputs
            
        Returns:
            Dict with 'logits', optionally 'embeddings' and tokenizer outputs
        """
        # Encode with tokenizer
        with torch.set_grad_enabled(not self.freeze_tokenizer):
            tokenizer_out = self.tokenizer(x)
        
        # Choose representation
        if self.use_pre_quantized:
            embeddings = tokenizer_out['z']  # Pre-quantized
        else:
            embeddings = tokenizer_out['z_q']  # Quantized
        
        # Classify
        logits = self.classifier(embeddings)
        
        result = {
            'logits': logits,
            'embeddings': embeddings,
            'indices': tokenizer_out['indices'],
        }
        
        if return_tokenizer_outputs:
            result['tokenizer_outputs'] = tokenizer_out
            
        return result


class MultiModalClassifier(nn.Module):
    """
    Multi-modal classifier for EEG + fNIRS fusion.
    
    Supports different fusion strategies:
    - early: Concatenate features before pooling
    - late: Separate classifiers, average logits
    - hybrid: Project to shared space, then early fusion
    """
    
    def __init__(
        self,
        eeg_tokenizer: nn.Module,
        fnirs_tokenizer: nn.Module,
        num_classes: int = 2,
        fusion_type: Literal['early', 'late', 'hybrid'] = 'early',
        pool_type: str = 'mean',
        hidden_dims: Optional[list] = None,
        projection_dim: Optional[int] = None,
        dropout: float = 0.1,
        freeze_tokenizers: bool = True,
    ):
        """
        Args:
            eeg_tokenizer: Pre-trained EEG tokenizer
            fnirs_tokenizer: Pre-trained fNIRS tokenizer
            num_classes: Number of output classes
            fusion_type: Fusion strategy
            pool_type: Pooling strategy
            hidden_dims: Hidden layer dimensions
            projection_dim: Dimension for hybrid fusion projection
            dropout: Dropout probability
            freeze_tokenizers: Whether to freeze both tokenizers
        """
        super().__init__()
        
        self.eeg_tokenizer = eeg_tokenizer
        self.fnirs_tokenizer = fnirs_tokenizer
        self.fusion_type = fusion_type
        self.freeze_tokenizers = freeze_tokenizers
        
        if freeze_tokenizers:
            for param in eeg_tokenizer.parameters():
                param.requires_grad = False
            for param in fnirs_tokenizer.parameters():
                param.requires_grad = False
        
        self.eeg_dim = eeg_tokenizer.latent_dim
        self.fnirs_dim = fnirs_tokenizer.latent_dim
        
        if fusion_type == 'early':
            # Concatenate embeddings, joint classifier
            combined_dim = self.eeg_dim + self.fnirs_dim
            self.classifier = TokenClassifierHead(
                input_dim=combined_dim,
                num_classes=num_classes,
                pool_type=pool_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            
        elif fusion_type == 'late':
            # Separate classifiers
            self.eeg_classifier = TokenClassifierHead(
                input_dim=self.eeg_dim,
                num_classes=num_classes,
                pool_type=pool_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            self.fnirs_classifier = TokenClassifierHead(
                input_dim=self.fnirs_dim,
                num_classes=num_classes,
                pool_type=pool_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            # Learnable fusion weights
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            
        elif fusion_type == 'hybrid':
            # Project to shared space
            proj_dim = projection_dim or max(self.eeg_dim, self.fnirs_dim)
            self.eeg_proj = nn.Linear(self.eeg_dim, proj_dim)
            self.fnirs_proj = nn.Linear(self.fnirs_dim, proj_dim)
            
            # Combined classifier
            self.classifier = TokenClassifierHead(
                input_dim=proj_dim * 2,
                num_classes=num_classes,
                pool_type=pool_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def _encode_modality(
        self, 
        x: torch.Tensor, 
        tokenizer: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single modality."""
        with torch.set_grad_enabled(not self.freeze_tokenizers):
            outputs = tokenizer(x)
        return outputs['z_q'], outputs['indices']
    
    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            eeg: EEG signal [B, T_eeg] or [B, C_eeg, T_eeg]
            fnirs: fNIRS signal [B, T_fnirs] or [B, C_fnirs, T_fnirs]
            
        Returns:
            Dict with 'logits', 'eeg_embeddings', 'fnirs_embeddings'
        """
        # Encode both modalities
        eeg_emb, eeg_indices = self._encode_modality(eeg, self.eeg_tokenizer)
        fnirs_emb, fnirs_indices = self._encode_modality(fnirs, self.fnirs_tokenizer)
        
        if self.fusion_type == 'early':
            # Pool each modality first (since they have different lengths)
            eeg_pooled = eeg_emb.mean(dim=1)  # [B, D_eeg]
            fnirs_pooled = fnirs_emb.mean(dim=1)  # [B, D_fnirs]
            
            # Concatenate
            combined = torch.cat([eeg_pooled, fnirs_pooled], dim=-1)  # [B, D_eeg + D_fnirs]
            combined = self.dropout(combined)
            
            # Add temporal dimension back for classifier
            combined = combined.unsqueeze(1)  # [B, 1, D]
            logits = self.classifier(combined)
            
        elif self.fusion_type == 'late':
            # Separate classification
            eeg_logits = self.eeg_classifier(eeg_emb)
            fnirs_logits = self.fnirs_classifier(fnirs_emb)
            
            # Weighted average
            w = torch.sigmoid(self.fusion_weight)
            logits = w * eeg_logits + (1 - w) * fnirs_logits
            
        elif self.fusion_type == 'hybrid':
            # Project to shared space
            eeg_proj = self.eeg_proj(eeg_emb.mean(dim=1))  # [B, proj_dim]
            fnirs_proj = self.fnirs_proj(fnirs_emb.mean(dim=1))  # [B, proj_dim]
            
            # Concatenate projections
            combined = torch.cat([eeg_proj, fnirs_proj], dim=-1)
            combined = self.dropout(combined)
            combined = combined.unsqueeze(1)
            
            logits = self.classifier(combined)
        
        return {
            'logits': logits,
            'eeg_embeddings': eeg_emb,
            'fnirs_embeddings': fnirs_emb,
            'eeg_indices': eeg_indices,
            'fnirs_indices': fnirs_indices,
        }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/', 3)[0])
    
    from src.tokenizers import VQVAETokenizer
    
    print("Testing EndToEndClassifier...")
    
    # Create tokenizer
    tokenizer = VQVAETokenizer(
        seq_length=512,
        input_channels=1,
        codebook_size=512,
        embedding_dim=64,
    )
    
    # Create classifier
    classifier = EndToEndClassifier(
        tokenizer=tokenizer,
        num_classes=2,
        pool_type='mean',
        hidden_dims=[128],
        freeze_tokenizer=True,
    )
    
    # Test forward
    x = torch.randn(8, 512)  # [B, T]
    outputs = classifier(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    print(f"Indices shape: {outputs['indices'].shape}")
    
    # Test MultiModalClassifier
    print("\nTesting MultiModalClassifier...")
    
    eeg_tokenizer = VQVAETokenizer(
        seq_length=512,
        input_channels=1,
        codebook_size=512,
        embedding_dim=64,
    )
    
    fnirs_tokenizer = VQVAETokenizer(
        seq_length=26,  # 2.5s @ 10Hz
        input_channels=1,
        codebook_size=256,
        embedding_dim=32,
    )
    
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\n  Fusion type: {fusion_type}")
        mm_classifier = MultiModalClassifier(
            eeg_tokenizer=eeg_tokenizer,
            fnirs_tokenizer=fnirs_tokenizer,
            num_classes=2,
            fusion_type=fusion_type,
            hidden_dims=[64],
        )
        
        eeg_signal = torch.randn(8, 512)
        fnirs_signal = torch.randn(8, 26)
        
        mm_outputs = mm_classifier(eeg_signal, fnirs_signal)
        print(f"    Logits shape: {mm_outputs['logits'].shape}")
    
    print("\n✓ All tests passed!")
