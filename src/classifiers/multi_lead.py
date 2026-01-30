"""
Multi-lead classifier that processes multiple EEG/fNIRS channels.

Design:
- Tokenizer: Single-channel, applied independently to each lead
- Classifier: Aggregates token representations from multiple leads

This design separates temporal pattern encoding (tokenizer) from 
spatial aggregation (classifier).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, List, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for lead positions."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] where L is sequence length (e.g., num_leads)"""
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class MultiLeadClassifier(nn.Module):
    """
    Classifier that processes multiple leads (channels) independently through
    a tokenizer and then aggregates the representations.
    
    Architecture:
    1. Per-lead tokenization: [B, C, T] → C × [B, T'] token sequences
    2. Per-lead pooling: [B, C, T', D] → [B, C, D]
    3. Cross-lead aggregation: [B, C, D] → [B, D']
    4. Classification: [B, D'] → [B, num_classes]
    """
    
    def __init__(
        self,
        tokenizer: nn.Module,
        num_classes: int = 2,
        num_leads: int = 30,
        aggregation: Literal['mean', 'attention', 'transformer'] = 'attention',
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        freeze_tokenizer: bool = True,
    ):
        """
        Args:
            tokenizer: Pre-trained single-channel tokenizer
            num_classes: Number of output classes
            num_leads: Number of EEG/fNIRS channels
            aggregation: How to aggregate across leads
            hidden_dim: Hidden dimension for aggregation
            num_heads: Number of attention heads (for attention/transformer)
            num_layers: Number of transformer layers
            dropout: Dropout probability
            freeze_tokenizer: Whether to freeze tokenizer parameters
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.num_leads = num_leads
        self.aggregation = aggregation
        self.freeze_tokenizer = freeze_tokenizer
        
        if freeze_tokenizer:
            for param in tokenizer.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from tokenizer
        self.token_dim = tokenizer.latent_dim
        
        # Lead-level projection (optional, for dimension reduction)
        self.lead_proj = nn.Linear(self.token_dim, hidden_dim)
        
        if aggregation == 'mean':
            # Simple mean pooling across leads
            self.aggregator = nn.Identity()
            aggregated_dim = hidden_dim
            
        elif aggregation == 'attention':
            # Learnable attention over leads
            self.lead_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            aggregated_dim = hidden_dim
            
        elif aggregation == 'transformer':
            # Transformer encoder for cross-lead interaction
            self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_leads + 1, dropout=dropout)
            
            # CLS token for classification
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            aggregated_dim = hidden_dim
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(aggregated_dim),
            nn.Dropout(dropout),
            nn.Linear(aggregated_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def _tokenize_leads(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize each lead independently.
        
        Args:
            x: [B, C, T] multi-lead signal
            
        Returns:
            embeddings: [B, C, T', D] token embeddings for each lead
            indices: [B, C, T'] token indices for each lead
        """
        B, C, T = x.shape
        
        # Reshape to process all leads as batch
        x_flat = x.view(B * C, T)  # [B*C, T]
        
        with torch.set_grad_enabled(not self.freeze_tokenizer):
            out = self.tokenizer(x_flat)
        
        z_q = out['z_q']  # [B*C, T', D]
        indices = out['indices']  # [B*C, T']
        
        T_prime, D = z_q.shape[1], z_q.shape[2]
        
        # Reshape back to separate leads
        z_q = z_q.view(B, C, T_prime, D)  # [B, C, T', D]
        indices = indices.view(B, C, T_prime)  # [B, C, T']
        
        return z_q, indices
    
    def _pool_lead_tokens(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Pool tokens within each lead.
        
        Args:
            z_q: [B, C, T', D] token embeddings
            
        Returns:
            [B, C, D] pooled lead representations
        """
        return z_q.mean(dim=2)  # Average over time dimension
    
    def _aggregate_leads(self, lead_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features across leads.
        
        Args:
            lead_features: [B, C, D] lead-level features
            
        Returns:
            [B, D] aggregated features
        """
        B, C, D = lead_features.shape
        
        if self.aggregation == 'mean':
            return lead_features.mean(dim=1)  # [B, D]
            
        elif self.aggregation == 'attention':
            # Compute attention weights
            attn_scores = self.lead_attention(lead_features)  # [B, C, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, C, 1]
            
            # Weighted sum
            aggregated = (lead_features * attn_weights).sum(dim=1)  # [B, D]
            return aggregated
            
        elif self.aggregation == 'transformer':
            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            x = torch.cat([cls_tokens, lead_features], dim=1)  # [B, C+1, D]
            
            # Positional encoding
            x = self.pos_encoding(x)
            
            # Transformer
            x = self.transformer(x)  # [B, C+1, D]
            
            # Return CLS token
            return x[:, 0, :]  # [B, D]
    
    def forward(
        self,
        x: torch.Tensor,
        return_lead_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Multi-lead signal [B, C, T]
            return_lead_features: Whether to return per-lead features
            
        Returns:
            Dict with 'logits', optionally 'lead_features', 'indices', 'attention_weights'
        """
        # Tokenize each lead
        z_q, indices = self._tokenize_leads(x)  # [B, C, T', D], [B, C, T']
        
        # Pool within each lead
        lead_pooled = self._pool_lead_tokens(z_q)  # [B, C, D]
        
        # Project to hidden dimension
        lead_features = self.lead_proj(lead_pooled)  # [B, C, hidden_dim]
        
        # Aggregate across leads
        aggregated = self._aggregate_leads(lead_features)  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(aggregated)  # [B, num_classes]
        
        result = {
            'logits': logits,
            'indices': indices,
        }
        
        if return_lead_features:
            result['lead_features'] = lead_features
            result['token_embeddings'] = z_q
            
            if self.aggregation == 'attention':
                attn_scores = self.lead_attention(lead_features)
                result['attention_weights'] = F.softmax(attn_scores, dim=1).squeeze(-1)
        
        return result


class DualModalityMultiLeadClassifier(nn.Module):
    """
    Classifier for dual-modality (EEG + fNIRS) multi-lead data.
    
    Each modality is processed by its respective tokenizer, then features
    are fused for classification.
    """
    
    def __init__(
        self,
        eeg_tokenizer: nn.Module,
        fnirs_tokenizer: nn.Module,
        num_classes: int = 2,
        eeg_num_leads: int = 30,
        fnirs_num_leads: int = 36,
        aggregation: Literal['mean', 'attention', 'transformer'] = 'attention',
        fusion: Literal['early', 'late', 'cross_attention'] = 'early',
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        freeze_tokenizers: bool = True,
    ):
        """
        Args:
            eeg_tokenizer: Pre-trained EEG tokenizer (single-channel)
            fnirs_tokenizer: Pre-trained fNIRS tokenizer (single-channel)
            num_classes: Number of output classes
            eeg_num_leads: Number of EEG channels
            fnirs_num_leads: Number of fNIRS channels
            aggregation: How to aggregate within each modality
            fusion: How to fuse modalities
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            freeze_tokenizers: Whether to freeze tokenizers
        """
        super().__init__()
        
        self.eeg_tokenizer = eeg_tokenizer
        self.fnirs_tokenizer = fnirs_tokenizer
        self.fusion = fusion
        self.freeze_tokenizers = freeze_tokenizers
        
        if freeze_tokenizers:
            for param in eeg_tokenizer.parameters():
                param.requires_grad = False
            for param in fnirs_tokenizer.parameters():
                param.requires_grad = False
        
        # Per-modality multi-lead processors
        self.eeg_processor = MultiLeadProcessor(
            token_dim=eeg_tokenizer.latent_dim,
            num_leads=eeg_num_leads,
            aggregation=aggregation,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers // 2,
            dropout=dropout,
        )
        
        self.fnirs_processor = MultiLeadProcessor(
            token_dim=fnirs_tokenizer.latent_dim,
            num_leads=fnirs_num_leads,
            aggregation=aggregation,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers // 2,
            dropout=dropout,
        )
        
        # Fusion and classification
        if fusion == 'early':
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            
        elif fusion == 'late':
            self.eeg_classifier = nn.Linear(hidden_dim, num_classes)
            self.fnirs_classifier = nn.Linear(hidden_dim, num_classes)
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            
        elif fusion == 'cross_attention':
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
    
    def _tokenize_modality(
        self, 
        x: torch.Tensor, 
        tokenizer: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize multi-lead signal."""
        B, C, T = x.shape
        x_flat = x.view(B * C, T)
        
        with torch.set_grad_enabled(not self.freeze_tokenizers):
            out = tokenizer(x_flat)
        
        z_q = out['z_q']
        indices = out['indices']
        
        T_prime, D = z_q.shape[1], z_q.shape[2]
        
        z_q = z_q.view(B, C, T_prime, D)
        indices = indices.view(B, C, T_prime)
        
        return z_q, indices
    
    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            eeg: EEG signal [B, C_eeg, T_eeg]
            fnirs: fNIRS signal [B, C_fnirs, T_fnirs]
            
        Returns:
            Dict with 'logits' and intermediate features
        """
        # Tokenize each modality
        eeg_tokens, eeg_indices = self._tokenize_modality(eeg, self.eeg_tokenizer)
        fnirs_tokens, fnirs_indices = self._tokenize_modality(fnirs, self.fnirs_tokenizer)
        
        # Process each modality
        eeg_feat = self.eeg_processor(eeg_tokens)  # [B, hidden_dim]
        fnirs_feat = self.fnirs_processor(fnirs_tokens)  # [B, hidden_dim]
        
        if self.fusion == 'early':
            combined = torch.cat([eeg_feat, fnirs_feat], dim=-1)
            logits = self.classifier(combined)
            
        elif self.fusion == 'late':
            eeg_logits = self.eeg_classifier(eeg_feat)
            fnirs_logits = self.fnirs_classifier(fnirs_feat)
            w = torch.sigmoid(self.fusion_weight)
            logits = w * eeg_logits + (1 - w) * fnirs_logits
            
        elif self.fusion == 'cross_attention':
            # Cross-modal attention
            eeg_attended, _ = self.cross_attention(
                eeg_feat.unsqueeze(1),
                fnirs_feat.unsqueeze(1),
                fnirs_feat.unsqueeze(1),
            )
            eeg_attended = eeg_attended.squeeze(1)
            
            combined = torch.cat([eeg_attended, fnirs_feat], dim=-1)
            logits = self.classifier(combined)
        
        return {
            'logits': logits,
            'eeg_features': eeg_feat,
            'fnirs_features': fnirs_feat,
            'eeg_indices': eeg_indices,
            'fnirs_indices': fnirs_indices,
        }


class MultiLeadProcessor(nn.Module):
    """Helper module to process multi-lead token embeddings."""
    
    def __init__(
        self,
        token_dim: int,
        num_leads: int,
        aggregation: str,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        
        self.aggregation = aggregation
        
        self.lead_proj = nn.Linear(token_dim, hidden_dim)
        
        if aggregation == 'attention':
            self.lead_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif aggregation == 'transformer':
            self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_leads + 1, dropout=dropout)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: [B, C, T', D] token embeddings
            
        Returns:
            [B, hidden_dim] aggregated features
        """
        B, C, T_prime, D = z_q.shape
        
        # Pool within each lead
        lead_pooled = z_q.mean(dim=2)  # [B, C, D]
        
        # Project
        lead_features = self.lead_proj(lead_pooled)  # [B, C, hidden_dim]
        
        if self.aggregation == 'mean':
            return lead_features.mean(dim=1)
            
        elif self.aggregation == 'attention':
            attn_scores = self.lead_attention(lead_features)
            attn_weights = F.softmax(attn_scores, dim=1)
            return (lead_features * attn_weights).sum(dim=1)
            
        elif self.aggregation == 'transformer':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, lead_features], dim=1)
            x = self.pos_encoding(x)
            x = self.transformer(x)
            return x[:, 0, :]


class RawMultiLeadClassifier(nn.Module):
    """
    Baseline classifier that operates on raw signals directly without tokenization.
    Uses 1D CNN for temporal feature extraction followed by cross-lead aggregation.
    
    This serves as a baseline to compare against tokenizer-based approaches.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        num_leads: int = 30,
        input_length: int = 800,
        aggregation: Literal['mean', 'attention', 'transformer'] = 'attention',
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_classes: Number of output classes
            num_leads: Number of EEG/fNIRS channels
            input_length: Length of input signal per lead
            aggregation: How to aggregate across leads
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_leads = num_leads
        self.aggregation = aggregation
        self.hidden_dim = hidden_dim
        
        # Temporal feature extraction CNN (per lead)
        # Design: progressively reduce temporal dimension while increasing channels
        self.temporal_encoder = nn.Sequential(
            # Input: [B*C, 1, T]
            nn.Conv1d(1, 32, kernel_size=25, stride=4, padding=12),  # T -> T/4
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),  # T/4 -> T/8
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),  # T/8 -> T/16
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.AdaptiveAvgPool1d(4),  # Fixed output length
        )
        
        # Project pooled features
        self.lead_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        if aggregation == 'mean':
            self.aggregator = nn.Identity()
            aggregated_dim = hidden_dim
            
        elif aggregation == 'attention':
            self.lead_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            aggregated_dim = hidden_dim
            
        elif aggregation == 'transformer':
            self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_leads + 1, dropout=dropout)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            aggregated_dim = hidden_dim
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(aggregated_dim),
            nn.Dropout(dropout),
            nn.Linear(aggregated_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_lead_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Multi-lead signal [B, C, T]
            return_lead_features: Whether to return per-lead features
            
        Returns:
            Dict with 'logits', optionally 'lead_features'
        """
        B, C, T = x.shape
        
        # Reshape for per-lead processing
        x_flat = x.view(B * C, 1, T)  # [B*C, 1, T]
        
        # Extract temporal features
        temporal_features = self.temporal_encoder(x_flat)  # [B*C, hidden_dim, 4]
        temporal_features = temporal_features.flatten(1)  # [B*C, hidden_dim*4]
        
        # Project to hidden dim
        lead_features = self.lead_proj(temporal_features)  # [B*C, hidden_dim]
        lead_features = lead_features.view(B, C, self.hidden_dim)  # [B, C, hidden_dim]
        
        # Aggregate across leads
        aggregated = self._aggregate_leads(lead_features)  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(aggregated)  # [B, num_classes]
        
        result = {'logits': logits}
        
        if return_lead_features:
            result['lead_features'] = lead_features
            
        return result
    
    def _aggregate_leads(self, lead_features: torch.Tensor) -> torch.Tensor:
        """Aggregate features across leads."""
        B, C, D = lead_features.shape
        
        if self.aggregation == 'mean':
            return lead_features.mean(dim=1)
            
        elif self.aggregation == 'attention':
            attn_scores = self.lead_attention(lead_features)  # [B, C, 1]
            attn_weights = F.softmax(attn_scores, dim=1)
            aggregated = (lead_features * attn_weights).sum(dim=1)
            return aggregated
            
        elif self.aggregation == 'transformer':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, lead_features], dim=1)
            x = self.pos_encoding(x)
            x = self.transformer(x)
            return x[:, 0, :]


class RawDualModalityClassifier(nn.Module):
    """
    Baseline dual-modality classifier using raw signals without tokenization.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        eeg_num_leads: int = 30,
        fnirs_num_leads: int = 36,
        eeg_input_length: int = 800,
        fnirs_input_length: int = 40,
        aggregation: Literal['mean', 'attention', 'transformer'] = 'attention',
        fusion: Literal['early', 'late', 'cross_attention'] = 'early',
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion = fusion
        
        # EEG temporal encoder
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=4, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        
        # fNIRS temporal encoder (smaller kernel sizes for shorter signals)
        self.fnirs_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        
        # Lead projections
        self.eeg_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.fnirs_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Lead aggregation modules
        self.eeg_num_leads = eeg_num_leads
        self.fnirs_num_leads = fnirs_num_leads
        self.aggregation = aggregation
        
        if aggregation == 'attention':
            self.eeg_lead_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.fnirs_lead_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif aggregation == 'transformer':
            self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max(eeg_num_leads, fnirs_num_leads) + 1, dropout=dropout)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True,
            )
            self.eeg_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fnirs_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fusion and classification
        if fusion == 'early':
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_input_dim = hidden_dim
        elif fusion == 'late':
            classifier_input_dim = hidden_dim * 2
        elif fusion == 'cross_attention':
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_input_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion: {fusion}")
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def _process_modality(self, x: torch.Tensor, encoder: nn.Module, proj: nn.Module, 
                          lead_attn: Optional[nn.Module] = None, 
                          transformer: Optional[nn.Module] = None) -> torch.Tensor:
        """Process a single modality."""
        B, C, T = x.shape
        
        x_flat = x.view(B * C, 1, T)
        temporal_features = encoder(x_flat)
        temporal_features = temporal_features.flatten(1)
        lead_features = proj(temporal_features)
        lead_features = lead_features.view(B, C, self.hidden_dim)
        
        if self.aggregation == 'mean':
            return lead_features.mean(dim=1)
        elif self.aggregation == 'attention':
            attn_scores = lead_attn(lead_features)
            attn_weights = F.softmax(attn_scores, dim=1)
            return (lead_features * attn_weights).sum(dim=1)
        elif self.aggregation == 'transformer':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, lead_features], dim=1)
            x = self.pos_encoding(x)
            x = transformer(x)
            return x[:, 0, :]
    
    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            eeg: [B, C_eeg, T_eeg] EEG signals
            fnirs: [B, C_fnirs, T_fnirs] fNIRS signals
            
        Returns:
            Dict with 'logits'
        """
        # Process each modality
        if self.aggregation == 'attention':
            eeg_features = self._process_modality(eeg, self.eeg_encoder, self.eeg_proj, self.eeg_lead_attn)
            fnirs_features = self._process_modality(fnirs, self.fnirs_encoder, self.fnirs_proj, self.fnirs_lead_attn)
        elif self.aggregation == 'transformer':
            eeg_features = self._process_modality(eeg, self.eeg_encoder, self.eeg_proj, transformer=self.eeg_transformer)
            fnirs_features = self._process_modality(fnirs, self.fnirs_encoder, self.fnirs_proj, transformer=self.fnirs_transformer)
        else:
            eeg_features = self._process_modality(eeg, self.eeg_encoder, self.eeg_proj)
            fnirs_features = self._process_modality(fnirs, self.fnirs_encoder, self.fnirs_proj)
        
        # Fusion
        if self.fusion == 'early':
            combined = torch.cat([eeg_features, fnirs_features], dim=-1)
            fused = self.fusion_proj(combined)
        elif self.fusion == 'late':
            fused = torch.cat([eeg_features, fnirs_features], dim=-1)
        elif self.fusion == 'cross_attention':
            eeg_expanded = eeg_features.unsqueeze(1)
            fnirs_expanded = fnirs_features.unsqueeze(1)
            attn_out, _ = self.cross_attn(eeg_expanded, fnirs_expanded, fnirs_expanded)
            combined = torch.cat([eeg_features, attn_out.squeeze(1)], dim=-1)
            fused = self.fusion_proj(combined)
        
        logits = self.classifier(fused)
        
        return {'logits': logits}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/', 3)[0])
    
    from src.tokenizers import VQVAETokenizer, FSQTokenizer
    
    print("Testing MultiLeadClassifier...")
    
    # Create tokenizer (single-channel)
    tokenizer = VQVAETokenizer(
        seq_length=1000,  # 5s @ 200Hz
        input_channels=1,
        codebook_size=512,
        embedding_dim=64,
    )
    
    # Create multi-lead classifier
    for agg in ['mean', 'attention', 'transformer']:
        print(f"\n  Aggregation: {agg}")
        classifier = MultiLeadClassifier(
            tokenizer=tokenizer,
            num_classes=2,
            num_leads=30,
            aggregation=agg,
            hidden_dim=128,
            freeze_tokenizer=True,
        )
        
        # Test with multi-lead input
        x = torch.randn(4, 30, 1000)  # [B, C, T]
        outputs = classifier(x)
        
        print(f"    Input shape: {x.shape}")
        print(f"    Logits shape: {outputs['logits'].shape}")
        print(f"    Indices shape: {outputs['indices'].shape}")
    
    print("\n\nTesting DualModalityMultiLeadClassifier...")
    
    eeg_tokenizer = VQVAETokenizer(
        seq_length=1000,  # 5s @ 200Hz
        input_channels=1,
        codebook_size=512,
        embedding_dim=64,
    )
    
    fnirs_tokenizer = FSQTokenizer(
        seq_length=50,  # 5s @ 10Hz
        input_channels=1,
        levels=[8, 8, 8],
    )
    
    for fusion in ['early', 'late', 'cross_attention']:
        print(f"\n  Fusion: {fusion}")
        classifier = DualModalityMultiLeadClassifier(
            eeg_tokenizer=eeg_tokenizer,
            fnirs_tokenizer=fnirs_tokenizer,
            num_classes=2,
            eeg_num_leads=30,
            fnirs_num_leads=36,
            aggregation='attention',
            fusion=fusion,
            hidden_dim=128,
            freeze_tokenizers=True,
        )
        
        eeg = torch.randn(4, 30, 1000)
        fnirs = torch.randn(4, 36, 50)
        
        outputs = classifier(eeg, fnirs)
        print(f"    EEG shape: {eeg.shape}")
        print(f"    fNIRS shape: {fnirs.shape}")
        print(f"    Logits shape: {outputs['logits'].shape}")
    
    print("\n✓ All tests passed!")
