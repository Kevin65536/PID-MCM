"""
Simple classifiers for token-based representations.

These classifiers take either:
1. Token indices [B, T'] - converted to embeddings via codebook lookup
2. Quantized latents [B, T', D] - direct continuous representations
3. Pre-quantized latents [B, T', D] - encoder outputs before quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict, Any


class TokenClassifierHead(nn.Module):
    """
    Simple classification head: Pool -> Linear.
    
    Supports multiple pooling strategies and optional hidden layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pool_type: Literal['mean', 'max', 'first', 'last', 'attention'] = 'mean',
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input features (D)
            num_classes: Number of output classes
            pool_type: Pooling strategy
            hidden_dims: Optional hidden layer dimensions (e.g., [128])
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pool_type = pool_type
        
        # Attention pooling if needed
        if pool_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.Tanh(),
                nn.Linear(input_dim // 4, 1),
            )
        
        # Build classification layers
        layers = []
        in_dim = input_dim
        
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool temporal dimension.
        
        Args:
            x: [B, T', D]
            
        Returns:
            pooled: [B, D]
        """
        if self.pool_type == 'mean':
            return x.mean(dim=1)
        elif self.pool_type == 'max':
            return x.max(dim=1)[0]
        elif self.pool_type == 'first':
            return x[:, 0, :]
        elif self.pool_type == 'last':
            return x[:, -1, :]
        elif self.pool_type == 'attention':
            # Compute attention weights
            attn_scores = self.attention(x)  # [B, T', 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, T', 1]
            return (x * attn_weights).sum(dim=1)  # [B, D]
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T', D] sequence of embeddings
            
        Returns:
            logits: [B, num_classes]
        """
        pooled = self._pool(x)  # [B, D]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


class SimpleTokenClassifier(nn.Module):
    """
    Complete classifier that works with tokenized representations.
    
    Can operate in multiple modes:
    - 'indices': Takes token indices, looks up embeddings from codebook
    - 'embeddings': Takes quantized embeddings directly
    - 'both': Supports both input types
    
    The tokenizer can be optionally frozen or fine-tuned.
    """
    
    def __init__(
        self,
        tokenizer: Optional[nn.Module] = None,
        embedding_dim: int = 64,
        num_classes: int = 2,
        codebook_size: Optional[int] = None,
        pool_type: Literal['mean', 'max', 'first', 'last', 'attention'] = 'mean',
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        freeze_tokenizer: bool = True,
        input_mode: Literal['indices', 'embeddings', 'raw'] = 'embeddings',
    ):
        """
        Args:
            tokenizer: Pre-trained tokenizer (optional, needed for 'raw' mode)
            embedding_dim: Dimension of token embeddings
            num_classes: Number of output classes
            codebook_size: Size of codebook (for learnable embedding if no tokenizer)
            pool_type: Pooling strategy
            hidden_dims: Hidden layer dimensions for classifier
            dropout: Dropout probability
            freeze_tokenizer: Whether to freeze tokenizer during training
            input_mode: 
                - 'indices': Input is token indices, lookup embeddings
                - 'embeddings': Input is already embeddings [B, T', D]
                - 'raw': Input is raw signal, use tokenizer to encode
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.input_mode = input_mode
        self.freeze_tokenizer = freeze_tokenizer
        
        # Tokenizer for raw mode
        self.tokenizer = tokenizer
        if tokenizer is not None and freeze_tokenizer:
            for param in tokenizer.parameters():
                param.requires_grad = False
        
        # Learnable embedding if indices mode without tokenizer
        if input_mode == 'indices' and tokenizer is None:
            if codebook_size is None:
                raise ValueError("codebook_size required for indices mode without tokenizer")
            self.embedding = nn.Embedding(codebook_size, embedding_dim)
        else:
            self.embedding = None
        
        # Classification head
        self.classifier = TokenClassifierHead(
            input_dim=embedding_dim,
            num_classes=num_classes,
            pool_type=pool_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        mode: Optional[str] = None
    ) -> torch.Tensor:
        """
        Get embeddings from input based on mode.
        
        Args:
            x: Input tensor (shape depends on mode)
            mode: Override default input_mode
            
        Returns:
            embeddings: [B, T', D]
        """
        mode = mode or self.input_mode
        
        if mode == 'embeddings':
            # Input is already embeddings
            return x
        
        elif mode == 'indices':
            # Input is token indices [B, T']
            if self.tokenizer is not None:
                return self.tokenizer.get_embedding(x)
            elif self.embedding is not None:
                return self.embedding(x)
            else:
                raise ValueError("No embedding source for indices mode")
        
        elif mode == 'raw':
            # Input is raw signal, encode with tokenizer
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for raw mode")
            
            with torch.set_grad_enabled(not self.freeze_tokenizer):
                outputs = self.tokenizer(x)
                return outputs['z_q']  # Use quantized latents
        
        else:
            raise ValueError(f"Unknown input mode: {mode}")
    
    def forward(
        self, 
        x: torch.Tensor,
        mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (shape depends on mode)
            mode: Override default input_mode
            
        Returns:
            Dict with 'logits' and optionally 'embeddings'
        """
        embeddings = self.get_embeddings(x, mode)
        logits = self.classifier(embeddings)
        
        return {
            'logits': logits,
            'embeddings': embeddings,
        }
    
    def predict(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """Get class predictions."""
        outputs = self.forward(x, mode)
        return outputs['logits'].argmax(dim=-1)
    
    def predict_proba(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """Get class probabilities."""
        outputs = self.forward(x, mode)
        return F.softmax(outputs['logits'], dim=-1)


class RawSignalClassifier(nn.Module):
    """
    Baseline classifier that works directly on raw signals without tokenization.
    
    Uses the same encoder architecture as tokenizers for fair comparison.
    """
    
    def __init__(
        self,
        seq_length: int = 512,
        input_channels: int = 1,
        num_classes: int = 2,
        encoder_dims: list = [64, 128, 256],
        encoder_kernel: int = 7,
        encoder_stride: int = 2,
        latent_dim: int = 64,
        pool_type: str = 'mean',
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            seq_length: Input sequence length
            input_channels: Number of input channels
            num_classes: Number of output classes
            encoder_dims: Encoder hidden dimensions (same as tokenizer)
            encoder_kernel: Encoder kernel size
            encoder_stride: Encoder stride
            latent_dim: Latent dimension
            pool_type: Pooling type
            hidden_dims: Classifier hidden dims
            dropout: Dropout rate
        """
        super().__init__()
        
        # Import encoder from tokenizer base
        from ..tokenizers.base import Conv1dEncoder
        
        self.encoder = Conv1dEncoder(
            input_dim=input_channels,
            hidden_dims=encoder_dims,
            kernel_size=encoder_kernel,
            stride=encoder_stride,
            output_dim=latent_dim,
        )
        
        self.classifier = TokenClassifierHead(
            input_dim=latent_dim,
            num_classes=num_classes,
            pool_type=pool_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Raw signal [B, T] or [B, C, T]
            
        Returns:
            Dict with 'logits' and 'embeddings'
        """
        embeddings = self.encoder(x)  # [B, T', D]
        logits = self.classifier(embeddings)  # [B, num_classes]
        
        return {
            'logits': logits,
            'embeddings': embeddings,
        }


if __name__ == '__main__':
    print("Testing SimpleTokenClassifier...")
    
    # Test with embeddings input
    batch_size = 8
    seq_len = 32
    embed_dim = 64
    num_classes = 2
    
    classifier = SimpleTokenClassifier(
        embedding_dim=embed_dim,
        num_classes=num_classes,
        pool_type='attention',
        hidden_dims=[128],
        dropout=0.1,
        input_mode='embeddings',
    )
    
    # Test forward
    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    outputs = classifier(embeddings)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    
    # Test with indices input
    classifier_idx = SimpleTokenClassifier(
        embedding_dim=embed_dim,
        num_classes=num_classes,
        codebook_size=512,
        pool_type='mean',
        input_mode='indices',
    )
    
    indices = torch.randint(0, 512, (batch_size, seq_len))
    outputs_idx = classifier_idx(indices)
    
    print(f"\nIndices input shape: {indices.shape}")
    print(f"Logits shape: {outputs_idx['logits'].shape}")
    
    # Test RawSignalClassifier
    print("\nTesting RawSignalClassifier...")
    
    raw_classifier = RawSignalClassifier(
        seq_length=512,
        input_channels=1,
        num_classes=2,
        encoder_dims=[32, 64, 128],
        latent_dim=64,
        pool_type='mean',
    )
    
    raw_signal = torch.randn(batch_size, 512)
    raw_outputs = raw_classifier(raw_signal)
    
    print(f"Raw signal shape: {raw_signal.shape}")
    print(f"Logits shape: {raw_outputs['logits'].shape}")
    print(f"Embeddings shape: {raw_outputs['embeddings'].shape}")
    
    print("\n✓ All tests passed!")
