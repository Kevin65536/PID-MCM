import torch
import torch.nn as nn

class ELPEncoder(nn.Module):
    """
    Explicit Latent Partitioning (ELP) Encoder.
    
    Uses learnable query tokens to probe the input sequence and extract
    disjoint information components:
    - z_r: Redundancy
    - z_u_eeg: Unique EEG
    - z_u_fnirs: Unique fNIRS
    - z_s: Synergy
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Modality-specific Embeddings (Linear projection)
        self.embed_eeg = nn.Linear(input_dim, hidden_dim)
        self.embed_fnirs = nn.Linear(input_dim, hidden_dim)
        
        # 2. Learnable Query Tokens
        # We define them as parameters. 
        # Shape: [1, 1, hidden_dim] for broadcasting
        self.token_r = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.token_u_eeg = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.token_u_fnirs = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.token_s = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 3. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Projectors (Optional, if we want different spaces for losses)
        # For now, we use the raw token outputs.
        
    def forward(self, x_eeg, x_fnirs, mask_eeg=None, mask_fnirs=None):
        """
        Args:
            x_eeg: [Batch, SeqLen, Dim]
            x_fnirs: [Batch, SeqLen, Dim]
            mask_eeg: Boolean mask (True = masked/ignored)
            mask_fnirs: Boolean mask
        """
        B = x_eeg.shape[0]
        
        # Embed inputs
        emb_eeg = self.embed_eeg(x_eeg)
        emb_fnirs = self.embed_fnirs(x_fnirs)
        
        # Apply masking (replace masked tokens with learnable MASK token or zero)
        # TODO: Implement proper masking logic (e.g. MAE style)
        
        # Concatenate inputs
        # Sequence: [EEG_Tokens, fNIRS_Tokens]
        input_seq = torch.cat([emb_eeg, emb_fnirs], dim=1)
        
        # Append Query Tokens to the sequence
        # We want the query tokens to attend to the input sequence.
        # In a standard Encoder, they interact with everything.
        tokens = torch.cat([
            self.token_r.expand(B, -1, -1),
            self.token_u_eeg.expand(B, -1, -1),
            self.token_u_fnirs.expand(B, -1, -1),
            self.token_s.expand(B, -1, -1)
        ], dim=1)
        
        # Full sequence: [Queries, Inputs]
        full_seq = torch.cat([tokens, input_seq], dim=1)
        
        # Pass through Transformer
        out_seq = self.transformer(full_seq)
        
        # Extract updated Query Tokens (first 4 tokens)
        z_r = out_seq[:, 0]
        z_u_eeg = out_seq[:, 1]
        z_u_fnirs = out_seq[:, 2]
        z_s = out_seq[:, 3]
        
        return {
            'z_r': z_r,
            'z_u_eeg': z_u_eeg,
            'z_u_fnirs': z_u_fnirs,
            'z_s': z_s
        }
