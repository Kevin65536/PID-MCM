"""
Tokenizer Registry and Factory.

This module provides a unified interface for creating and managing tokenizers.
All tokenizers should be registered here to be usable via the unified training script.
"""

from typing import Dict, Type, Any, Optional
import torch.nn as nn


# Global tokenizer registry
_TOKENIZER_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_tokenizer(name: str):
    """
    Decorator to register a tokenizer class.
    
    Usage:
        @register_tokenizer("my_tokenizer")
        class MyTokenizer(nn.Module):
            ...
    """
    def decorator(cls):
        _TOKENIZER_REGISTRY[name] = cls
        return cls
    return decorator


def get_tokenizer_class(name: str) -> Type[nn.Module]:
    """Get a registered tokenizer class by name."""
    if name not in _TOKENIZER_REGISTRY:
        available = list(_TOKENIZER_REGISTRY.keys())
        raise ValueError(f"Unknown tokenizer: {name}. Available: {available}")
    return _TOKENIZER_REGISTRY[name]


def create_tokenizer(config: Dict[str, Any]) -> nn.Module:
    """
    Create a tokenizer from config.
    
    The config should have a 'model' section with 'type' specifying the tokenizer type.
    """
    model_cfg = config.get('model', {})
    tokenizer_type = model_cfg.get('type', 'patch_vqvae')
    
    cls = get_tokenizer_class(tokenizer_type)
    
    # Build kwargs from config
    kwargs = _build_tokenizer_kwargs(tokenizer_type, config)
    
    return cls(**kwargs)


def _build_tokenizer_kwargs(tokenizer_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build tokenizer constructor kwargs from config.
    
    Different tokenizers have different parameter structures, so we handle them here.
    """
    model_cfg = config.get('model', {})
    patch_cfg = model_cfg.get('patch', {})
    encoder_cfg = model_cfg.get('encoder', {})
    decoder_cfg = model_cfg.get('decoder', {})
    quantizer_cfg = model_cfg.get('quantizer', {})
    loss_cfg = config.get('loss', {})
    
    # Common parameters
    base_kwargs = {
        'seq_length': model_cfg.get('seq_length', 800),
        'patch_size': patch_cfg.get('size', 200),
        'input_channels': model_cfg.get('input_channels', 1),
    }

    if tokenizer_type == 'vqvae':
        return {
            'seq_length': model_cfg.get('seq_length', 800),
            'input_channels': model_cfg.get('input_channels', 1),
            'codebook_size': quantizer_cfg.get('codebook_size', 512),
            'embedding_dim': quantizer_cfg.get('embedding_dim', 64),
            'commitment_cost': quantizer_cfg.get('commitment_cost', 0.25),
            'ema_decay': quantizer_cfg.get('ema_decay', 0.99),
            'encoder_dims': encoder_cfg.get('hidden_dims', [64, 128, 256]),
            'encoder_kernel': encoder_cfg.get('kernel_size', 7),
            'encoder_stride': encoder_cfg.get('stride', 2),
        }

    elif tokenizer_type == 'fsq':
        return {
            'seq_length': model_cfg.get('seq_length', 800),
            'input_channels': model_cfg.get('input_channels', 1),
            'levels': quantizer_cfg.get('levels', [8, 8, 8, 8]),
            'encoder_dims': encoder_cfg.get('hidden_dims', [64, 128, 256]),
            'encoder_kernel': encoder_cfg.get('kernel_size', 7),
            'encoder_stride': encoder_cfg.get('stride', 2),
        }
    
    if tokenizer_type in ['patch_vqvae', 'time_patch_vqvae']:
        return {
            **base_kwargs,
            'codebook_size': quantizer_cfg.get('codebook_size', 2048),
            'embedding_dim': quantizer_cfg.get('embedding_dim', 64),
            'hidden_dim': encoder_cfg.get('hidden_dim', 256),
            'num_layers': encoder_cfg.get('num_layers', 2),
            'encoder_type': encoder_cfg.get('type', 'cnn'),
            'commitment_cost': quantizer_cfg.get('commitment_cost', 0.25),
            'ema_decay': quantizer_cfg.get('ema_decay', 0.99),
        }
    
    elif tokenizer_type in ['freq_patch_vqvae', 'freq_patch_vqvae_v2']:
        return {
            **base_kwargs,
            'codebook_size': quantizer_cfg.get('codebook_size', 2048),
            'embedding_dim': quantizer_cfg.get('embedding_dim', 64),
            'hidden_dim': encoder_cfg.get('hidden_dim', 256),
            'num_layers': encoder_cfg.get('num_layers', 2),
            'encoder_type': encoder_cfg.get('type', 'multiscale'),
            'commitment_cost': quantizer_cfg.get('commitment_cost', 0.1),
            'ema_decay': quantizer_cfg.get('ema_decay', 0.99),
            'amplitude_loss_weight': loss_cfg.get('amplitude', {}).get('weight', 1.0),
            'phase_loss_weight': loss_cfg.get('phase', {}).get('weight', 0.5),
            'time_loss_weight': loss_cfg.get('time', {}).get('weight', 1.0),
            'use_log_amplitude': loss_cfg.get('use_log_amplitude', True),
        }
    
    elif tokenizer_type in ['neurorvq']:
        return {
            'patch_size': patch_cfg.get('size', 200),
            'n_embed': quantizer_cfg.get('n_embed', quantizer_cfg.get('num_codes', 8192)),
            'code_dim': quantizer_cfg.get('code_dim', 64),
            'num_quantizers': quantizer_cfg.get('num_quantizers', 8),
            'out_chans': encoder_cfg.get('out_chans', encoder_cfg.get('hidden_channels', 8)),
            'beta': quantizer_cfg.get('beta', 1.0),
            'decay': quantizer_cfg.get('decay', 0.99),
            'kmeans_init': quantizer_cfg.get('kmeans_init', True),
        }
    
    elif tokenizer_type in ['neurorvq_fnirs']:
        return {
            'patch_size': patch_cfg.get('size', 40),
            'n_embed': quantizer_cfg.get('n_embed', quantizer_cfg.get('num_codes', 4096)),
            'code_dim': quantizer_cfg.get('code_dim', 32),
            'num_quantizers': quantizer_cfg.get('num_quantizers', 4),
            'out_chans': encoder_cfg.get('out_chans', encoder_cfg.get('hidden_channels', 8)),
            'beta': quantizer_cfg.get('beta', 1.0),
            'decay': quantizer_cfg.get('decay', 0.99),
            'kmeans_init': quantizer_cfg.get('kmeans_init', True),
        }
    
    elif tokenizer_type in ['labram_vqnsp', 'labram_vqnsp_eeg']:
        return {
            'patch_size': patch_cfg.get('size', 200),
            'seq_length': model_cfg.get('seq_length', 800),
            'encoder_embed_dim': encoder_cfg.get('embed_dim', 256),
            'encoder_depth': encoder_cfg.get('depth', 6),
            'encoder_num_heads': encoder_cfg.get('num_heads', 8),
            'decoder_embed_dim': decoder_cfg.get('embed_dim', 256),
            'decoder_depth': decoder_cfg.get('depth', 3),
            'decoder_num_heads': decoder_cfg.get('num_heads', 8),
            'codebook_size': quantizer_cfg.get('codebook_size', quantizer_cfg.get('n_embed', 8192)),
            'codebook_dim': quantizer_cfg.get('codebook_dim', quantizer_cfg.get('code_dim', 64)),
            'beta': quantizer_cfg.get('beta', 1.0),
            'decay': quantizer_cfg.get('decay', 0.99),
            'kmeans_init': quantizer_cfg.get('kmeans_init', True),
            'amplitude_weight': loss_cfg.get('amplitude', {}).get('weight', 1.0),
            'phase_weight': loss_cfg.get('phase', {}).get('weight', 1.0),
            'time_weight': loss_cfg.get('time', {}).get('weight', 0.5),
            'dropout': model_cfg.get('dropout', 0.0),
            'drop_path': model_cfg.get('drop_path', 0.1),
            'use_smooth_l1': loss_cfg.get('use_smooth_l1', False),
        }
    
    elif tokenizer_type in ['labram_vqnsp_fnirs']:
        return {
            'patch_size': patch_cfg.get('size', 40),
            'seq_length': model_cfg.get('seq_length', 200),
            'encoder_embed_dim': encoder_cfg.get('embed_dim', 128),
            'encoder_depth': encoder_cfg.get('depth', 4),
            'encoder_num_heads': encoder_cfg.get('num_heads', 4),
            'decoder_embed_dim': decoder_cfg.get('embed_dim', 128),
            'decoder_depth': decoder_cfg.get('depth', 2),
            'decoder_num_heads': decoder_cfg.get('num_heads', 4),
            'codebook_size': quantizer_cfg.get('codebook_size', quantizer_cfg.get('n_embed', 4096)),
            'codebook_dim': quantizer_cfg.get('codebook_dim', quantizer_cfg.get('code_dim', 32)),
            'beta': quantizer_cfg.get('beta', 0.5),
            'decay': quantizer_cfg.get('decay', 0.99),
            'kmeans_init': quantizer_cfg.get('kmeans_init', True),
            'amplitude_weight': loss_cfg.get('amplitude', {}).get('weight', 1.0),
            'phase_weight': loss_cfg.get('phase', {}).get('weight', 0.5),
            'time_weight': loss_cfg.get('time', {}).get('weight', 1.0),
            'dropout': model_cfg.get('dropout', 0.0),
            'drop_path': model_cfg.get('drop_path', 0.0),
            'use_smooth_l1': loss_cfg.get('use_smooth_l1', False),
        }

    elif tokenizer_type in ['shared_labram_vqnsp']:
        eeg_cfg = model_cfg.get('eeg', {})
        fnirs_cfg = model_cfg.get('fnirs', {})
        align_cfg = loss_cfg.get('alignment', {})
        eeg_loss_cfg = loss_cfg.get('eeg', {})
        fnirs_loss_cfg = loss_cfg.get('fnirs', {})
        validation_cfg = config.get('validation', {})
        return {
            'eeg_seq_length': eeg_cfg.get('seq_length', 2000),
            'eeg_patch_size': eeg_cfg.get('patch_size', 400),
            'eeg_channels': eeg_cfg.get('channels', 30),
            'eeg_encoder_embed_dim': eeg_cfg.get('encoder_embed_dim', 256),
            'eeg_encoder_depth': eeg_cfg.get('encoder_depth', 8),
            'eeg_encoder_num_heads': eeg_cfg.get('encoder_num_heads', 8),
            'eeg_decoder_embed_dim': eeg_cfg.get('decoder_embed_dim', 256),
            'eeg_decoder_depth': eeg_cfg.get('decoder_depth', 4),
            'eeg_decoder_num_heads': eeg_cfg.get('decoder_num_heads', 8),
            'fnirs_seq_length': fnirs_cfg.get('seq_length', 100),
            'fnirs_patch_size': fnirs_cfg.get('patch_size', 20),
            'fnirs_channels': fnirs_cfg.get('channels', 36),
            'fnirs_encoder_embed_dim': fnirs_cfg.get('encoder_embed_dim', 160),
            'fnirs_encoder_depth': fnirs_cfg.get('encoder_depth', 6),
            'fnirs_encoder_num_heads': fnirs_cfg.get('encoder_num_heads', 4),
            'fnirs_decoder_embed_dim': fnirs_cfg.get('decoder_embed_dim', 160),
            'fnirs_decoder_depth': fnirs_cfg.get('decoder_depth', 3),
            'fnirs_decoder_num_heads': fnirs_cfg.get('decoder_num_heads', 4),
            'codebook_size': quantizer_cfg.get('codebook_size', 4096),
            'codebook_dim': quantizer_cfg.get('codebook_dim', 64),
            'beta': quantizer_cfg.get('beta', 1.0),
            'decay': quantizer_cfg.get('decay', 0.99),
            'kmeans_init': quantizer_cfg.get('kmeans_init', True),
            'revive_dead_codes': quantizer_cfg.get('revive_dead_codes', True),
            'dead_code_threshold': quantizer_cfg.get('dead_code_threshold', 10),
            'eeg_amplitude_weight': eeg_loss_cfg.get('amplitude_weight', 1.0),
            'eeg_phase_weight': eeg_loss_cfg.get('phase_weight', 1.0),
            'eeg_time_weight': eeg_loss_cfg.get('time_weight', 0.75),
            'fnirs_amplitude_weight': fnirs_loss_cfg.get('amplitude_weight', 1.0),
            'fnirs_phase_weight': fnirs_loss_cfg.get('phase_weight', 0.25),
            'fnirs_time_weight': fnirs_loss_cfg.get('time_weight', 1.25),
            'latent_alignment_weight': align_cfg.get('latent_weight', 0.1),
            'assignment_alignment_weight': align_cfg.get('assignment_weight', 0.05),
            'assignment_temperature': align_cfg.get('temperature', 0.2),
            'alignment_lag_candidates': align_cfg.get('lag_candidates', validation_cfg.get('lag_set', [0])),
            'alignment_selection': align_cfg.get('selection', 'min'),
            'alignment_compare_mode': align_cfg.get('compare_mode', 'variable'),
            'dropout': model_cfg.get('dropout', 0.0),
            'drop_path': model_cfg.get('drop_path', 0.1),
            'use_smooth_l1': loss_cfg.get('use_smooth_l1', False),
        }

    elif tokenizer_type in ['factorized_labram_vqnsp']:
        eeg_cfg = model_cfg.get('eeg', {})
        fnirs_cfg = model_cfg.get('fnirs', {})
        shared_cfg = model_cfg.get('shared', {})
        eeg_private_cfg = model_cfg.get('eeg_private', {})
        fnirs_private_cfg = model_cfg.get('fnirs_private', {})
        branch_dropout_cfg = model_cfg.get('branch_dropout', {})
        align_cfg = loss_cfg.get('alignment', {})
        eeg_loss_cfg = loss_cfg.get('eeg', {})
        fnirs_loss_cfg = loss_cfg.get('fnirs', {})
        validation_cfg = config.get('validation', {})
        return {
            'eeg_seq_length': eeg_cfg.get('seq_length', 2000),
            'eeg_patch_size': eeg_cfg.get('patch_size', 400),
            'eeg_channels': eeg_cfg.get('channels', 30),
            'eeg_encoder_embed_dim': eeg_cfg.get('encoder_embed_dim', 256),
            'eeg_encoder_depth': eeg_cfg.get('encoder_depth', 8),
            'eeg_encoder_num_heads': eeg_cfg.get('encoder_num_heads', 8),
            'eeg_decoder_embed_dim': eeg_cfg.get('decoder_embed_dim', 256),
            'eeg_decoder_depth': eeg_cfg.get('decoder_depth', 4),
            'eeg_decoder_num_heads': eeg_cfg.get('decoder_num_heads', 8),
            'fnirs_seq_length': fnirs_cfg.get('seq_length', 100),
            'fnirs_patch_size': fnirs_cfg.get('patch_size', 20),
            'fnirs_channels': fnirs_cfg.get('channels', 36),
            'fnirs_encoder_embed_dim': fnirs_cfg.get('encoder_embed_dim', 160),
            'fnirs_encoder_depth': fnirs_cfg.get('encoder_depth', 6),
            'fnirs_encoder_num_heads': fnirs_cfg.get('encoder_num_heads', 4),
            'fnirs_decoder_embed_dim': fnirs_cfg.get('decoder_embed_dim', 160),
            'fnirs_decoder_depth': fnirs_cfg.get('decoder_depth', 3),
            'fnirs_decoder_num_heads': fnirs_cfg.get('decoder_num_heads', 4),
            'shared_codebook_size': shared_cfg.get('codebook_size', 128),
            'shared_codebook_dim': shared_cfg.get('codebook_dim', 48),
            'eeg_private_codebook_size': eeg_private_cfg.get('codebook_size', 256),
            'eeg_private_codebook_dim': eeg_private_cfg.get('codebook_dim', 64),
            'fnirs_private_codebook_size': fnirs_private_cfg.get('codebook_size', 128),
            'fnirs_private_codebook_dim': fnirs_private_cfg.get('codebook_dim', 48),
            'beta': quantizer_cfg.get('beta', 1.0),
            'decay': quantizer_cfg.get('decay', 0.99),
            'kmeans_init': quantizer_cfg.get('kmeans_init', True),
            'revive_dead_codes': quantizer_cfg.get('revive_dead_codes', True),
            'dead_code_threshold': quantizer_cfg.get('dead_code_threshold', 10),
            'eeg_amplitude_weight': eeg_loss_cfg.get('amplitude_weight', 1.0),
            'eeg_phase_weight': eeg_loss_cfg.get('phase_weight', 1.0),
            'eeg_time_weight': eeg_loss_cfg.get('time_weight', 0.75),
            'fnirs_amplitude_weight': fnirs_loss_cfg.get('amplitude_weight', 1.0),
            'fnirs_phase_weight': fnirs_loss_cfg.get('phase_weight', 0.25),
            'fnirs_time_weight': fnirs_loss_cfg.get('time_weight', 1.25),
            'latent_alignment_weight': align_cfg.get('latent_weight', 0.05),
            'coupling_weight': align_cfg.get('coupling_weight', align_cfg.get('assignment_weight', 0.05)),
            'orthogonality_weight': align_cfg.get('orthogonality_weight', 0.01),
            'assignment_temperature': align_cfg.get('temperature', 0.2),
            'alignment_lag_candidates': align_cfg.get('lag_candidates', validation_cfg.get('lag_set', [0])),
            'alignment_selection': align_cfg.get('selection', 'min'),
            'alignment_compare_mode': align_cfg.get('compare_mode', 'variable'),
            'shared_branch_dropout': branch_dropout_cfg.get('shared', 0.0),
            'eeg_private_branch_dropout': branch_dropout_cfg.get('eeg_private', 0.0),
            'fnirs_private_branch_dropout': branch_dropout_cfg.get('fnirs_private', 0.0),
            'dropout': model_cfg.get('dropout', 0.0),
            'drop_path': model_cfg.get('drop_path', 0.1),
            'use_smooth_l1': loss_cfg.get('use_smooth_l1', False),
        }
     
    else:
        raise ValueError(f"Unknown tokenizer type for kwargs building: {tokenizer_type}")


def list_tokenizers() -> list:
    """List all registered tokenizers."""
    return list(_TOKENIZER_REGISTRY.keys())


# ============================================================================
# Standard Output Interface
# ============================================================================
# 
# All tokenizers should return a dictionary with the following standardized keys:
#
# Required:
#   - 'loss': Total loss for optimization (float tensor)
#   - 'reconstructed' or 'x_rec': Reconstructed signal [B, T]
#   - 'tokens' or 'indices': Token indices
#
# Recommended loss breakdown (optional):
#   - 'rec_loss': Total reconstruction loss
#   - 'vq_loss': VQ commitment/codebook loss
#   - 'amp_loss' or 'amplitude_loss': Amplitude reconstruction loss
#   - 'phase_loss': Phase reconstruction loss  
#   - 'time_loss': Time domain reconstruction loss
#
# Codebook stats (optional):
#   - 'utilization' or 'code_utilization': Codebook utilization ratio
#   - 'perplexity': Codebook perplexity
#
# ============================================================================


class StandardizedOutput:
    """
    Helper class to standardize tokenizer outputs.
    
    Maps various tokenizer output formats to a standard interface.
    """
    
    # Mapping from various names to standard names
    LOSS_MAPPINGS = {
        'loss': 'loss',
        'total_loss': 'loss',
        'rec_loss': 'rec_loss',
        'reconstruction_loss': 'rec_loss',
        'vq_loss': 'vq_loss',
        'commitment_loss': 'vq_loss',
        'codebook_loss': 'codebook_loss',
        'amp_loss': 'amp_loss',
        'amplitude_loss': 'amp_loss',
        'phase_loss': 'phase_loss',
        'time_loss': 'time_loss',
    }
    
    SIGNAL_MAPPINGS = {
        'reconstructed': 'reconstructed',
        'x_rec': 'reconstructed',
        'reconstruction': 'reconstructed',
    }
    
    TOKEN_MAPPINGS = {
        'tokens': 'tokens',
        'indices': 'tokens',
        'token_indices': 'tokens',
    }
    
    UTIL_MAPPINGS = {
        'utilization': 'utilization',
        'code_utilization': 'utilization',
        'codebook_utilization': 'utilization',
        'usage_ratios': 'usage_ratios',
    }
    
    @classmethod
    def standardize(cls, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tokenizer outputs to standard format.
        
        This allows the training script to work with any tokenizer
        without knowing its specific output format.
        """
        result = {}
        
        # Copy all original keys
        result.update(outputs)
        
        # Add standardized aliases
        for old_key, new_key in cls.LOSS_MAPPINGS.items():
            if old_key in outputs and new_key not in result:
                result[new_key] = outputs[old_key]
        
        for old_key, new_key in cls.SIGNAL_MAPPINGS.items():
            if old_key in outputs and new_key not in result:
                result[new_key] = outputs[old_key]
        
        for old_key, new_key in cls.TOKEN_MAPPINGS.items():
            if old_key in outputs and new_key not in result:
                result[new_key] = outputs[old_key]
        
        for old_key, new_key in cls.UTIL_MAPPINGS.items():
            if old_key in outputs and new_key not in result:
                result[new_key] = outputs[old_key]
        
        # Compute total loss if not provided
        if 'loss' not in result:
            # Try to sum up component losses
            total_loss = 0.0
            if 'rec_loss' in result:
                total_loss = result['rec_loss']
            if 'vq_loss' in result:
                total_loss = total_loss + result['vq_loss']
            if total_loss != 0.0:
                result['loss'] = total_loss
        
        return result
    
    @classmethod
    def get_loss(cls, outputs: Dict[str, Any]) -> Any:
        """Get the total loss from outputs."""
        std = cls.standardize(outputs)
        return std.get('loss')
    
    @classmethod
    def get_loss_breakdown(cls, outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Get all loss components for logging.
        
        Collects any key ending with '_loss' and common loss-related keys.
        Training scripts can use this to log detailed loss breakdown.
        """
        breakdown = {}
        
        # Collect all keys ending with '_loss'
        for key, val in outputs.items():
            if key.endswith('_loss') or key in ['commitment_loss', 'codebook_loss']:
                if val is not None:
                    if hasattr(val, 'item'):
                        breakdown[key] = val.item()
                    elif isinstance(val, (int, float)):
                        breakdown[key] = float(val)
        
        return breakdown
    
    @classmethod
    def get_utilization(cls, outputs: Dict[str, Any]) -> float:
        """Get codebook utilization from outputs."""
        std = cls.standardize(outputs)
        util = std.get('utilization', 0.0)
        if hasattr(util, 'item'):
            return util.item()
        return float(util) if util else 0.0
