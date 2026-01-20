"""
Probe Experiment for EEG-fNIRS Tokenizer Coupling Analysis

This script implements the diagnostic experiments described in:
docs/reliable_survey/probe_experiment_design_for_tokenizer.md

Experiments:
1. Codebook usage and entropy analysis (single modality)
2. Token interpretability check (time-frequency / hemodynamic patterns)
3. Cross-modal conditional distribution P(z^fNIRS | z^EEG)

Usage:
    python probe_eeg_fnirs_coupling.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tokenizers.patch_vqvae import PatchVQVAETokenizer
from src.data.eeg_fnirs_dataset import EEGfNIRSDataset


class ProbeExperiment:
    """Diagnostic experiments for EEG-fNIRS tokenizer coupling."""
    
    def __init__(
        self,
        eeg_checkpoint: str,
        fnirs_checkpoint: str,
        data_root: str = "data/EEG+NIRS Single-Trial",
        output_dir: str = "experiments/probe_results",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_root = data_root
        
        # Load tokenizers
        print("Loading EEG tokenizer...")
        self.eeg_tokenizer = self._load_tokenizer(eeg_checkpoint, modality='eeg')
        print("Loading fNIRS tokenizer...")
        self.fnirs_tokenizer = self._load_tokenizer(fnirs_checkpoint, modality='fnirs')
        
        # Tokenizer specs
        self.eeg_fs = 200  # Hz
        self.fnirs_fs = 10  # Hz
        self.eeg_patch_size = 200  # 1 second
        self.fnirs_patch_size = 20  # 2 seconds
        
    def _load_tokenizer(self, checkpoint_path: str, modality: str) -> PatchVQVAETokenizer:
        """Load a tokenizer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Infer config from state dict
        state_dict = checkpoint['model_state_dict']
        
        # Get embedding dimension from quantizer
        embedding_weight = state_dict['quantizer.embedding.weight']
        codebook_size, embedding_dim = embedding_weight.shape
        
        # Determine encoder type and hidden dim
        if 'encoder.conv.0.weight' in state_dict:
            encoder_type = 'cnn'
            encoder_conv_weight = state_dict['encoder.conv.0.weight']
            hidden_dim = encoder_conv_weight.shape[0]
        elif 'encoder.encoder.0.weight' in state_dict:
            encoder_type = 'mlp'
            encoder_weight = state_dict['encoder.encoder.0.weight']
            hidden_dim = encoder_weight.shape[0]
        else:
            encoder_type = 'cnn'
            hidden_dim = 256
        
        # Determine patch size based on modality
        if modality == 'eeg':
            seq_length = 800
            patch_size = 200
        else:  # fnirs
            seq_length = 40
            patch_size = 20
        
        tokenizer = PatchVQVAETokenizer(
            seq_length=seq_length,
            patch_size=patch_size,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2 if encoder_type == 'cnn' else 3,
            encoder_type=encoder_type,
        )
        
        tokenizer.load_state_dict(state_dict)
        tokenizer.to(self.device)
        tokenizer.eval()
        
        print(f"  Loaded {modality} tokenizer: codebook={codebook_size}, dim={embedding_dim}, type={encoder_type}")
        return tokenizer
    
    def load_paired_data(self, subject_ids: list, task: str = 'motor_imagery'):
        """Load paired EEG and fNIRS data."""
        print(f"Loading paired data for subjects {subject_ids}...")
        
        # Load EEG
        eeg_dataset = EEGfNIRSDataset(
            data_root=self.data_root,
            modality='eeg',
            subject_ids=subject_ids,
            task=task,
            window_samples=800,
            window_offset_ms=500,
            normalize=True,
            exclude_eog=True,
        )
        
        # Load fNIRS
        fnirs_dataset = EEGfNIRSDataset(
            data_root=self.data_root,
            modality='fnirs',
            subject_ids=subject_ids,
            task=task,
            window_samples=40,
            window_offset_ms=500,
            normalize=True,
            hbo_only=True,
        )
        
        print(f"  EEG samples: {len(eeg_dataset)}")
        print(f"  fNIRS samples: {len(fnirs_dataset)}")
        
        return eeg_dataset, fnirs_dataset
    
    @torch.no_grad()
    def tokenize_dataset(self, dataset, tokenizer, modality: str):
        """Tokenize entire dataset and return tokens with metadata."""
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        
        all_tokens = []
        all_signals = []
        all_labels = []
        all_subjects = []
        
        for batch in loader:
            if isinstance(batch, dict):
                x = batch['data']  # [B, C, T]
                labels = batch.get('label', torch.zeros(x.shape[0]))
                subjects = batch.get('subject', torch.zeros(x.shape[0]))
            else:
                x = batch[0]
                labels = torch.zeros(x.shape[0])
                subjects = torch.zeros(x.shape[0])
            
            x = x.to(self.device)
            B, C, T = x.shape
            
            # Tokenize each channel
            x_flat = x.view(B * C, T)
            outputs = tokenizer(x_flat)
            indices = outputs['indices']  # [B*C, N_tokens]
            
            # Reshape back
            N_tokens = indices.shape[1]
            indices = indices.view(B, C, N_tokens)
            
            all_tokens.append(indices.cpu())
            all_signals.append(x.cpu())
            all_labels.append(labels)
            all_subjects.append(subjects)
        
        tokens = torch.cat(all_tokens, dim=0)
        signals = torch.cat(all_signals, dim=0)
        labels = torch.cat(all_labels, dim=0)
        subjects = torch.cat(all_subjects, dim=0)
        
        print(f"  {modality} tokenized: {tokens.shape}")
        return {
            'tokens': tokens,  # [N_samples, C, N_tokens]
            'signals': signals,  # [N_samples, C, T]
            'labels': labels,
            'subjects': subjects,
        }
    
    def experiment_1_codebook_analysis(self, tokens: torch.Tensor, codebook_size: int, 
                                       modality: str):
        """
        Experiment 1: Codebook usage rate and entropy analysis.
        """
        print(f"\n{'='*60}")
        print(f"Experiment 1: Codebook Analysis ({modality})")
        print(f"{'='*60}")
        
        # Flatten all tokens
        flat_tokens = tokens.flatten().numpy()
        
        # Token frequency histogram
        counts = np.bincount(flat_tokens, minlength=codebook_size)
        total = len(flat_tokens)
        probs = counts / total
        
        # Metrics
        used_codes = np.sum(counts > 0)
        usage_rate = used_codes / codebook_size
        
        # Entropy
        mask = probs > 0
        entropy = -np.sum(probs[mask] * np.log(probs[mask]))
        max_entropy = np.log(codebook_size)
        normalized_entropy = entropy / max_entropy
        
        # Top-k codes
        top_k = 20
        top_indices = np.argsort(counts)[::-1][:top_k]
        top_counts = counts[top_indices]
        top_probs = probs[top_indices]
        
        print(f"  Codebook size: {codebook_size}")
        print(f"  Used codes: {used_codes} ({usage_rate*100:.1f}%)")
        print(f"  Entropy: {entropy:.3f} (max: {max_entropy:.3f})")
        print(f"  Normalized entropy: {normalized_entropy:.3f}")
        print(f"  Top-5 codes: {top_indices[:5]} with probs {top_probs[:5]}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Token frequency histogram (sorted)
        sorted_counts = np.sort(counts)[::-1]
        axes[0].bar(range(len(sorted_counts)), sorted_counts, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Code Rank')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{modality} Token Frequency (Zipf-like)')
        axes[0].set_yscale('log')
        
        # 2. Cumulative distribution
        cum_probs = np.cumsum(sorted_counts) / total
        axes[1].plot(range(len(cum_probs)), cum_probs, color='darkgreen', linewidth=2)
        axes[1].axhline(0.9, color='red', linestyle='--', label='90% coverage')
        axes[1].axhline(0.95, color='orange', linestyle='--', label='95% coverage')
        axes[1].set_xlabel('Number of Codes')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title(f'{modality} Cumulative Coverage')
        axes[1].legend()
        
        # 3. Usage heatmap (if codebook is reasonable size)
        if codebook_size <= 2048:
            size = int(np.ceil(np.sqrt(codebook_size)))
            usage_grid = np.zeros(size * size)
            usage_grid[:codebook_size] = np.log1p(counts)
            usage_grid = usage_grid.reshape(size, size)
            im = axes[2].imshow(usage_grid, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=axes[2])
            axes[2].set_title(f'{modality} Codebook Usage (log scale)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'exp1_codebook_{modality}.png', dpi=150)
        plt.close()
        
        return {
            'codebook_size': codebook_size,
            'used_codes': int(used_codes),
            'usage_rate': float(usage_rate),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'top_codes': top_indices.tolist(),
            'top_probs': top_probs.tolist(),
        }
    
    def experiment_2_token_patterns(self, tokens: torch.Tensor, signals: torch.Tensor,
                                    codebook_size: int, modality: str, fs: float):
        """
        Experiment 2: Token interpretability - visualize patterns for top codes.
        """
        print(f"\n{'='*60}")
        print(f"Experiment 2: Token Pattern Analysis ({modality})")
        print(f"{'='*60}")
        
        # Get token counts
        flat_tokens = tokens.flatten().numpy()
        counts = np.bincount(flat_tokens, minlength=codebook_size)
        
        # Select top-k codes to visualize
        top_k = 8
        top_codes = np.argsort(counts)[::-1][:top_k]
        
        # For each code, collect corresponding signal patches
        # tokens: [N, C, N_tokens], signals: [N, C, T]
        N, C, N_tokens = tokens.shape
        patch_size = signals.shape[2] // N_tokens
        
        fig, axes = plt.subplots(top_k, 3, figsize=(15, top_k * 2.5))
        
        for i, code in enumerate(top_codes):
            # Find all patches with this code
            mask = (tokens == code)  # [N, C, N_tokens]
            
            # Collect patches
            patches = []
            for n in range(min(N, 100)):  # Limit samples
                for c in range(C):
                    for t in range(N_tokens):
                        if mask[n, c, t]:
                            start = t * patch_size
                            end = start + patch_size
                            patch = signals[n, c, start:end].numpy()
                            patches.append(patch)
            
            if len(patches) == 0:
                continue
                
            patches = np.array(patches)
            
            # 1. Average waveform with confidence interval
            mean_patch = np.mean(patches, axis=0)
            std_patch = np.std(patches, axis=0)
            time = np.arange(patch_size) / fs
            
            axes[i, 0].plot(time, mean_patch, 'b-', linewidth=2)
            axes[i, 0].fill_between(time, mean_patch - std_patch, mean_patch + std_patch,
                                    alpha=0.3, color='blue')
            axes[i, 0].set_title(f'Code {code} (n={len(patches)})')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Amplitude')
            
            # 2. Power spectrum
            if modality == 'eeg':
                freqs = np.fft.rfftfreq(patch_size, 1/fs)
                psds = []
                for patch in patches[:100]:
                    fft = np.abs(np.fft.rfft(patch))
                    psds.append(fft ** 2)
                mean_psd = np.mean(psds, axis=0)
                
                axes[i, 1].semilogy(freqs, mean_psd, 'g-', linewidth=2)
                axes[i, 1].set_xlabel('Frequency (Hz)')
                axes[i, 1].set_ylabel('Power')
                axes[i, 1].set_title('Power Spectrum')
                axes[i, 1].set_xlim([0, 50])
            else:  # fNIRS - show derivative (trend)
                derivatives = np.diff(patches, axis=1)
                mean_deriv = np.mean(derivatives, axis=0)
                axes[i, 1].plot(time[:-1], mean_deriv, 'r-', linewidth=2)
                axes[i, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
                axes[i, 1].set_xlabel('Time (s)')
                axes[i, 1].set_ylabel('dHbO/dt')
                axes[i, 1].set_title('HbO Derivative (Trend)')
            
            # 3. Sample patches overlay
            for j in range(min(20, len(patches))):
                axes[i, 2].plot(time, patches[j], alpha=0.3, linewidth=0.5)
            axes[i, 2].plot(time, mean_patch, 'r-', linewidth=2, label='Mean')
            axes[i, 2].set_xlabel('Time (s)')
            axes[i, 2].set_title('Sample Patches')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'exp2_token_patterns_{modality}.png', dpi=150)
        plt.close()
        
        print(f"  Visualized top {top_k} codes")
        return {'top_codes': top_codes.tolist()}
    
    def experiment_3_cross_modal_coupling(self, eeg_data: dict, fnirs_data: dict):
        """
        Experiment 3: Cross-modal conditional distribution P(z^fNIRS | z^EEG).
        
        Key insight: Each trial contains synchronized EEG and fNIRS data.
        We analyze co-occurrence patterns of tokens.
        """
        print(f"\n{'='*60}")
        print("Experiment 3: Cross-Modal Coupling Analysis")
        print(f"{'='*60}")
        
        eeg_tokens = eeg_data['tokens']  # [N, C_eeg, T_eeg]
        fnirs_tokens = fnirs_data['tokens']  # [N, C_fnirs, T_fnirs]
        eeg_labels = eeg_data['labels']
        
        N = min(eeg_tokens.shape[0], fnirs_tokens.shape[0])
        eeg_tokens = eeg_tokens[:N]
        fnirs_tokens = fnirs_tokens[:N]
        eeg_labels = eeg_labels[:N]
        
        print(f"  Paired samples: {N}")
        print(f"  EEG tokens shape: {eeg_tokens.shape}")
        print(f"  fNIRS tokens shape: {fnirs_tokens.shape}")
        
        eeg_codebook_size = self.eeg_tokenizer.codebook_size
        fnirs_codebook_size = self.fnirs_tokenizer.codebook_size
        
        # =====================================================================
        # 3.1 Joint token co-occurrence matrix
        # =====================================================================
        print("\n  3.1 Computing co-occurrence matrix...")
        
        # For each trial, count (EEG_token, fNIRS_token) pairs
        cooccurrence = np.zeros((eeg_codebook_size, fnirs_codebook_size))
        
        for n in range(N):
            eeg_flat = eeg_tokens[n].flatten().numpy()
            fnirs_flat = fnirs_tokens[n].flatten().numpy()
            
            # Count all pairs in this trial
            for e in eeg_flat:
                for f in fnirs_flat:
                    cooccurrence[e, f] += 1
        
        # Normalize to get joint probability
        joint_prob = cooccurrence / cooccurrence.sum()
        
        # Marginal probabilities
        p_eeg = joint_prob.sum(axis=1)  # P(z_EEG)
        p_fnirs = joint_prob.sum(axis=0)  # P(z_fNIRS)
        
        # =====================================================================
        # 3.2 Conditional entropy and mutual information
        # =====================================================================
        print("  3.2 Computing information-theoretic measures...")
        
        # H(fNIRS)
        mask = p_fnirs > 0
        h_fnirs = -np.sum(p_fnirs[mask] * np.log(p_fnirs[mask] + 1e-10))
        
        # H(EEG)
        mask = p_eeg > 0
        h_eeg = -np.sum(p_eeg[mask] * np.log(p_eeg[mask] + 1e-10))
        
        # H(fNIRS | EEG) = H(EEG, fNIRS) - H(EEG)
        mask = joint_prob > 0
        h_joint = -np.sum(joint_prob[mask] * np.log(joint_prob[mask] + 1e-10))
        h_fnirs_given_eeg = h_joint - h_eeg
        
        # Mutual Information I(EEG; fNIRS) = H(fNIRS) - H(fNIRS|EEG)
        mutual_info = h_fnirs - h_fnirs_given_eeg
        
        # Normalized MI
        normalized_mi = mutual_info / h_fnirs if h_fnirs > 0 else 0
        
        print(f"    H(fNIRS): {h_fnirs:.4f}")
        print(f"    H(EEG): {h_eeg:.4f}")
        print(f"    H(fNIRS|EEG): {h_fnirs_given_eeg:.4f}")
        print(f"    I(EEG; fNIRS): {mutual_info:.4f}")
        print(f"    Normalized MI: {normalized_mi:.4f}")
        print(f"    Entropy reduction: {(1 - h_fnirs_given_eeg/h_fnirs)*100:.1f}%")
        
        # =====================================================================
        # 3.3 Conditional distribution for top EEG codes
        # =====================================================================
        print("  3.3 Analyzing conditional distributions...")
        
        # Find top EEG codes
        eeg_counts = p_eeg * cooccurrence.sum()
        top_eeg_codes = np.argsort(eeg_counts)[::-1][:10]
        
        # For each top EEG code, compute P(fNIRS | EEG=e)
        conditional_dists = {}
        kl_divergences = {}
        
        for e in top_eeg_codes:
            if cooccurrence[e].sum() > 0:
                cond_prob = cooccurrence[e] / cooccurrence[e].sum()
                conditional_dists[int(e)] = cond_prob
                
                # KL divergence from marginal
                kl = 0
                for f in range(fnirs_codebook_size):
                    if cond_prob[f] > 0 and p_fnirs[f] > 0:
                        kl += cond_prob[f] * np.log(cond_prob[f] / p_fnirs[f])
                kl_divergences[int(e)] = kl
        
        # =====================================================================
        # 3.4 Shuffle baseline (destroy temporal alignment)
        # =====================================================================
        print("  3.4 Computing shuffle baseline...")
        
        # Shuffle fnirs tokens across trials
        shuffle_idx = np.random.permutation(N)
        fnirs_shuffled = fnirs_tokens[shuffle_idx]
        
        cooccurrence_shuffle = np.zeros((eeg_codebook_size, fnirs_codebook_size))
        for n in range(N):
            eeg_flat = eeg_tokens[n].flatten().numpy()
            fnirs_flat = fnirs_shuffled[n].flatten().numpy()
            for e in eeg_flat:
                for f in fnirs_flat:
                    cooccurrence_shuffle[e, f] += 1
        
        joint_shuffle = cooccurrence_shuffle / cooccurrence_shuffle.sum()
        p_eeg_shuffle = joint_shuffle.sum(axis=1)
        
        mask = joint_shuffle > 0
        h_joint_shuffle = -np.sum(joint_shuffle[mask] * np.log(joint_shuffle[mask] + 1e-10))
        mask = p_eeg_shuffle > 0
        h_eeg_shuffle = -np.sum(p_eeg_shuffle[mask] * np.log(p_eeg_shuffle[mask] + 1e-10))
        h_fnirs_given_eeg_shuffle = h_joint_shuffle - h_eeg_shuffle
        mi_shuffle = h_fnirs - h_fnirs_given_eeg_shuffle
        
        print(f"    Shuffle baseline MI: {mi_shuffle:.4f}")
        print(f"    MI improvement over baseline: {(mutual_info - mi_shuffle):.4f}")
        
        # =====================================================================
        # 3.5 Visualization
        # =====================================================================
        print("  3.5 Generating visualizations...")
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Co-occurrence matrix (downsampled for visualization)
        ax1 = fig.add_subplot(2, 3, 1)
        # Downsample if too large
        max_display = 100
        step_eeg = max(1, eeg_codebook_size // max_display)
        step_fnirs = max(1, fnirs_codebook_size // max_display)
        cooc_display = cooccurrence[::step_eeg, ::step_fnirs]
        im = ax1.imshow(np.log1p(cooc_display), cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax1)
        ax1.set_xlabel('fNIRS Token')
        ax1.set_ylabel('EEG Token')
        ax1.set_title('Co-occurrence Matrix (log scale)')
        
        # 2. Marginal distributions
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.bar(range(min(100, len(p_eeg))), np.sort(p_eeg)[::-1][:100], 
                alpha=0.7, label='P(EEG)', color='blue')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Probability')
        ax2.set_title('Marginal Distribution P(z_EEG)')
        ax2.set_yscale('log')
        
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.bar(range(min(100, len(p_fnirs))), np.sort(p_fnirs)[::-1][:100],
                alpha=0.7, label='P(fNIRS)', color='green')
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Probability')
        ax3.set_title('Marginal Distribution P(z_fNIRS)')
        ax3.set_yscale('log')
        
        # 3. Conditional vs marginal for top EEG codes
        ax4 = fig.add_subplot(2, 3, 4)
        for i, e in enumerate(list(kl_divergences.keys())[:5]):
            cond = conditional_dists[e]
            top_fnirs = np.argsort(cond)[::-1][:20]
            ax4.plot(range(20), cond[top_fnirs], '-o', alpha=0.7, 
                     label=f'EEG={e}, KL={kl_divergences[e]:.2f}')
        ax4.axhline(np.max(p_fnirs), color='k', linestyle='--', label='Max marginal')
        ax4.set_xlabel('fNIRS Token Rank')
        ax4.set_ylabel('P(fNIRS | EEG)')
        ax4.set_title('Conditional Distributions for Top EEG Codes')
        ax4.legend(fontsize=8)
        
        # 4. KL divergence distribution
        ax5 = fig.add_subplot(2, 3, 5)
        kl_values = list(kl_divergences.values())
        ax5.hist(kl_values, bins=20, color='purple', alpha=0.7)
        ax5.axvline(np.mean(kl_values), color='red', linestyle='--', 
                    label=f'Mean={np.mean(kl_values):.2f}')
        ax5.set_xlabel('KL Divergence')
        ax5.set_ylabel('Count')
        ax5.set_title('KL(P(fNIRS|EEG) || P(fNIRS))')
        ax5.legend()
        
        # 5. MI comparison: real vs shuffled
        ax6 = fig.add_subplot(2, 3, 6)
        bars = ax6.bar(['Real', 'Shuffled'], [mutual_info, mi_shuffle], 
                       color=['forestgreen', 'lightcoral'])
        ax6.set_ylabel('Mutual Information (nats)')
        ax6.set_title('EEG-fNIRS Coupling Strength')
        ax6.bar_label(bars, fmt='%.3f')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp3_cross_modal_coupling.png', dpi=150)
        plt.close()
        
        # =====================================================================
        # 3.6 Task-specific analysis
        # =====================================================================
        print("  3.6 Task-specific coupling analysis...")
        
        unique_labels = torch.unique(eeg_labels).numpy()
        task_mi = {}
        
        fig, axes = plt.subplots(1, len(unique_labels), figsize=(5*len(unique_labels), 4))
        if len(unique_labels) == 1:
            axes = [axes]
        
        for idx, label in enumerate(unique_labels):
            mask = eeg_labels.numpy() == label
            n_samples = mask.sum()
            
            if n_samples < 10:
                continue
            
            eeg_task = eeg_tokens[mask]
            fnirs_task = fnirs_tokens[mask]
            
            # Compute task-specific cooccurrence
            cooc_task = np.zeros((eeg_codebook_size, fnirs_codebook_size))
            for n in range(eeg_task.shape[0]):
                eeg_flat = eeg_task[n].flatten().numpy()
                fnirs_flat = fnirs_task[n].flatten().numpy()
                for e in eeg_flat:
                    for f in fnirs_flat:
                        cooc_task[e, f] += 1
            
            joint_task = cooc_task / (cooc_task.sum() + 1e-10)
            p_eeg_task = joint_task.sum(axis=1)
            p_fnirs_task = joint_task.sum(axis=0)
            
            mask_j = joint_task > 0
            h_joint_task = -np.sum(joint_task[mask_j] * np.log(joint_task[mask_j] + 1e-10))
            mask_e = p_eeg_task > 0
            h_eeg_task = -np.sum(p_eeg_task[mask_e] * np.log(p_eeg_task[mask_e] + 1e-10))
            mask_f = p_fnirs_task > 0
            h_fnirs_task = -np.sum(p_fnirs_task[mask_f] * np.log(p_fnirs_task[mask_f] + 1e-10))
            
            h_fnirs_given_eeg_task = h_joint_task - h_eeg_task
            mi_task = h_fnirs_task - h_fnirs_given_eeg_task
            task_mi[int(label)] = float(mi_task)
            
            # Visualize
            cooc_display = cooc_task[::step_eeg, ::step_fnirs]
            im = axes[idx].imshow(np.log1p(cooc_display), cmap='hot', aspect='auto')
            axes[idx].set_title(f'Task {int(label)} (n={n_samples})\nMI={mi_task:.3f}')
            axes[idx].set_xlabel('fNIRS Token')
            axes[idx].set_ylabel('EEG Token')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp3_task_specific_coupling.png', dpi=150)
        plt.close()
        
        print(f"    Task-specific MI: {task_mi}")
        
        return {
            'h_fnirs': float(h_fnirs),
            'h_eeg': float(h_eeg),
            'h_fnirs_given_eeg': float(h_fnirs_given_eeg),
            'mutual_information': float(mutual_info),
            'normalized_mi': float(normalized_mi),
            'entropy_reduction': float((1 - h_fnirs_given_eeg/h_fnirs)),
            'mi_shuffle_baseline': float(mi_shuffle),
            'mi_improvement': float(mutual_info - mi_shuffle),
            'task_mi': task_mi,
            'kl_divergences': {k: float(v) for k, v in kl_divergences.items()},
        }
    
    def run_all_experiments(self, subject_ids: list = None):
        """Run all probe experiments."""
        if subject_ids is None:
            subject_ids = list(range(1, 26))  # Train + val subjects
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_dir / f"probe_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nOutput directory: {self.output_dir}")
        
        # Load data
        eeg_dataset, fnirs_dataset = self.load_paired_data(subject_ids)
        
        # Tokenize
        print("\nTokenizing datasets...")
        eeg_data = self.tokenize_dataset(eeg_dataset, self.eeg_tokenizer, 'eeg')
        fnirs_data = self.tokenize_dataset(fnirs_dataset, self.fnirs_tokenizer, 'fnirs')
        
        results = {}
        
        # Experiment 1: Codebook analysis
        results['exp1_eeg'] = self.experiment_1_codebook_analysis(
            eeg_data['tokens'], self.eeg_tokenizer.codebook_size, 'EEG'
        )
        results['exp1_fnirs'] = self.experiment_1_codebook_analysis(
            fnirs_data['tokens'], self.fnirs_tokenizer.codebook_size, 'fNIRS'
        )
        
        # Experiment 2: Token patterns
        results['exp2_eeg'] = self.experiment_2_token_patterns(
            eeg_data['tokens'], eeg_data['signals'],
            self.eeg_tokenizer.codebook_size, 'EEG', self.eeg_fs
        )
        results['exp2_fnirs'] = self.experiment_2_token_patterns(
            fnirs_data['tokens'], fnirs_data['signals'],
            self.fnirs_tokenizer.codebook_size, 'fNIRS', self.fnirs_fs
        )
        
        # Experiment 3: Cross-modal coupling
        results['exp3_coupling'] = self.experiment_3_cross_modal_coupling(eeg_data, fnirs_data)
        
        # Save results
        with open(self.output_dir / 'probe_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("All experiments completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")
        
        return results


def main():
    # Paths to trained tokenizers
    eeg_checkpoint = "experiments/runs/eeg_patch_vqvae_1s_20260116_185829/checkpoints/best_model.pt"
    fnirs_checkpoint = "experiments/runs/fnirs_patch_vqvae_2s_v2_20260119_115413/checkpoints/best_model.pt"
    
    # Initialize experiment
    probe = ProbeExperiment(
        eeg_checkpoint=eeg_checkpoint,
        fnirs_checkpoint=fnirs_checkpoint,
        data_root="data/EEG+NIRS Single-Trial",
        output_dir="experiments/probe_results",
    )
    
    # Run all experiments
    results = probe.run_all_experiments(subject_ids=list(range(1, 26)))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nEEG Codebook:")
    print(f"  Usage rate: {results['exp1_eeg']['usage_rate']*100:.1f}%")
    print(f"  Normalized entropy: {results['exp1_eeg']['normalized_entropy']:.3f}")
    
    print(f"\nfNIRS Codebook:")
    print(f"  Usage rate: {results['exp1_fnirs']['usage_rate']*100:.1f}%")
    print(f"  Normalized entropy: {results['exp1_fnirs']['normalized_entropy']:.3f}")
    
    print(f"\nCross-Modal Coupling:")
    print(f"  Mutual Information: {results['exp3_coupling']['mutual_information']:.4f}")
    print(f"  Normalized MI: {results['exp3_coupling']['normalized_mi']:.4f}")
    print(f"  Entropy Reduction: {results['exp3_coupling']['entropy_reduction']*100:.1f}%")
    print(f"  MI vs Shuffle: {results['exp3_coupling']['mi_improvement']:.4f} improvement")


if __name__ == '__main__':
    main()
