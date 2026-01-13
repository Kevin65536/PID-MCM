"""
Visualization tools for tokenizer training experiments.

Generates and saves:
1. Training curves (loss, perplexity, utilization)
2. Reconstruction quality samples
3. Spectral comparison (PSD)
4. Codebook usage histogram
5. Token embedding visualization (t-SNE/PCA)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

# Matplotlib setup
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Optional: sklearn for t-SNE
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TokenizerVisualizer:
    """
    Visualization toolkit for tokenizer experiments.
    
    Usage:
        visualizer = TokenizerVisualizer(run_dir)
        visualizer.plot_training_curves(metrics_history)
        visualizer.plot_reconstruction_samples(original, reconstructed)
        visualizer.plot_codebook_usage(code_usage)
        visualizer.save_all_figures()
    """
    
    def __init__(
        self,
        run_dir: Union[str, Path],
        figsize_base: tuple = (10, 6),
        dpi: int = 150,
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        """
        Initialize visualizer.
        
        Args:
            run_dir: Directory to save figures
            figsize_base: Base figure size
            dpi: Figure resolution
            style: Matplotlib style
        """
        self.run_dir = Path(run_dir)
        self.figures_dir = self.run_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize_base = figsize_base
        self.dpi = dpi
        
        # Try to set style, fall back if not available
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass  # Use default
        
        # Color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'tertiary': '#F18F01',
            'success': '#2ECC71',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'light': '#95A5A6',
        }
        
        # Store generated figure paths
        self.generated_figures: List[Path] = []
    
    # =========================================================================
    # Training Curves
    # =========================================================================
    
    def plot_training_curves(
        self,
        metrics_history: List[Dict[str, float]],
        save: bool = True,
        filename: str = "training_curves.png"
    ) -> Optional[plt.Figure]:
        """
        Plot training curves: loss, perplexity, utilization.
        
        Args:
            metrics_history: List of epoch metrics dicts
            save: Whether to save the figure
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        if not metrics_history:
            print("[Viz] Warning: Empty metrics history")
            return None
        
        epochs = [m.get('epoch', i+1) for i, m in enumerate(metrics_history)]
        
        fig = plt.figure(figsize=(self.figsize_base[0] * 1.5, self.figsize_base[1] * 1.2))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Reconstruction Loss
        ax1 = fig.add_subplot(gs[0, 0])
        train_recon = [m.get('reconstruction_mse', np.nan) for m in metrics_history]
        val_recon = [m.get('val_reconstruction_mse', np.nan) for m in metrics_history]
        
        ax1.plot(epochs, train_recon, label='Train', color=self.colors['primary'], linewidth=2)
        ax1.plot(epochs, val_recon, label='Val', color=self.colors['secondary'], linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.set_title('Reconstruction Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Perplexity
        ax2 = fig.add_subplot(gs[0, 1])
        perplexity = [m.get('perplexity', np.nan) for m in metrics_history]
        
        ax2.plot(epochs, perplexity, color=self.colors['tertiary'], linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Codebook Perplexity')
        ax2.grid(True, alpha=0.3)
        
        # Add reference line at max possible perplexity
        if perplexity and not np.isnan(perplexity[0]):
            ax2.axhline(y=max(perplexity) * 1.1, color=self.colors['light'], 
                       linestyle=':', alpha=0.5, label='Target: High')
        
        # 3. Code Utilization
        ax3 = fig.add_subplot(gs[0, 2])
        utilization = [m.get('code_utilization', np.nan) for m in metrics_history]
        
        ax3.plot(epochs, utilization, color=self.colors['success'], linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Utilization')
        ax3.set_title('Code Utilization')
        ax3.set_ylim([0, 1.05])
        ax3.axhline(y=0.9, color=self.colors['light'], linestyle=':', alpha=0.5, label='Target: 90%')
        ax3.grid(True, alpha=0.3)
        
        # 4. Dead Codes
        ax4 = fig.add_subplot(gs[1, 0])
        dead_codes = [m.get('dead_codes', np.nan) for m in metrics_history]
        
        ax4.plot(epochs, dead_codes, color=self.colors['danger'], linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Dead Codes')
        ax4.set_title('Dead Codes Count')
        ax4.grid(True, alpha=0.3)
        
        # 5. Learning Rate
        ax5 = fig.add_subplot(gs[1, 1])
        lr = [m.get('lr', np.nan) for m in metrics_history]
        
        ax5.plot(epochs, lr, color=self.colors['primary'], linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('Learning Rate Schedule')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
        
        # 6. Spectral Loss (if available)
        ax6 = fig.add_subplot(gs[1, 2])
        spectral = [m.get('spectral_mse', np.nan) for m in metrics_history]
        
        if not all(np.isnan(spectral)):
            ax6.plot(epochs, spectral, color=self.colors['secondary'], linewidth=2)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Spectral MSE')
            ax6.set_title('Spectral Loss')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Spectral loss\nnot enabled', 
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=12, color=self.colors['light'])
            ax6.set_title('Spectral Loss')
            ax6.axis('off')
        
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / filename
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.generated_figures.append(path)
            plt.close(fig)
            print(f"[Viz] Saved: {path.name}")
            return None
        else:
            return fig
    
    # =========================================================================
    # Reconstruction Samples
    # =========================================================================
    
    def plot_reconstruction_samples(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        n_samples: int = 4,
        sample_indices: Optional[List[int]] = None,
        fs: float = 200.0,
        save: bool = True,
        filename: str = "reconstruction_samples.png"
    ) -> Optional[plt.Figure]:
        """
        Plot original vs reconstructed signal comparison.
        
        Args:
            original: Original signals [B, T] or [B, C, T]
            reconstructed: Reconstructed signals [B, T] or [B, C, T]
            n_samples: Number of samples to plot
            sample_indices: Specific indices to plot
            fs: Sampling frequency (for time axis)
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        # Convert to numpy
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        # Handle dimensions
        if original.ndim == 3:
            original = original[:, 0, :]  # Take first channel
            reconstructed = reconstructed[:, 0, :]
        
        # Select samples
        n_available = min(n_samples, len(original))
        if sample_indices is None:
            sample_indices = np.linspace(0, len(original)-1, n_available, dtype=int)
        
        fig, axes = plt.subplots(n_available, 2, figsize=(14, 3 * n_available))
        if n_available == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(sample_indices[:n_available]):
            orig = original[idx]
            recon = reconstructed[idx]
            t = np.arange(len(orig)) / fs
            
            # Time domain
            ax_time = axes[i, 0]
            ax_time.plot(t, orig, label='Original', color=self.colors['primary'], 
                        alpha=0.8, linewidth=1.5)
            ax_time.plot(t, recon, label='Reconstructed', color=self.colors['secondary'],
                        alpha=0.8, linewidth=1.5, linestyle='--')
            ax_time.set_xlabel('Time (s)')
            ax_time.set_ylabel('Amplitude')
            ax_time.set_title(f'Sample {idx}: Time Domain')
            ax_time.legend(loc='upper right')
            ax_time.grid(True, alpha=0.3)
            
            # Error
            mse = np.mean((orig - recon) ** 2)
            ax_time.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax_time.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residual
            ax_res = axes[i, 1]
            residual = orig - recon
            ax_res.plot(t, residual, color=self.colors['danger'], alpha=0.7, linewidth=1)
            ax_res.fill_between(t, residual, alpha=0.3, color=self.colors['danger'])
            ax_res.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_res.set_xlabel('Time (s)')
            ax_res.set_ylabel('Residual')
            ax_res.set_title(f'Sample {idx}: Reconstruction Error')
            ax_res.grid(True, alpha=0.3)
        
        fig.suptitle('Reconstruction Quality', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / filename
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.generated_figures.append(path)
            plt.close(fig)
            print(f"[Viz] Saved: {path.name}")
            return None
        else:
            return fig
    
    # =========================================================================
    # Spectral Comparison
    # =========================================================================
    
    def plot_spectral_comparison(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        fs: float = 200.0,
        n_samples: int = 100,
        freq_range: Optional[tuple] = None,
        save: bool = True,
        filename: str = "spectral_comparison.png"
    ) -> Optional[plt.Figure]:
        """
        Plot power spectral density comparison.
        
        Args:
            original: Original signals
            reconstructed: Reconstructed signals
            fs: Sampling frequency
            n_samples: Number of samples to average PSD over
            freq_range: (fmin, fmax) to display
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        # Convert to numpy
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        # Handle dimensions
        if original.ndim == 3:
            original = original[:, 0, :]
            reconstructed = reconstructed[:, 0, :]
        
        # Limit samples
        n_use = min(n_samples, len(original))
        original = original[:n_use]
        reconstructed = reconstructed[:n_use]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Compute average PSD
        def compute_avg_psd(signals):
            """Compute Welch PSD averaged over samples."""
            from scipy.signal import welch
            psds = []
            for sig in signals:
                freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
                psds.append(psd)
            return freqs, np.mean(psds, axis=0), np.std(psds, axis=0)
        
        try:
            freq, psd_orig, std_orig = compute_avg_psd(original)
            _, psd_recon, std_recon = compute_avg_psd(reconstructed)
        except ImportError:
            # Fallback to simple FFT
            def simple_psd(signals):
                psds = []
                for sig in signals:
                    fft = np.fft.rfft(sig)
                    psd = np.abs(fft) ** 2
                    psds.append(psd)
                freqs = np.fft.rfftfreq(len(signals[0]), 1/fs)
                return freqs, np.mean(psds, axis=0), np.std(psds, axis=0)
            freq, psd_orig, std_orig = simple_psd(original)
            _, psd_recon, std_recon = simple_psd(reconstructed)
        
        # Apply frequency range
        if freq_range:
            mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
            freq = freq[mask]
            psd_orig, std_orig = psd_orig[mask], std_orig[mask]
            psd_recon, std_recon = psd_recon[mask], std_recon[mask]
        
        # 1. PSD Comparison (log scale)
        ax1 = axes[0]
        ax1.semilogy(freq, psd_orig, label='Original', color=self.colors['primary'], 
                    linewidth=2, alpha=0.9)
        ax1.fill_between(freq, psd_orig - std_orig, psd_orig + std_orig, 
                        alpha=0.2, color=self.colors['primary'])
        ax1.semilogy(freq, psd_recon, label='Reconstructed', color=self.colors['secondary'],
                    linewidth=2, alpha=0.9, linestyle='--')
        ax1.fill_between(freq, psd_recon - std_recon, psd_recon + std_recon,
                        alpha=0.2, color=self.colors['secondary'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.set_title('PSD Comparison (Log Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PSD Ratio
        ax2 = axes[1]
        ratio = (psd_recon + 1e-10) / (psd_orig + 1e-10)
        ax2.plot(freq, ratio, color=self.colors['tertiary'], linewidth=2)
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Perfect reconstruction')
        ax2.fill_between(freq, np.ones_like(ratio) * 0.8, np.ones_like(ratio) * 1.2,
                        alpha=0.2, color=self.colors['success'], label='±20% tolerance')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD Ratio (Recon/Orig)')
        ax2.set_title('Spectral Fidelity Ratio')
        ax2.set_ylim([0, 2])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Spectral Error
        ax3 = axes[2]
        spectral_error = np.abs(psd_orig - psd_recon)
        ax3.semilogy(freq, spectral_error, color=self.colors['danger'], linewidth=2)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('|PSD Error|')
        ax3.set_title('Absolute Spectral Error')
        ax3.grid(True, alpha=0.3)
        
        # Annotate mean error
        mean_error = np.mean(spectral_error)
        ax3.axhline(y=mean_error, color=self.colors['light'], linestyle='--', 
                   label=f'Mean: {mean_error:.4f}')
        ax3.legend()
        
        fig.suptitle('Spectral Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / filename
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.generated_figures.append(path)
            plt.close(fig)
            print(f"[Viz] Saved: {path.name}")
            return None
        else:
            return fig
    
    # =========================================================================
    # Codebook Usage
    # =========================================================================
    
    def plot_codebook_usage(
        self,
        indices: torch.Tensor,
        codebook_size: int,
        save: bool = True,
        filename: str = "codebook_usage.png"
    ) -> Optional[plt.Figure]:
        """
        Plot codebook usage histogram and statistics.
        
        Args:
            indices: Token indices [B, T] or flattened
            codebook_size: Total codebook size
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        # Flatten and count
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu()
            flat = indices.flatten()
            usage = torch.bincount(flat, minlength=codebook_size).numpy()
        else:
            flat = indices.flatten()
            usage = np.bincount(flat, minlength=codebook_size)
        
        # Sort by usage
        sorted_indices = np.argsort(usage)[::-1]
        sorted_usage = usage[sorted_indices]
        
        # Statistics
        total_tokens = usage.sum()
        used_codes = (usage > 0).sum()
        dead_codes = codebook_size - used_codes
        utilization = used_codes / codebook_size
        
        # Entropy and perplexity
        probs = usage / (total_tokens + 1e-10)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        perplexity = np.exp(entropy)
        
        # Gini coefficient
        n = len(usage)
        sorted_asc = np.sort(usage)
        cumsum = np.cumsum(sorted_asc)
        gini = (n + 1 - 2 * np.sum(cumsum) / (cumsum[-1] + 1e-10)) / n
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Usage histogram (sorted)
        ax1 = axes[0, 0]
        x = np.arange(codebook_size)
        ax1.bar(x, sorted_usage, color=self.colors['primary'], alpha=0.7, width=1.0)
        ax1.set_xlabel('Code Index (sorted by usage)')
        ax1.set_ylabel('Usage Count')
        ax1.set_title('Codebook Usage (Sorted)')
        ax1.set_xlim([0, codebook_size])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Mark dead codes region
        if dead_codes > 0:
            ax1.axvspan(used_codes, codebook_size, alpha=0.3, color=self.colors['danger'],
                       label=f'Dead codes ({dead_codes})')
            ax1.legend()
        
        # 2. Cumulative usage (Lorenz curve)
        ax2 = axes[0, 1]
        cumulative = np.cumsum(sorted_usage) / (total_tokens + 1e-10)
        ax2.plot(x / codebook_size * 100, cumulative * 100, 
                color=self.colors['primary'], linewidth=2, label='Actual')
        ax2.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Perfect uniform')
        ax2.fill_between(x / codebook_size * 100, cumulative * 100, 
                        alpha=0.3, color=self.colors['primary'])
        ax2.set_xlabel('% of Codes (sorted)')
        ax2.set_ylabel('% of Tokens')
        ax2.set_title(f'Cumulative Usage (Gini={gini:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 100])
        ax2.set_ylim([0, 100])
        
        # 3. Usage distribution (log histogram)
        ax3 = axes[1, 0]
        nonzero_usage = usage[usage > 0]
        if len(nonzero_usage) > 0:
            bins = np.logspace(0, np.log10(max(nonzero_usage) + 1), 30)
            ax3.hist(nonzero_usage, bins=bins, color=self.colors['secondary'], 
                    alpha=0.7, edgecolor='white')
            ax3.set_xscale('log')
            ax3.set_xlabel('Usage Count (log scale)')
            ax3.set_ylabel('Number of Codes')
            ax3.set_title('Usage Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = (
            f"Codebook Statistics\n"
            f"{'='*30}\n\n"
            f"Total codebook size:  {codebook_size:,}\n"
            f"Active codes:         {used_codes:,} ({utilization*100:.1f}%)\n"
            f"Dead codes:           {dead_codes:,} ({100-utilization*100:.1f}%)\n\n"
            f"Total tokens:         {total_tokens:,}\n"
            f"Perplexity:           {perplexity:.1f}\n"
            f"Max perplexity:       {codebook_size}\n"
            f"Normalized pplx:      {perplexity/codebook_size*100:.1f}%\n\n"
            f"Gini coefficient:     {gini:.3f}\n"
            f"(0=uniform, 1=concentrated)\n\n"
            f"Most used code:       {sorted_indices[0]} ({sorted_usage[0]:,} times)\n"
            f"Median usage:         {np.median(nonzero_usage):.0f}\n" if len(nonzero_usage) > 0 else ""
        )
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=12, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        fig.suptitle('Codebook Health Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / filename
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.generated_figures.append(path)
            plt.close(fig)
            print(f"[Viz] Saved: {path.name}")
            return None
        else:
            return fig
    
    # =========================================================================
    # Token Embedding Visualization
    # =========================================================================
    
    def plot_token_embeddings(
        self,
        embeddings: torch.Tensor,
        usage: Optional[torch.Tensor] = None,
        method: str = 'tsne',
        perplexity: int = 30,
        save: bool = True,
        filename: str = "token_embeddings.png"
    ) -> Optional[plt.Figure]:
        """
        Visualize codebook embeddings using dimensionality reduction.
        
        Args:
            embeddings: Codebook vectors [K, D]
            usage: Usage count for each code [K] (for coloring)
            method: 'tsne' or 'pca'
            perplexity: t-SNE perplexity
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        if not HAS_SKLEARN:
            print("[Viz] Warning: sklearn not available, skipping embedding visualization")
            return None
        
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if usage is not None and isinstance(usage, torch.Tensor):
            usage = usage.detach().cpu().numpy()
        
        K, D = embeddings.shape
        
        # Reduce dimensionality
        if method == 'tsne' and K > perplexity * 3:
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        else:
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embeddings)
            method = 'pca'  # Update for title
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Colored by usage
        ax1 = axes[0]
        if usage is not None:
            # Log scale for better visibility
            log_usage = np.log1p(usage)
            scatter = ax1.scatter(reduced[:, 0], reduced[:, 1], 
                                 c=log_usage, cmap='viridis',
                                 s=30, alpha=0.7)
            plt.colorbar(scatter, ax=ax1, label='Log(1 + usage)')
        else:
            ax1.scatter(reduced[:, 0], reduced[:, 1], 
                       color=self.colors['primary'], s=30, alpha=0.7)
        
        ax1.set_xlabel(f'{method.upper()} 1')
        ax1.set_ylabel(f'{method.upper()} 2')
        ax1.set_title(f'Codebook Embeddings ({method.upper()}) - Usage Colored')
        ax1.grid(True, alpha=0.3)
        
        # 2. Highlight dead vs active codes
        ax2 = axes[1]
        if usage is not None:
            active_mask = usage > 0
            dead_mask = usage == 0
            
            ax2.scatter(reduced[active_mask, 0], reduced[active_mask, 1],
                       color=self.colors['success'], s=30, alpha=0.7, label='Active')
            if dead_mask.any():
                ax2.scatter(reduced[dead_mask, 0], reduced[dead_mask, 1],
                           color=self.colors['danger'], s=30, alpha=0.7, label='Dead')
            ax2.legend()
        else:
            ax2.scatter(reduced[:, 0], reduced[:, 1],
                       color=self.colors['primary'], s=30, alpha=0.7)
        
        ax2.set_xlabel(f'{method.upper()} 1')
        ax2.set_ylabel(f'{method.upper()} 2')
        ax2.set_title('Active vs Dead Codes')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Codebook Embedding Space (K={K}, D={D})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / filename
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.generated_figures.append(path)
            plt.close(fig)
            print(f"[Viz] Saved: {path.name}")
            return None
        else:
            return fig
    
    # =========================================================================
    # Summary Report
    # =========================================================================
    
    def generate_summary_figure(
        self,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        save: bool = True,
        filename: str = "experiment_summary.png"
    ) -> Optional[plt.Figure]:
        """
        Generate a summary figure with key metrics.
        
        Args:
            metrics: Final metrics dict
            config: Experiment config
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Build summary text
        model_type = config.get('model', {}).get('type', 'unknown')
        exp_name = config.get('experiment', {}).get('name', 'unknown')
        modality = config.get('data', {}).get('modality', 'unknown')
        
        if model_type == 'fsq':
            levels = config.get('model', {}).get('quantizer', {}).get('levels', [])
            codebook_info = f"Levels: {levels}\nCodebook: {np.prod(levels) if levels else 'N/A'}"
        else:
            cb_size = config.get('model', {}).get('quantizer', {}).get('codebook_size', 'N/A')
            codebook_info = f"Codebook size: {cb_size}"
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    EXPERIMENT SUMMARY                        ║
╠══════════════════════════════════════════════════════════════╣
║  Experiment:  {exp_name:<47}║
║  Model:       {model_type.upper():<47}║
║  Modality:    {modality:<47}║
╠══════════════════════════════════════════════════════════════╣
║  {codebook_info:<60}║
╠══════════════════════════════════════════════════════════════╣
║                      FINAL METRICS                           ║
╠══════════════════════════════════════════════════════════════╣
"""
        
        for key, value in metrics.items():
            if isinstance(value, float):
                line = f"║  {key:<25} {value:>32.4f} ║"
            else:
                line = f"║  {key:<25} {str(value):>32} ║"
            summary += line + "\n"
        
        summary += "╚══════════════════════════════════════════════════════════════╝"
        
        ax.text(0.02, 0.98, summary, transform=ax.transAxes,
               fontsize=10, family='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
        
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / filename
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.generated_figures.append(path)
            plt.close(fig)
            print(f"[Viz] Saved: {path.name}")
            return None
        else:
            return fig
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_generated_figures(self) -> List[Path]:
        """Get list of all generated figure paths."""
        return self.generated_figures
    
    def save_figure_manifest(self, filename: str = "figures_manifest.json"):
        """Save manifest of generated figures."""
        manifest = {
            'run_dir': str(self.run_dir),
            'figures': [str(p.name) for p in self.generated_figures],
            'count': len(self.generated_figures)
        }
        
        path = self.figures_dir / filename
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"[Viz] Manifest saved: {path.name}")


# =============================================================================
# Convenience function
# =============================================================================

def visualize_tokenizer_run(
    run_dir: Union[str, Path],
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    metrics_history: List[Dict[str, float]],
    config: Dict[str, Any],
    final_metrics: Dict[str, Any],
    device: torch.device = torch.device('cpu')
) -> List[Path]:
    """
    Generate all visualizations for a tokenizer run.
    
    Args:
        run_dir: Run directory
        model: Trained model
        test_loader: Test data loader
        metrics_history: Training history
        config: Experiment config
        final_metrics: Final test metrics
        device: Torch device
    
    Returns:
        List of generated figure paths
    """
    visualizer = TokenizerVisualizer(run_dir)
    
    # 1. Training curves
    visualizer.plot_training_curves(metrics_history)
    
    # 2. Get test samples for visualization
    model.eval()
    all_originals = []
    all_reconstructed = []
    all_indices = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            outputs = model(x)
            all_originals.append(x.cpu())
            all_reconstructed.append(outputs['x_rec'].cpu())
            all_indices.append(outputs['indices'].cpu())
            
            if len(all_originals) * x.shape[0] >= 200:
                break
    
    original = torch.cat(all_originals, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)
    indices = torch.cat(all_indices, dim=0)
    
    # 3. Reconstruction samples
    visualizer.plot_reconstruction_samples(original, reconstructed)
    
    # 4. Spectral comparison
    fs = config.get('data', {}).get('sampling_rate', 200)
    visualizer.plot_spectral_comparison(original, reconstructed, fs=fs)
    
    # 5. Codebook usage
    codebook_size = model.get_codebook_size()
    visualizer.plot_codebook_usage(indices, codebook_size)
    
    # 6. Token embeddings (if available)
    if hasattr(model, 'get_codebook_embeddings'):
        embeddings = model.get_codebook_embeddings()
        if embeddings is not None:
            # Compute usage for coloring
            flat_indices = indices.flatten()
            usage = torch.bincount(flat_indices, minlength=codebook_size)
            visualizer.plot_token_embeddings(embeddings, usage)
    
    # 7. Summary figure
    visualizer.generate_summary_figure(final_metrics, config)
    
    # Save manifest
    visualizer.save_figure_manifest()
    
    return visualizer.get_generated_figures()
