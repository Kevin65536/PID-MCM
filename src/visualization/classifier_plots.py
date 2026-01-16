"""
Visualization tools for classifier training experiments.

Generates and saves:
1. Training curves (loss, accuracy)
2. Confusion matrix
3. ROC curve and AUC
4. Per-class performance breakdown
5. Learning dynamics analysis
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
import matplotlib.patches as mpatches

# Optional: sklearn for metrics
try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ClassifierVisualizer:
    """
    Visualization toolkit for classifier experiments.
    
    Usage:
        visualizer = ClassifierVisualizer(run_dir)
        visualizer.plot_training_curves(metrics_history)
        visualizer.plot_confusion_matrix(conf_matrix, class_names)
        visualizer.plot_roc_curve(labels, probs)
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
                pass
        
        # Color palette (consistent with tokenizer_plots.py)
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
        Plot training curves: loss, accuracy over epochs.
        
        Args:
            metrics_history: List of epoch metrics dicts with 'loss', 'accuracy', etc.
            save: Whether to save the figure
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        if not metrics_history:
            print("[Viz] Warning: Empty metrics history")
            return None
        
        epochs = [m.get('epoch', i+1) for i, m in enumerate(metrics_history)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Loss curves
        ax1 = axes[0, 0]
        train_loss = [m.get('loss', np.nan) for m in metrics_history]
        val_loss = [m.get('val_loss', np.nan) for m in metrics_history]
        
        ax1.plot(epochs, train_loss, label='Train', color=self.colors['primary'], linewidth=2)
        ax1.plot(epochs, val_loss, label='Val', color=self.colors['secondary'], linewidth=2, linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax2 = axes[0, 1]
        train_acc = [m.get('accuracy', np.nan) * 100 for m in metrics_history]
        val_acc = [m.get('val_accuracy', np.nan) * 100 for m in metrics_history]
        
        ax2.plot(epochs, train_acc, label='Train', color=self.colors['primary'], linewidth=2)
        ax2.plot(epochs, val_acc, label='Val', color=self.colors['secondary'], linewidth=2, linestyle='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # Add reference lines
        ax2.axhline(y=50, color=self.colors['light'], linestyle=':', alpha=0.5, label='Chance')
        
        # 3. Learning rate
        ax3 = axes[1, 0]
        lr = [m.get('lr', np.nan) for m in metrics_history]
        
        if not all(np.isnan(lr)):
            ax3.plot(epochs, lr, color=self.colors['tertiary'], linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Learning rate\nnot logged', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color=self.colors['light'])
            ax3.axis('off')
        
        # 4. Training vs Validation gap (overfitting indicator)
        ax4 = axes[1, 1]
        gap = [t - v for t, v in zip(train_acc, val_acc)]
        
        ax4.plot(epochs, gap, color=self.colors['warning'], linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.fill_between(epochs, gap, alpha=0.3, color=self.colors['warning'])
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Train - Val Accuracy (%)')
        ax4.set_title('Generalization Gap')
        ax4.grid(True, alpha=0.3)
        
        # Add annotation
        if len(gap) > 0:
            final_gap = gap[-1]
            ax4.annotate(f'Final gap: {final_gap:.1f}%',
                        xy=(epochs[-1], gap[-1]),
                        xytext=(-80, 20),
                        textcoords='offset points',
                        fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='gray'))
        
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
    # Confusion Matrix
    # =========================================================================
    
    def plot_confusion_matrix(
        self,
        conf_matrix: Union[np.ndarray, List[List[int]]],
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        save: bool = True,
        filename: str = "confusion_matrix.png"
    ) -> Optional[plt.Figure]:
        """
        Plot confusion matrix heatmap.
        
        Args:
            conf_matrix: Confusion matrix [n_classes, n_classes]
            class_names: Names for each class
            normalize: Whether to show percentages
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        cm = np.array(conf_matrix)
        n_classes = cm.shape[0]
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Raw counts
        ax1 = axes[0]
        im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.set_title('Confusion Matrix (Counts)')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax1.set_xticks(np.arange(n_classes))
        ax1.set_yticks(np.arange(n_classes))
        ax1.set_xticklabels(class_names)
        ax1.set_yticklabels(class_names)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Normalized (percentages)
        ax2 = axes[1]
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax2.set_title('Confusion Matrix (Normalized)')
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                ax2.text(j, i, format(cm_norm[i, j], '.1%'),
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        
        ax2.set_xticks(np.arange(n_classes))
        ax2.set_yticks(np.arange(n_classes))
        ax2.set_xticklabels(class_names)
        ax2.set_yticklabels(class_names)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        fig.suptitle('Classification Results', fontsize=14, fontweight='bold')
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
    # ROC Curve
    # =========================================================================
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        class_names: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "roc_curve.png"
    ) -> Optional[plt.Figure]:
        """
        Plot ROC curve and AUC.
        
        Args:
            y_true: True labels
            y_probs: Predicted probabilities [n_samples, n_classes]
            class_names: Names for each class
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        if not HAS_SKLEARN:
            print("[Viz] Warning: sklearn not available, skipping ROC curve")
            return None
        
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        
        # Handle binary classification
        if y_probs.ndim == 1 or y_probs.shape[1] == 2:
            # Binary case
            if y_probs.ndim == 2:
                y_score = y_probs[:, 1]  # Probability of positive class
            else:
                y_score = y_probs
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # ROC curve
            ax1 = axes[0]
            ax1.plot(fpr, tpr, color=self.colors['primary'], linewidth=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], color=self.colors['light'], linestyle='--', linewidth=1)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # Fill area under curve
            ax1.fill_between(fpr, tpr, alpha=0.3, color=self.colors['primary'])
            
            # Precision-Recall curve
            ax2 = axes[1]
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            
            ax2.plot(recall, precision, color=self.colors['secondary'], linewidth=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)
            ax2.fill_between(recall, precision, alpha=0.3, color=self.colors['secondary'])
            
        else:
            # Multi-class case (one-vs-rest)
            n_classes = y_probs.shape[1]
            if class_names is None:
                class_names = [f'Class {i}' for i in range(n_classes)]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['tertiary'], self.colors['success']]
            
            for i in range(n_classes):
                y_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                       label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], color=self.colors['light'], linestyle='--', linewidth=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves (One-vs-Rest)')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Classification Performance', fontsize=14, fontweight='bold')
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
    # Per-Subject Performance
    # =========================================================================
    
    def plot_per_subject_performance(
        self,
        subject_accuracies: Dict[int, float],
        save: bool = True,
        filename: str = "per_subject_performance.png"
    ) -> Optional[plt.Figure]:
        """
        Plot per-subject classification accuracy.
        
        Args:
            subject_accuracies: Dict mapping subject_id to accuracy
            save: Whether to save
            filename: Output filename
        
        Returns:
            Figure object if save=False
        """
        subjects = sorted(subject_accuracies.keys())
        accuracies = [subject_accuracies[s] * 100 for s in subjects]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Bar chart
        ax1 = axes[0]
        bars = ax1.bar(range(len(subjects)), accuracies, color=self.colors['primary'], alpha=0.7)
        
        # Color bars by performance
        mean_acc = np.mean(accuracies)
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc >= mean_acc + 10:
                bar.set_color(self.colors['success'])
            elif acc < mean_acc - 10:
                bar.set_color(self.colors['danger'])
        
        ax1.axhline(y=mean_acc, color='black', linestyle='--', linewidth=1, label=f'Mean: {mean_acc:.1f}%')
        ax1.axhline(y=50, color=self.colors['light'], linestyle=':', alpha=0.5, label='Chance')
        
        ax1.set_xticks(range(len(subjects)))
        ax1.set_xticklabels([f'S{s}' for s in subjects], rotation=45)
        ax1.set_xlabel('Subject')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Per-Subject Classification Accuracy')
        ax1.set_ylim([0, 100])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Distribution
        ax2 = axes[1]
        ax2.hist(accuracies, bins=10, color=self.colors['secondary'], alpha=0.7, edgecolor='white')
        ax2.axvline(x=mean_acc, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.1f}%')
        ax2.axvline(x=np.median(accuracies), color=self.colors['tertiary'], linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(accuracies):.1f}%')
        
        ax2.set_xlabel('Accuracy (%)')
        ax2.set_ylabel('Number of Subjects')
        ax2.set_title('Accuracy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (
            f"Statistics:\n"
            f"Mean: {mean_acc:.1f}%\n"
            f"Std: {np.std(accuracies):.1f}%\n"
            f"Min: {np.min(accuracies):.1f}%\n"
            f"Max: {np.max(accuracies):.1f}%"
        )
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle('Subject-Level Analysis', fontsize=14, fontweight='bold')
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
        exp_name = config.get('experiment', {}).get('name', 'unknown')
        modality = config.get('data', {}).get('modality', 'unknown')
        classifier_type = config.get('classifier', {}).get('type', 'unknown')
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                 CLASSIFICATION SUMMARY                       ║
╠══════════════════════════════════════════════════════════════╣
║  Experiment:  {exp_name:<47}║
║  Classifier:  {classifier_type:<47}║
║  Modality:    {modality:<47}║
╠══════════════════════════════════════════════════════════════╣
║                      TEST METRICS                            ║
╠══════════════════════════════════════════════════════════════╣
"""
        
        metric_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 
                      'best_val_accuracy']
        
        for key in metric_keys:
            value = metrics.get(key)
            if value is not None:
                if isinstance(value, float):
                    line = f"║  {key:<25} {value*100:>29.2f}% ║"
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

def visualize_classifier_run(
    run_dir: Union[str, Path],
    metrics_history: List[Dict[str, float]],
    final_metrics: Dict[str, Any],
    config: Dict[str, Any],
    y_true: Optional[np.ndarray] = None,
    y_probs: Optional[np.ndarray] = None,
    subject_accuracies: Optional[Dict[int, float]] = None,
    class_names: Optional[List[str]] = None,
) -> List[Path]:
    """
    Generate all visualizations for a classifier run.
    
    Args:
        run_dir: Run directory
        metrics_history: Training history
        final_metrics: Final test metrics
        config: Experiment config
        y_true: True labels for ROC curve
        y_probs: Predicted probabilities for ROC curve
        subject_accuracies: Per-subject accuracy dict
        class_names: Names for each class
    
    Returns:
        List of generated figure paths
    """
    visualizer = ClassifierVisualizer(run_dir)
    
    # 1. Training curves
    visualizer.plot_training_curves(metrics_history)
    
    # 2. Confusion matrix (if available)
    conf_matrix = final_metrics.get('test_confusion_matrix')
    if conf_matrix is not None:
        visualizer.plot_confusion_matrix(conf_matrix, class_names=class_names)
    
    # 3. ROC curve (if labels and probs provided)
    if y_true is not None and y_probs is not None:
        visualizer.plot_roc_curve(y_true, y_probs, class_names=class_names)
    
    # 4. Per-subject performance (if provided)
    if subject_accuracies is not None:
        visualizer.plot_per_subject_performance(subject_accuracies)
    
    # 5. Summary figure
    visualizer.generate_summary_figure(final_metrics, config)
    
    # Save manifest
    visualizer.save_figure_manifest()
    
    return visualizer.get_generated_figures()
