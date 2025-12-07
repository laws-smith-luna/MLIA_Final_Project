"""
Results Visualization for Paper
================================

Purpose: Generate publication-quality figures and tables from
experiment results.

Outputs:
- Comparison tables (LaTeX format)
- Training curves
- Prediction vs. ground truth plots
- Error analysis figures
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class ResultsVisualizer:
    """
    Generate all figures and tables for the paper
    """
    
    def __init__(self, results_dir='experiment_results'):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
    
    def load_experiment_results(self, experiment_name):
        """Load results from a specific experiment"""
        exp_dir = self.results_dir / experiment_name
        
        with open(exp_dir / 'results.json', 'r') as f:
            results = json.load(f)
        
        # Load training history if available
        history_path = exp_dir / 'checkpoints' / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                results['training_history'] = json.load(f)
        
        return results
    
    def plot_training_curves(self, experiment_names, save_path=None):
        """
        Plot training and validation loss curves
        
        Args:
            experiment_names: List of experiment names to plot
            save_path: Where to save figure (None = show)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for exp_name in experiment_names:
            results = self.load_experiment_results(exp_name)
            
            if 'training_history' not in results:
                print(f"Warning: No training history for {exp_name}")
                continue
            
            history = results['training_history']
            epochs = range(1, len(history['train_losses']) + 1)
            
            # Plot train and val losses
            ax.plot(epochs, history['train_losses'], 
                   label=f"{results['model_name']} (Train)", 
                   linestyle='--', alpha=0.7)
            ax.plot(epochs, history['val_losses'], 
                   label=f"{results['model_name']} (Val)", 
                   linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Training and Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved training curves to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_model_comparison(self, csv_path=None, save_path=None):
        """
        Bar plot comparing model performance
        
        Args:
            csv_path: Path to model_comparison.csv
            save_path: Where to save figure
        """
        if csv_path is None:
            csv_path = self.results_dir / 'model_comparison.csv'
        
        df = pd.read_csv(csv_path)
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Validation MAE
        ax = axes[0, 0]
        df_sorted = df.sort_values('val_mae')
        ax.barh(df_sorted['model_name'], df_sorted['val_mae'])
        ax.set_xlabel('Mean Absolute Error')
        ax.set_title('Validation MAE (Lower is Better)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Validation R²
        ax = axes[0, 1]
        df_sorted = df.sort_values('val_r2', ascending=False)
        ax.barh(df_sorted['model_name'], df_sorted['val_r2'])
        ax.set_xlabel('R² Score')
        ax.set_title('R² Score (Higher is Better)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 3. Training Time
        ax = axes[1, 0]
        df_sorted = df.sort_values('training_time_seconds')
        ax.barh(df_sorted['model_name'], df_sorted['training_time_seconds'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Training Time', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Model Size
        ax = axes[1, 1]
        df_sorted = df.sort_values('num_parameters')
        ax.barh(df_sorted['model_name'], df_sorted['num_parameters'] / 1e6)
        ax.set_xlabel('Parameters (Millions)')
        ax.set_title('Model Size', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved comparison plot to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_predictions(self, model_path, val_loader, num_samples=5, save_path=None):
        """
        Plot predicted vs. actual TOS curves
        
        Args:
            model_path: Path to trained model checkpoint
            val_loader: Validation data loader
            num_samples: Number of samples to plot
            save_path: Where to save figure
        """
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine model type from path
        if 'sequencer' in str(model_path):
            from sequencer_model import Sequencer2D_S
            model = Sequencer2D_S()
        else:
            print("Model type detection not implemented for baselines yet")
            return
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get predictions
        masks, targets = next(iter(val_loader))
        with torch.no_grad():
            predictions = model(masks)
        
        # Plot
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            ax = axes[i]
            
            pred = predictions[i].numpy()
            true = targets[i].numpy()
            
            x = np.arange(126)
            ax.plot(x, true, 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
            ax.plot(x, pred, 'r--', label='Prediction', linewidth=2)
            
            # Calculate error
            mse = ((pred - true) ** 2).mean()
            mae = abs(pred - true).mean()
            
            ax.set_xlabel('Time Point')
            ax.set_ylabel('TOS Value')
            ax.set_title(f'Sample {i+1} - MSE: {mse:.3f}, MAE: {mae:.3f}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved prediction plots to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def generate_latex_table(self, csv_path=None, save_path=None):
        """
        Generate LaTeX table for paper
        
        Args:
            csv_path: Path to model_comparison.csv
            save_path: Where to save .tex file
        """
        if csv_path is None:
            csv_path = self.results_dir / 'model_comparison.csv'
        
        df = pd.read_csv(csv_path)
        
        # Select and rename columns for paper
        df_paper = df[[
            'model_name', 
            'num_parameters',
            'val_mse', 
            'val_mae', 
            'val_r2'
        ]].copy()
        
        df_paper.columns = [
            'Model',
            'Parameters',
            'MSE ↓',
            'MAE ↓',
            'R² ↑'
        ]
        
        # Format numbers
        df_paper['Parameters'] = df_paper['Parameters'].apply(lambda x: f"{x/1e6:.2f}M")
        df_paper['MSE ↓'] = df_paper['MSE ↓'].apply(lambda x: f"{x:.4f}")
        df_paper['MAE ↓'] = df_paper['MAE ↓'].apply(lambda x: f"{x:.4f}")
        df_paper['R² ↑'] = df_paper['R² ↑'].apply(lambda x: f"{x:.4f}")
        
        # Sort by MAE
        df_paper = df_paper.sort_values('MAE ↓')
        
        # Generate LaTeX
        latex = df_paper.to_latex(
            index=False,
            caption='Comparison of different models for cardiac TOS prediction. '
                   'Lower MSE and MAE are better; higher R² is better.',
            label='tab:model_comparison',
            position='htbp'
        )
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(latex)
            print(f"📄 Saved LaTeX table to {save_path}")
        
        print("\nLaTeX Table:")
        print(latex)
        
        return latex
    
    def generate_all_figures(self, experiment_names=None):
        """
        Generate all figures for the paper at once
        
        Args:
            experiment_names: List of experiments to include
        """
        print("\n" + "=" * 70)
        print("Generating All Paper Figures")
        print("=" * 70)
        
        # 1. Model comparison bar plots
        print("\n1. Generating model comparison plots...")
        self.plot_model_comparison(
            save_path=self.figures_dir / 'model_comparison.png'
        )
        
        # 2. Training curves (if experiment names provided)
        if experiment_names:
            print("\n2. Generating training curves...")
            self.plot_training_curves(
                experiment_names,
                save_path=self.figures_dir / 'training_curves.png'
            )
        
        # 3. LaTeX table
        print("\n3. Generating LaTeX table...")
        self.generate_latex_table(
            save_path=self.figures_dir / 'model_comparison.tex'
        )
        
        print("\n" + "=" * 70)
        print(f"All figures saved to: {self.figures_dir}")
        print("=" * 70)
        print("\nFiles generated:")
        print("  - model_comparison.png    (Model performance bars)")
        print("  - training_curves.png     (Loss over epochs)")
        print("  - model_comparison.tex    (LaTeX table for paper)")


# === Convenience function ===
def quick_visualize(results_dir='experiment_results'):
    """
    Quick function to generate all visualizations
    
    Usage:
        from results_visualization import quick_visualize
        quick_visualize()
    """
    viz = ResultsVisualizer(results_dir)
    
    # Check what experiments exist
    exp_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir() and d.name != 'figures']
    exp_names = [d.name for d in exp_dirs]
    
    print(f"Found {len(exp_names)} experiments:")
    for name in exp_names:
        print(f"  - {name}")
    
    # Generate all figures
    viz.generate_all_figures(experiment_names=exp_names)
    
    return viz


# === Example usage ===
if __name__ == "__main__":
    print("""
Results Visualization Ready!

After running experiments, use:

    from results_visualization import quick_visualize
    quick_visualize()

This will generate:
- Bar plots comparing all models
- Training curves showing convergence
- LaTeX table ready for your paper

All saved to experiment_results/figures/
    """)