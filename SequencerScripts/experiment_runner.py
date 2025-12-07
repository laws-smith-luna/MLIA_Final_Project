"""
Experiment Runner for Systematic Model Comparison
==================================================

Purpose: Run multiple experiments with different configurations and
automatically track results for paper.

Features:
- Define experiments in config files
- Run all baselines + Sequencer automatically
- Track all metrics and hyperparameters
- Save results to CSV for analysis
"""

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

from baseline_models import get_model, count_parameters
from training_pipeline import Trainer


class ExperimentConfig:
    """
    Configuration for a single experiment
    
    Stores all hyperparameters and settings for reproducibility
    """
    
    def __init__(
        self,
        model_name,
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=50,
        weight_decay=0.05,
        experiment_name=None,
        model_kwargs=None
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.model_kwargs = model_kwargs or {}
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{model_name}_{timestamp}"
        else:
            self.experiment_name = experiment_name
    
    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'experiment_name': self.experiment_name,
            'model_kwargs': self.model_kwargs
        }
    
    def __repr__(self):
        return f"ExperimentConfig({self.experiment_name})"


class ExperimentRunner:
    """
    Runs experiments and tracks results
    
    Purpose: Systematically run multiple experiments and save all
    results for comparison and paper writing.
    """
    
    def __init__(
        self,
        results_dir='experiment_results',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.device = device
        self.results = []
    
    def run_experiment(self, config, train_loader, val_loader):
        """
        Run a single experiment with given configuration
        
        Args:
            config: ExperimentConfig instance
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            results: Dictionary with all metrics
        """
        print("\n" + "=" * 80)
        print(f"Running Experiment: {config.experiment_name}")
        print("=" * 80)
        print(f"Model: {config.model_name}")
        print(f"Learning Rate: {config.learning_rate}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Epochs: {config.num_epochs}")
        print("-" * 80)
        
        # Create model
        model = get_model(config.model_name, **config.model_kwargs)
        num_params = count_parameters(model)
        print(f"Model Parameters: {num_params:,}")
        
        # Create experiment-specific checkpoint directory
        checkpoint_dir = self.results_dir / config.experiment_name / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config.learning_rate,
            device=self.device,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        # Train
        start_time = time.time()
        trainer.train(num_epochs=config.num_epochs, save_every=10)
        training_time = time.time() - start_time
        
        # Get final metrics
        final_train_loss = trainer.train_losses[-1]
        final_val_loss = trainer.val_losses[-1]
        best_val_loss = trainer.best_val_loss
        
        # Calculate additional metrics on validation set
        val_metrics = self._calculate_metrics(model, val_loader)
        
        # Compile results
        results = {
            # Experiment info
            'experiment_name': config.experiment_name,
            'model_name': config.model_name,
            'timestamp': datetime.now().isoformat(),
            
            # Model info
            'num_parameters': num_params,
            
            # Hyperparameters
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'weight_decay': config.weight_decay,
            
            # Training metrics
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'training_time_seconds': training_time,
            
            # Validation metrics
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
            
            # Paths
            'checkpoint_dir': str(checkpoint_dir),
            'best_model_path': str(checkpoint_dir / 'best_model.pth')
        }
        
        # Save results
        self._save_experiment_results(config, results)
        self.results.append(results)
        
        print("\n" + "=" * 80)
        print(f"Experiment Complete: {config.experiment_name}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        print("=" * 80)
        
        return results
    
    def _calculate_metrics(self, model, data_loader):
        """
        Calculate comprehensive metrics on a dataset
        
        Returns:
            metrics: Dict with MSE, MAE, R²
        """
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for masks, targets in data_loader:
                masks = masks.to(self.device)
                predictions = model(masks).cpu().numpy()
                targets = targets.numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Concatenate all batches (handles different batch sizes)
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        mse = ((predictions - targets) ** 2).mean()
        mae = abs(predictions - targets).mean()
        
        # R² score
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def _save_experiment_results(self, config, results):
        """Save experiment configuration and results"""
        exp_dir = self.results_dir / config.experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Save results
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def run_all_baselines(
        self,
        train_loader,
        val_loader,
        learning_rate=1e-3,
        num_epochs=50
    ):
        """
        Run all baseline models + Sequencer with same settings
        
        Purpose: Fair comparison across all models
        """
        models_to_test = ['linear', 'mlp', 'cnn', 'lstm', 'sequencer']
        
        print("\n" + "=" * 80)
        print("Running Complete Baseline Comparison")
        print("=" * 80)
        print(f"Models: {', '.join(models_to_test)}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print("=" * 80)
        
        all_results = []
        
        for model_name in models_to_test:
            config = ExperimentConfig(
                model_name=model_name,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                experiment_name=f"{model_name}_baseline"
            )
            
            results = self.run_experiment(config, train_loader, val_loader)
            all_results.append(results)
        
        # Save comparison table
        self.save_comparison_table(all_results)
        
        return all_results
    
    def save_comparison_table(self, results=None):
        """
        Save comparison table as CSV
        
        Creates a paper-ready table comparing all models
        """
        if results is None:
            results = self.results
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Select key columns for comparison
        comparison_cols = [
            'model_name',
            'num_parameters',
            'best_val_loss',
            'val_mse',
            'val_mae',
            'val_r2',
            'training_time_seconds'
        ]
        
        df_comparison = df[comparison_cols].copy()
        
        # Sort by validation loss (best first)
        df_comparison = df_comparison.sort_values('best_val_loss')
        
        # Save
        csv_path = self.results_dir / 'model_comparison.csv'
        df_comparison.to_csv(csv_path, index=False)
        
        print(f"\n📊 Comparison table saved to: {csv_path}")
        print("\nModel Comparison:")
        print(df_comparison.to_string(index=False))
        
        return df_comparison


# === Pre-defined experiment suites ===

def quick_test_experiments():
    """
    Quick test: Run all models for 5 epochs
    
    Use this to verify everything works before long runs
    """
    return [
        ExperimentConfig('linear', num_epochs=5, experiment_name='quick_linear'),
        ExperimentConfig('mlp', num_epochs=5, experiment_name='quick_mlp'),
        ExperimentConfig('cnn', num_epochs=5, experiment_name='quick_cnn'),
        ExperimentConfig('lstm', num_epochs=5, experiment_name='quick_lstm'),
        ExperimentConfig('sequencer', num_epochs=5, experiment_name='quick_sequencer'),
    ]


def full_comparison_experiments():
    """
    Full comparison: All models trained properly
    
    Use this when you have real data and want final results
    """
    return [
        ExperimentConfig('linear', num_epochs=50, experiment_name='full_linear'),
        ExperimentConfig('mlp', num_epochs=50, experiment_name='full_mlp'),
        ExperimentConfig('cnn', num_epochs=50, experiment_name='full_cnn'),
        ExperimentConfig('lstm', num_epochs=50, experiment_name='full_lstm'),
        ExperimentConfig('sequencer', num_epochs=100, experiment_name='full_sequencer'),
    ]


def hyperparameter_search_experiments():
    """
    Hyperparameter search for Sequencer
    
    Test different learning rates and batch sizes
    """
    experiments = []
    
    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [8, 16]
    
    for lr in learning_rates:
        for bs in batch_sizes:
            exp_name = f"sequencer_lr{lr}_bs{bs}"
            experiments.append(
                ExperimentConfig(
                    'sequencer',
                    learning_rate=lr,
                    batch_size=bs,
                    num_epochs=50,
                    experiment_name=exp_name
                )
            )
    
    return experiments


# === Example usage ===
if __name__ == "__main__":
    print("""
Experiment Runner Ready!

Usage Examples:

1. Quick test (5 epochs, all models):
   from experiment_runner import ExperimentRunner, quick_test_experiments
   from synthetic_data import create_data_loaders
   
   train_loader, val_loader = create_data_loaders()
   runner = ExperimentRunner()
   
   for config in quick_test_experiments():
       runner.run_experiment(config, train_loader, val_loader)
   
   runner.save_comparison_table()

2. Full comparison (for paper):
   experiments = full_comparison_experiments()
   for config in experiments:
       runner.run_experiment(config, train_loader, val_loader)

3. Run all baselines automatically:
   runner.run_all_baselines(train_loader, val_loader, num_epochs=50)

    """)