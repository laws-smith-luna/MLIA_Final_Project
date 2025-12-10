"""
Optimized Experiment Runner
============================

This script helps you find the best temporal processing approach
and then runs comprehensive experiments with the optimal settings.
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime
import json

from synthetic_data import create_data_loaders as create_synthetic_loaders
from experiment_runner import ExperimentRunner, full_comparison_experiments
from results_visualization import ResultsVisualizer


def temporal_mode_comparison(data_path, results_dir='temporal_comparison'):
    """
    Quick comparison of different temporal processing modes
    
    This runs Sequencer for just 10 epochs on each temporal mode
    to find which one works best for your data.
    """
    print("\n" + "="*80)
    print("TEMPORAL MODE COMPARISON - Finding Optimal Approach")
    print("="*80)
    
    from enhanced_data_loading import load_real_data
    from sequencer_model import Sequencer2D_S
    from training_pipeline import Trainer
    
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Temporal modes to test
    modes = [
        'single_frame',      # Baseline (what you have now)
        'average',           # Average all frames
        'peak_frame',        # Peak contraction frame
        'multi_frame',       # 5 key frames (RECOMMENDED)
        'temporal_stats'     # Statistical features
    ]
    
    results = []
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Testing Mode: {mode}")
        print(f"{'='*80}")
        
        # Load data with this temporal mode
        train_loader, val_loader, num_channels = load_real_data(
            data_path,
            temporal_mode=mode,
            batch_size=8
        )
        
        # Create model with appropriate number of channels
        model = Sequencer2D_S(in_channels=num_channels, num_outputs=126)
        
        # Quick training (10 epochs)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-3,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_dir=str(results_dir / mode)
        )
        
        trainer.train(num_epochs=10, save_every=10)
        
        # Record results
        result = {
            'mode': mode,
            'num_channels': num_channels,
            'best_val_loss': trainer.best_val_loss,
            'final_val_loss': trainer.val_losses[-1],
            'final_train_loss': trainer.train_losses[-1]
        }
        results.append(result)
        
        print(f"\n{mode} Results:")
        print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"  Final Val Loss: {result['final_val_loss']:.4f}")
    
    # Save comparison
    comparison_file = results_dir / 'temporal_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best mode
    best_result = min(results, key=lambda x: x['best_val_loss'])
    
    print("\n" + "="*80)
    print("TEMPORAL MODE COMPARISON RESULTS")
    print("="*80)
    
    for result in sorted(results, key=lambda x: x['best_val_loss']):
        marker = " ← BEST" if result == best_result else ""
        print(f"{result['mode']:20s} Val Loss: {result['best_val_loss']:.4f}{marker}")
    
    print(f"\n✓ Best temporal mode: {best_result['mode']}")
    print(f"  Validation Loss: {best_result['best_val_loss']:.4f}")
    print(f"  Input Channels: {best_result['num_channels']}")
    print(f"\nUse this mode for your final experiments!")
    print("="*80)
    
    return best_result['mode']


def run_full_experiments_with_best_mode(
    data_path,
    temporal_mode,
    results_dir='final_experiments'
):
    """
    Run complete experiments with the best temporal mode
    
    Tests Sequencer against all baselines with optimized settings
    """
    print("\n" + "="*80)
    print("RUNNING FULL EXPERIMENTS WITH OPTIMIZED SETTINGS")
    print("="*80)
    print(f"Temporal Mode: {temporal_mode}")
    print(f"Results Directory: {results_dir}")
    print("="*80)
    
    from enhanced_data_loading import load_real_data
    
    # Load data with best temporal mode
    train_loader, val_loader, num_channels = load_real_data(
        data_path,
        temporal_mode=temporal_mode,
        batch_size=8
    )
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(results_dir) / f"{temporal_mode}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize experiment runner
    runner = ExperimentRunner(results_dir=str(results_dir))
    
    # Define experiments
    from experiment_runner import ExperimentConfig
    
    experiments = [
    ExperimentConfig(
        'linear',
        num_epochs=50,
        experiment_name='baseline_linear',
        model_kwargs={'in_channels': num_channels, 'image_size': 80}
    ),
    ExperimentConfig(
        'mlp',
        num_epochs=50,
        experiment_name='baseline_mlp',
        model_kwargs={'in_channels': num_channels, 'image_size': 80}
    ),
    ExperimentConfig(
        'cnn',
        num_epochs=50,
        experiment_name='baseline_cnn',
        model_kwargs={'in_channels': num_channels, 'image_size': 80}
    ),
    ExperimentConfig(
        'lstm',
        num_epochs=50,
        experiment_name='baseline_lstm',
        model_kwargs={'in_channels': num_channels, 'image_size': 80}
    ),
    
    # Sequencer experiments (no image_size needed)
    ExperimentConfig(
        'sequencer',
        num_epochs=100,
        learning_rate=1e-3,
        experiment_name=f'sequencer_{temporal_mode}',
        model_kwargs={'in_channels': num_channels}
    ),
    ExperimentConfig(
        'sequencer',
        num_epochs=100,
        learning_rate=5e-4,
        experiment_name=f'sequencer_{temporal_mode}_lr5e4',
        model_kwargs={'in_channels': num_channels}
    ),
]
    
    # Run all experiments
    print(f"\nRunning {len(experiments)} experiments...")
    all_results = []
    
    for i, config in enumerate(experiments, 1):
        print(f"\n>>> Experiment {i}/{len(experiments)}")
        result = runner.run_experiment(config, train_loader, val_loader)
        all_results.append(result)
    
    # Save comparison table
    runner.save_comparison_table()
    
    # Generate visualizations
    print("\nGenerating figures...")
    viz = ResultsVisualizer(results_dir=str(results_dir))
    experiment_names = [exp.experiment_name for exp in experiments]
    viz.generate_all_figures(experiment_names=experiment_names)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results_dir, all_results, temporal_mode)
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - model_comparison.csv")
    print("  - figures/model_comparison.png")
    print("  - figures/training_curves.png")
    print("  - figures/model_comparison.tex")
    print("  - summary_report.txt")
    print("="*80)
    
    return results_dir


def generate_summary_report(results_dir, all_results, temporal_mode):
    """Generate comprehensive summary report"""
    report_path = results_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CARDIAC TOS PREDICTION - FINAL RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Temporal Processing Mode: {temporal_mode}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(all_results)}\n\n")
        
        # Sort by validation loss
        sorted_results = sorted(all_results, key=lambda x: x['best_val_loss'])
        
        f.write("="*80 + "\n")
        f.write("RANKING BY VALIDATION LOSS\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i}. {result['model_name'].upper():20s} "
                   f"Val Loss: {result['best_val_loss']:.4f}  "
                   f"MAE: {result['val_mae']:.4f}  "
                   f"R²: {result['val_r2']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BEST MODEL DETAILS\n")
        f.write("="*80 + "\n\n")
        
        best = sorted_results[0]
        f.write(f"Model: {best['model_name']}\n")
        f.write(f"Experiment: {best['experiment_name']}\n")
        f.write(f"Validation Loss: {best['best_val_loss']:.4f}\n")
        f.write(f"MAE: {best['val_mae']:.4f}\n")
        f.write(f"R²: {best['val_r2']:.4f}\n")
        f.write(f"Parameters: {best['num_parameters']:,}\n")
        f.write(f"Training Time: {best['training_time_seconds']:.1f}s\n\n")
        
        f.write("="*80 + "\n")
        f.write("PAPER WRITING CHECKLIST\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Copy LaTeX table from figures/model_comparison.tex\n")
        f.write("2. Include figures/model_comparison.png in results section\n")
        f.write("3. Include figures/training_curves.png to show convergence\n")
        f.write(f"4. Mention temporal processing mode: {temporal_mode}\n")
        f.write(f"5. Report best model performance: {best['val_mae']:.4f} MAE\n")
        f.write("6. Compare Sequencer vs baselines\n")
        f.write("7. Discuss why temporal information helps (if applicable)\n")
    
    print(f"Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized experiment runner for cardiac TOS prediction'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='2023-11-15-cine-myo-masks-and-TOS.npy',
        help='Path to data file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['compare_temporal', 'full_experiments', 'both'],
        default='both',
        help='What to run: compare_temporal, full_experiments, or both'
    )
    
    parser.add_argument(
        '--temporal-mode',
        type=str,
        default=None,
        help='Specify temporal mode directly (skips comparison)'
    )
    
    args = parser.parse_args()
    
    # Step 1: Find best temporal mode (if not specified)
    if args.mode in ['compare_temporal', 'both'] and args.temporal_mode is None:
        print("\nSTEP 1: Finding optimal temporal processing mode...")
        best_mode = temporal_mode_comparison(args.data_path)
    else:
        best_mode = args.temporal_mode or 'multi_frame'
        print(f"\nUsing specified temporal mode: {best_mode}")
    
    # Step 2: Run full experiments with best mode
    if args.mode in ['full_experiments', 'both']:
        print("\nSTEP 2: Running full experiments with optimal settings...")
        results_dir = run_full_experiments_with_best_mode(
            args.data_path,
            best_mode
        )
        print(f"\n✓ All experiments complete! Results in: {results_dir}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OPTIMIZED CARDIAC TOS PREDICTION                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script will:
1. Test different temporal processing modes (10 epochs each)
2. Identify the best approach for your data
3. Run full experiments (50-100 epochs) with optimal settings
4. Generate publication-ready results

USAGE:

1. Quick test all temporal modes then run full experiments:
   python optimized_experiment_runner.py --mode both

2. Just compare temporal modes:
   python optimized_experiment_runner.py --mode compare_temporal

3. Skip comparison and run with specific mode:
   python optimized_experiment_runner.py --mode full_experiments --temporal-mode multi_frame

RECOMMENDED: Run mode 'both' to let the system find the best approach!

Press Ctrl+C to cancel, or press Enter to start...
    """)
    input()
    main()