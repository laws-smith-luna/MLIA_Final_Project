"""
Master Script - Run Complete Experimental Suite
================================================

Purpose: One-click solution to run all experiments and generate
paper-ready results.

What this does:
1. Loads data (synthetic or real)
2. Runs all baseline models + Sequencer
3. Saves all results
4. Generates figures and tables
5. Creates summary report

Usage:
    python run_all_experiments.py --mode quick    # 5 epochs, test run
    python run_all_experiments.py --mode full     # Full experiments
    python run_all_experiments.py --real-data     # Use real data
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime

from synthetic_data import create_data_loaders as create_synthetic_loaders
from experiment_runner import (
    ExperimentRunner, 
    quick_test_experiments,
    full_comparison_experiments,
    hyperparameter_search_experiments
)
from results_visualization import ResultsVisualizer


def load_data(use_real_data, data_path, batch_size=8):
    """
    Load either synthetic or real data
    
    Returns:
        train_loader, val_loader
    """
    if use_real_data:
        print("📊 Loading REAL data...")
        from main_training import load_real_data
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        train_loader, val_loader = load_real_data(data_path, train_split=0.8)
    else:
        print("📊 Loading SYNTHETIC data...")
        train_loader, val_loader = create_synthetic_loaders(
            train_size=80,
            val_size=20,
            batch_size=batch_size
        )
    
    return train_loader, val_loader


def run_experiment_suite(
    mode='quick',
    use_real_data=False,
    data_path='2023-11-15-cine-myo-masks-and-TOS.npy',
    results_dir='experiment_results'
):
    """
    Run complete experiment suite
    
    Args:
        mode: 'quick' (5 epochs), 'full' (50-100 epochs), or 'hyperparam'
        use_real_data: Whether to use real data
        data_path: Path to real data file
        results_dir: Where to save results
    """
    print("\n" + "=" * 80)
    print("CARDIAC TOS PREDICTION - COMPLETE EXPERIMENTAL SUITE")
    print("=" * 80)
    print(f"Mode: {mode.upper()}")
    print(f"Data: {'REAL' if use_real_data else 'SYNTHETIC'}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Results Directory: {results_dir}")
    print("=" * 80)
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(results_dir) / f"{mode}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Results will be saved to: {results_dir}")
    
    # Load data
    print("\n" + "=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)
    train_loader, val_loader = load_data(use_real_data, data_path)
    
    # Select experiments based on mode
    print("\n" + "=" * 80)
    print("STEP 2: Selecting Experiments")
    print("=" * 80)
    
    if mode == 'quick':
        experiments = quick_test_experiments()
        print("Running QUICK TEST (5 epochs per model)")
    elif mode == 'full':
        experiments = full_comparison_experiments()
        print("Running FULL COMPARISON (50-100 epochs)")
    elif mode == 'hyperparam':
        experiments = hyperparameter_search_experiments()
        print("Running HYPERPARAMETER SEARCH (Sequencer only)")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(f"Total experiments: {len(experiments)}")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.experiment_name}")
    
    # Run experiments
    print("\n" + "=" * 80)
    print("STEP 3: Running Experiments")
    print("=" * 80)
    
    runner = ExperimentRunner(results_dir=str(results_dir))
    
    all_results = []
    for i, config in enumerate(experiments, 1):
        print(f"\n>>> Experiment {i}/{len(experiments)}")
        results = runner.run_experiment(config, train_loader, val_loader)
        all_results.append(results)
    
    # Save comparison table
    print("\n" + "=" * 80)
    print("STEP 4: Generating Comparison Table")
    print("=" * 80)
    runner.save_comparison_table()
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("STEP 5: Generating Figures and Tables")
    print("=" * 80)
    
    viz = ResultsVisualizer(results_dir=str(results_dir))
    experiment_names = [exp.experiment_name for exp in experiments]
    viz.generate_all_figures(experiment_names=experiment_names)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("STEP 6: Generating Summary Report")
    print("=" * 80)
    generate_summary_report(results_dir, all_results, mode)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {results_dir}")
    print("\n📊 Generated files:")
    print(f"  - model_comparison.csv         (All metrics)")
    print(f"  - figures/model_comparison.png (Bar plots)")
    print(f"  - figures/training_curves.png  (Loss curves)")
    print(f"  - figures/model_comparison.tex (LaTeX table)")
    print(f"  - summary_report.txt           (Text summary)")
    print("\n✅ Ready for paper writing!")
    print("=" * 80)
    
    return results_dir, all_results


def generate_summary_report(results_dir, all_results, mode):
    """
    Generate a text summary report
    """
    report_path = results_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CARDIAC TOS PREDICTION - EXPERIMENTAL RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment Mode: {mode}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(all_results)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by best validation loss
        sorted_results = sorted(all_results, key=lambda x: x['best_val_loss'])
        
        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i}. {result['model_name'].upper()}\n")
            f.write(f"   Best Val Loss: {result['best_val_loss']:.4f}\n")
            f.write(f"   Val MAE: {result['val_mae']:.4f}\n")
            f.write(f"   Val R²: {result['val_r2']:.4f}\n")
            f.write(f"   Parameters: {result['num_parameters']:,}\n")
            f.write(f"   Training Time: {result['training_time_seconds']:.2f}s\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("BEST MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        best = sorted_results[0]
        f.write(f"Model: {best['model_name']}\n")
        f.write(f"Validation Loss: {best['best_val_loss']:.4f}\n")
        f.write(f"MAE: {best['val_mae']:.4f}\n")
        f.write(f"R²: {best['val_r2']:.4f}\n")
        f.write(f"Model Path: {best['best_model_path']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS FOR PAPER\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Use figures/model_comparison.png for visual comparison\n")
        f.write("2. Include figures/training_curves.png to show convergence\n")
        f.write("3. Copy LaTeX table from figures/model_comparison.tex\n")
        f.write("4. Highlight that ")
        
        if best['model_name'] == 'sequencer':
            f.write("Sequencer achieved best performance\n")
        else:
            f.write(f"{best['model_name']} outperformed Sequencer\n")
            f.write("   Consider discussing why Sequencer underperformed\n")
        
        f.write("\n5. Discuss computational cost vs. performance tradeoff\n")
        f.write("6. Include error analysis and failure cases\n")
    
    print(f"📄 Summary report saved to {report_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run complete experimental suite for cardiac TOS prediction'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'full', 'hyperparam'],
        help='Experiment mode: quick (5 epochs), full (50-100 epochs), hyperparam (search)'
    )
    
    parser.add_argument(
        '--real-data',
        action='store_true',
        help='Use real data instead of synthetic'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='2023-11-15-cine-myo-masks-and-TOS.npy',
        help='Path to real data file'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='experiment_results',
        help='Base directory for results'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Run experiment suite
    results_dir, all_results = run_experiment_suite(
        mode=args.mode,
        use_real_data=args.real_data,
        data_path=args.data_path,
        results_dir=args.results_dir
    )
    
    return results_dir, all_results


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  MASTER EXPERIMENT RUNNER                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

This script will:
✓ Run all baseline models (Linear, MLP, CNN, LSTM)
✓ Run Sequencer model
✓ Compare all results
✓ Generate publication-ready figures and tables
✓ Create summary report

USAGE EXAMPLES:

1. Quick test (5 epochs, synthetic data):
   python run_all_experiments.py --mode quick

2. Full run with synthetic data (for testing pipeline):
   python run_all_experiments.py --mode full

3. Full run with REAL data (for final paper results):
   python run_all_experiments.py --mode full --real-data

4. Hyperparameter search:
   python run_all_experiments.py --mode hyperparam --real-data

Press Ctrl+C to cancel, or press Enter to start...
    """)
    
    input()
    main()