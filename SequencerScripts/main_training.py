"""
Main Training Script - Cardiac TOS Prediction
==============================================

Purpose: Complete end-to-end training pipeline with easy switching
between synthetic and real data.

Usage:
    python main_training.py              # Use synthetic data
    python main_training.py --real-data  # Use real data
"""

import torch
import argparse
import numpy as np
from pathlib import Path

from sequencer_model import Sequencer2D_S
from synthetic_data import create_data_loaders as create_synthetic_loaders
from training_pipeline import Trainer, test_model_prediction

# Try to import enhanced data loading, fall back to old method if not available
try:
    from enhanced_data_loading import load_real_data
    ENHANCED_LOADING_AVAILABLE = True
except ImportError:
    ENHANCED_LOADING_AVAILABLE = False
    print("Note: Enhanced data loading not available, using basic loading")


def load_real_data_basic(data_path, train_split=0.8):
    """
    Basic data loading (original version)
    Falls back to this if enhanced_data_loading.py not available
    """
    from torch.utils.data import Dataset, DataLoader
    
    print(f"Loading real data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    print(f"Total samples: {len(data)}")
    
    class RealCardiacDataset(Dataset):
        def __init__(self, data_samples):
            self.data = data_samples
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            mask_volume = sample['cine_lv_myo_masks_cropped']
            tos_curve = sample['TOS']
            
            # Use middle frame
            middle_frame = mask_volume.shape[2] // 2
            mask_frame = mask_volume[:, :, middle_frame]
            
            mask_tensor = torch.from_numpy(mask_frame).float().unsqueeze(0)
            tos_tensor = torch.from_numpy(tos_curve).float()
            
            return mask_tensor, tos_tensor
    
    num_train = int(len(data) * train_split)
    train_data = data[:num_train]
    val_data = data[num_train:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    train_dataset = RealCardiacDataset(train_data)
    val_dataset = RealCardiacDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, 1  # Return num_channels=1


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sequencer for Cardiac TOS Prediction')
    
    # Data mode
    parser.add_argument('--real-data', action='store_true',
                       help='Use real data instead of synthetic')
    parser.add_argument('--data-path', type=str, 
                       default='2023-11-15-cine-myo-masks-and-TOS.npy',
                       help='Path to real data .npy file')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data for training')
    
    # NEW: Temporal processing mode
    parser.add_argument('--temporal-mode', type=str, 
                       default='multi_frame',
                       choices=['single_frame', 'average', 'max_projection', 
                               'peak_frame', 'multi_frame', 'temporal_stats'],
                       help='How to process temporal dimension (requires enhanced_data_loading.py)')
    
    # Synthetic data parameters
    parser.add_argument('--train-size', type=int, default=80,
                       help='Number of training samples (synthetic only)')
    parser.add_argument('--val-size', type=int, default=20,
                       help='Number of validation samples (synthetic only)')
    
    # Common parameters
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    use_synthetic = not args.real_data
    
    print("=" * 80)
    print("Cardiac TOS Prediction with Sequencer2D-S")
    print("=" * 80)
    
    # Show data mode
    data_mode = "SYNTHETIC" if use_synthetic else "REAL"
    print(f"\n{'*' * 80}")
    print(f"{'*' * 30}  DATA MODE: {data_mode}  {'*' * (30 - len(data_mode))}")
    print(f"{'*' * 80}")
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create data loaders
    print(f"\nCreating data loaders...")
    
    if use_synthetic:
        print("📊 Using SYNTHETIC data for testing")
        print(f"  Train samples: {args.train_size}")
        print(f"  Val samples: {args.val_size}")
        
        train_loader, val_loader = create_synthetic_loaders(
            train_size=args.train_size,
            val_size=args.val_size,
            batch_size=args.batch_size
        )
        num_channels = 1
        
    else:
        print("📊 Using REAL data")
        print(f"  Data path: {args.data_path}")
        print(f"  Train split: {args.train_split}")
        
        if not Path(args.data_path).exists():
            raise FileNotFoundError(
                f"Real data file not found: {args.data_path}\n"
                f"Please check the path or get the file from Canvas/teammates"
            )
        
        # Use enhanced loading if available, otherwise basic
        if ENHANCED_LOADING_AVAILABLE:
            print(f"  Temporal mode: {args.temporal_mode}")
            train_loader, val_loader, num_channels = load_real_data(
                data_path=args.data_path,
                train_split=args.train_split,
                temporal_mode=args.temporal_mode,
                batch_size=args.batch_size
            )
        else:
            print("  Temporal mode: single_frame (basic loading)")
            train_loader, val_loader, num_channels = load_real_data_basic(
                data_path=args.data_path,
                train_split=args.train_split
            )
    
    print(f"  Batch size: {args.batch_size}")
    print(f"  Input channels: {num_channels}")
    
    # Initialize model
    print(f"\nInitializing Sequencer2D-S model...")
    print(f"  Input channels: {num_channels}")
    print(f"  Output dimension: 126 (TOS curve)")
    
    model = Sequencer2D_S(
        in_channels=num_channels,
        num_outputs=126
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print(f"\nSetting up trainer...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint (optional)
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 80)
    
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every
    )
    
    # Test predictions
    print("\nTesting predictions on validation set...")
    test_model_prediction(
        model=model,
        val_loader=val_loader,
        device=device,
        num_samples=3
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pth")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if '--help' in sys.argv or '-h' in sys.argv:
        parser = argparse.ArgumentParser(description='Train Sequencer for Cardiac TOS Prediction')
        parser.print_help()
        print("""

EXAMPLE USAGE:

1. Quick test with synthetic data:
   python main_training.py --num-epochs 5

2. Train on real data with single frame (basic):
   python main_training.py --real-data --num-epochs 50

3. Train on real data with multi-frame (RECOMMENDED):
   python main_training.py --real-data --temporal-mode multi_frame --num-epochs 100

4. Train with different temporal modes:
   python main_training.py --real-data --temporal-mode average --num-epochs 50
   python main_training.py --real-data --temporal-mode peak_frame --num-epochs 50
   python main_training.py --real-data --temporal-mode temporal_stats --num-epochs 50

5. Resume from checkpoint:
   python main_training.py --real-data --resume checkpoints/best_model.pth

NOTE: For optimal results, use optimized_experiment_runner.py which automatically
      tests all temporal modes and finds the best one!
        """)
    else:
        main()