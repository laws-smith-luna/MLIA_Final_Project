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

# ============================================================================
# CONFIGURATION: Switch between synthetic and real data
# ============================================================================
USE_SYNTHETIC_DATA = True  # Set to False to use real data

# Real data paths (update these when you have real data)
REAL_DATA_PATH = '2023-11-15-cine-myo-masks-and-TOS.npy'


def load_real_data(data_path, train_split=0.8):
    """
    Load real cardiac data from .npy file
    
    Args:
        data_path: Path to the .npy file
        train_split: Fraction of data to use for training
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    from torch.utils.data import Dataset, DataLoader
    
    print(f"Loading real data from {data_path}...")
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    
    print(f"Total samples in dataset: {len(data)}")
    
    class RealCardiacDataset(Dataset):
        """Dataset for real cardiac masks and TOS curves"""
        
        def __init__(self, data_samples):
            self.data = data_samples
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            
            # Get mask volume and TOS
            mask_volume = sample['cine_lv_myo_masks_cropped']  # (H, W, n_frames)
            tos_curve = sample['TOS']  # (126,)
            
            # For now, use middle frame (can be enhanced later)
            # Or average across frames, or use max projection
            middle_frame = mask_volume.shape[2] // 2
            mask_frame = mask_volume[:, :, middle_frame]
            
            # Convert to tensors
            mask_tensor = torch.from_numpy(mask_frame).float().unsqueeze(0)  # (1, H, W)
            tos_tensor = torch.from_numpy(tos_curve).float()  # (126,)
            
            return mask_tensor, tos_tensor
    
    # Split into train/val
    num_samples = len(data)
    num_train = int(num_samples * train_split)
    
    train_data = data[:num_train]
    val_data = data[num_train:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = RealCardiacDataset(train_data)
    val_dataset = RealCardiacDataset(val_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def parse_args():
    """
    Parse command line arguments
    
    Purpose: Allow easy configuration without editing code
    """
    parser = argparse.ArgumentParser(description='Train Sequencer for Cardiac TOS Prediction')
    
    # Data mode
    parser.add_argument('--real-data', action='store_true',
                       help='Use real data instead of synthetic')
    parser.add_argument('--data-path', type=str, default=REAL_DATA_PATH,
                       help='Path to real data .npy file')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data for training (real data only)')
    
    # Synthetic data parameters (only used if not --real-data)
    parser.add_argument('--train-size', type=int, default=80,
                       help='Number of training samples (synthetic only)')
    parser.add_argument('--val-size', type=int, default=20,
                       help='Number of validation samples (synthetic only)')
    
    # Common parameters
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--in-channels', type=int, default=1,
                       help='Input channels (1 for grayscale)')
    parser.add_argument('--num-outputs', type=int, default=126,
                       help='Number of TOS predictions')
    
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
    """
    Main training function
    
    Steps:
    1. Parse arguments
    2. Set up device
    3. Create data loaders (synthetic or real)
    4. Initialize model
    5. Create trainer
    6. Train model
    7. Test predictions
    """
    # Parse command line arguments
    args = parse_args()
    
    # Override USE_SYNTHETIC_DATA with command line flag
    use_synthetic = not args.real_data
    
    print("=" * 80)
    print("Cardiac TOS Prediction with Sequencer2D-S")
    print("=" * 80)
    
    # Show data mode
    data_mode = "SYNTHETIC" if use_synthetic else "REAL"
    print(f"\n{'*' * 80}")
    print(f"{'*' * 30}  DATA MODE: {data_mode}  {'*' * (30 - len(data_mode))}")
    print(f"{'*' * 80}")
    
    # === 1. Device Setup ===
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # === 2. Create Data Loaders ===
    print(f"\nCreating data loaders...")
    
    if use_synthetic:
        print("📊 Using SYNTHETIC data for testing")
        print(f"  Train samples: {args.train_size}")
        print(f"  Val samples: {args.val_size}")
        print(f"  Batch size: {args.batch_size}")
        
        train_loader, val_loader = create_synthetic_loaders(
            train_size=args.train_size,
            val_size=args.val_size,
            batch_size=args.batch_size
        )
    else:
        print("📊 Using REAL data")
        print(f"  Data path: {args.data_path}")
        print(f"  Train split: {args.train_split}")
        print(f"  Batch size: {args.batch_size}")
        
        # Check if file exists
        if not Path(args.data_path).exists():
            raise FileNotFoundError(
                f"Real data file not found: {args.data_path}\n"
                f"Please update --data-path or use --real-data flag"
            )
        
        train_loader, val_loader = load_real_data(
            data_path=args.data_path,
            train_split=args.train_split
        )
    
    # === 3. Initialize Model ===
    print(f"\nInitializing Sequencer2D-S model...")
    print(f"  Input channels: {args.in_channels}")
    print(f"  Output dimension: {args.num_outputs}")
    
    model = Sequencer2D_S(
        in_channels=args.in_channels,
        num_outputs=args.num_outputs
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # === 4. Create Trainer ===
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
    
    # === 5. Resume from Checkpoint (Optional) ===
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # === 6. Train Model ===
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 80)
    
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every
    )
    
    # === 7. Test Predictions ===
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


def print_usage_examples():
    """Print helpful usage examples"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                            QUICK START                                     ║
╚════════════════════════════════════════════════════════════════════════════╝

🔹 TEST WITH SYNTHETIC DATA (default):
  $ python main_training.py

🔹 TRAIN WITH REAL DATA:
  $ python main_training.py --real-data

🔹 SPECIFY REAL DATA PATH:
  $ python main_training.py --real-data --data-path path/to/your/data.npy

╔════════════════════════════════════════════════════════════════════════════╗
║                         EXAMPLE COMMANDS                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Quick test with synthetic data (5 epochs):
  $ python main_training.py --num-epochs 5 --train-size 20 --val-size 5

Train on real data for 100 epochs:
  $ python main_training.py --real-data --num-epochs 100

Use larger batch size (if you have GPU memory):
  $ python main_training.py --batch-size 16

Train with different learning rate:
  $ python main_training.py --learning-rate 5e-4

Resume from checkpoint:
  $ python main_training.py --resume checkpoints/latest_checkpoint.pth

Force CPU training:
  $ python main_training.py --device cpu

╔════════════════════════════════════════════════════════════════════════════╗
║                     SWITCHING FROM SYNTHETIC TO REAL                       ║
╚════════════════════════════════════════════════════════════════════════════╝

Step 1: Test everything with synthetic data
  $ python main_training.py --num-epochs 5

Step 2: When you have real data, just add --real-data flag
  $ python main_training.py --real-data --num-epochs 50

That's it! Everything else stays the same.

╔════════════════════════════════════════════════════════════════════════════╗
║                         KEY PARAMETERS                                     ║
╚════════════════════════════════════════════════════════════════════════════╝

Data Mode:
  --real-data          Use real data instead of synthetic
  --data-path PATH     Path to .npy file (default: 2023-11-15-cine-myo-masks-and-TOS.npy)
  --train-split 0.8    Train/val split for real data

Synthetic Data Only:
  --train-size 80      Number of synthetic training samples
  --val-size 20        Number of synthetic validation samples

Training:
  --num-epochs 50      Number of epochs
  --learning-rate 1e-3 Learning rate
  --batch-size 8       Batch size
  
Device:
  --device auto        Device: auto/cuda/cpu

Checkpointing:
  --checkpoint-dir DIR Where to save checkpoints
  --resume PATH        Resume from checkpoint
  --save-every 5       Save checkpoint every N epochs

    """)


if __name__ == "__main__":
    import sys
    
    # If --help or -h is passed, show usage examples too
    if '--help' in sys.argv or '-h' in sys.argv:
        parser = argparse.ArgumentParser(description='Train Sequencer for Cardiac TOS Prediction')
        parser.print_help()
        print_usage_examples()
    else:
        # Run training
        main()