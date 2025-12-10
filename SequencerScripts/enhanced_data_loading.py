"""
Enhanced Real Data Loading with Multiple Temporal Processing Options
=====================================================================

This provides multiple strategies for handling temporal cardiac data.
You can easily switch between them to find what works best.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class RealCardiacDataset(Dataset):
    """
    Enhanced dataset with multiple temporal processing strategies
    """
    
    def __init__(self, data_samples, temporal_mode='multi_frame'):
        """
        Args:
            data_samples: Array of data samples from .npy file
            temporal_mode: How to handle temporal dimension
                - 'single_frame': Use middle frame only (current approach)
                - 'average': Average all frames
                - 'max_projection': Max intensity projection
                - 'peak_frame': Use frame with maximum mask area
                - 'multi_frame': Use multiple key frames as channels (RECOMMENDED)
                - 'temporal_stats': Use statistical features across time
        """
        self.data = data_samples
        self.temporal_mode = temporal_mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        mask_volume = sample['cine_lv_myo_masks_cropped']  # (H, W, n_frames)
        tos_curve = sample['TOS']  # (126,)
        
        # Process temporal dimension based on mode
        if self.temporal_mode == 'single_frame':
            mask = self._single_frame(mask_volume)
            
        elif self.temporal_mode == 'average':
            mask = self._average_frames(mask_volume)
            
        elif self.temporal_mode == 'max_projection':
            mask = self._max_projection(mask_volume)
            
        elif self.temporal_mode == 'peak_frame':
            mask = self._peak_frame(mask_volume)
            
        elif self.temporal_mode == 'multi_frame':
            mask = self._multi_frame(mask_volume)
            
        elif self.temporal_mode == 'temporal_stats':
            mask = self._temporal_stats(mask_volume)
            
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")
        
        # Convert to tensors
        mask_tensor = torch.from_numpy(mask).float()
        tos_tensor = torch.from_numpy(tos_curve).float()
        
        return mask_tensor, tos_tensor
    
    # === Temporal Processing Methods ===
    
    def _single_frame(self, mask_volume):
        """Current approach: middle frame only"""
        middle_frame = mask_volume.shape[2] // 2
        mask_frame = mask_volume[:, :, middle_frame]
        return mask_frame[np.newaxis, ...]  # (1, H, W)
    
    def _average_frames(self, mask_volume):
        """Average across all frames"""
        mask_frame = mask_volume.mean(axis=2)
        return mask_frame[np.newaxis, ...]  # (1, H, W)
    
    def _max_projection(self, mask_volume):
        """Maximum intensity projection"""
        mask_frame = mask_volume.max(axis=2)
        return mask_frame[np.newaxis, ...]  # (1, H, W)
    
    def _peak_frame(self, mask_volume):
        """Use frame with maximum mask area (peak contraction)"""
        mask_areas = mask_volume.sum(axis=(0, 1))  # Sum over H, W
        peak_idx = mask_areas.argmax()
        mask_frame = mask_volume[:, :, peak_idx]
        return mask_frame[np.newaxis, ...]  # (1, H, W)
    
    def _multi_frame(self, mask_volume):
        """
        Use multiple key frames as separate channels (RECOMMENDED)
        
        This captures temporal dynamics without requiring model changes.
        Uses 5 frames: start, 1/4, middle, 3/4, end
        """
        n_frames = mask_volume.shape[2]
        
        # Select 5 evenly-spaced key frames
        indices = [
            0,                          # Start of cycle
            n_frames // 4,              # Early phase
            n_frames // 2,              # Middle phase
            3 * n_frames // 4,          # Late phase
            n_frames - 1                # End of cycle
        ]
        
        # Stack as channels: (5, H, W)
        key_frames = np.stack([mask_volume[:, :, i] for i in indices], axis=0)
        return key_frames
    
    def _temporal_stats(self, mask_volume):
        """
        Statistical features across time (mean, std, min, max)
        
        Captures temporal variation without multiple channels
        """
        mean_frame = mask_volume.mean(axis=2)
        std_frame = mask_volume.std(axis=2)
        min_frame = mask_volume.min(axis=2)
        max_frame = mask_volume.max(axis=2)
        
        # Stack as channels: (4, H, W)
        stats = np.stack([mean_frame, std_frame, min_frame, max_frame], axis=0)
        return stats


def load_real_data(data_path, train_split=0.8, temporal_mode='multi_frame', batch_size=8):
    """
    Load real cardiac data with specified temporal processing
    
    Args:
        data_path: Path to .npy file
        train_split: Fraction for training
        temporal_mode: How to process temporal dimension
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, num_input_channels
    """
    print(f"Loading real data from {data_path}...")
    print(f"Temporal mode: {temporal_mode}")
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    print(f"Total samples: {len(data)}")
    
    # Split into train/val
    num_train = int(len(data) * train_split)
    train_data = data[:num_train]
    val_data = data[num_train:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = RealCardiacDataset(train_data, temporal_mode=temporal_mode)
    val_dataset = RealCardiacDataset(val_data, temporal_mode=temporal_mode)
    
    # Determine number of input channels based on mode
    if temporal_mode == 'multi_frame':
        num_channels = 5  # 5 key frames
    elif temporal_mode == 'temporal_stats':
        num_channels = 4  # mean, std, min, max
    else:
        num_channels = 1  # single image
    
    print(f"Input channels: {num_channels}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, val_loader, num_channels


# === Quick Test Function ===
def test_temporal_modes(data_path):
    """
    Quick test of all temporal modes to see data shapes
    """
    print("="*70)
    print("Testing All Temporal Modes")
    print("="*70)
    
    modes = ['single_frame', 'average', 'max_projection', 
             'peak_frame', 'multi_frame', 'temporal_stats']
    
    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        try:
            train_loader, val_loader, num_channels = load_real_data(
                data_path, 
                temporal_mode=mode,
                batch_size=2
            )
            
            # Get sample batch
            masks, tos = next(iter(train_loader))
            print(f"Mask shape: {masks.shape}")
            print(f"TOS shape: {tos.shape}")
            print(f"Num channels: {num_channels}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "="*70)
    print("Recommendation: Use 'multi_frame' for best temporal representation")
    print("="*70)


if __name__ == "__main__":
    # Test with your data file
    test_temporal_modes('2023-11-15-cine-myo-masks-and-TOS.npy')