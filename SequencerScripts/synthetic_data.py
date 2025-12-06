import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
Synthetic Data Generator for Cardiac Regression
================================================

Purpose: Create fake cardiac mask sequences and TOS curves for testing
the Sequencer model before real data arrives.

Real data format (from README):
- mask_volume: (H, W, n_frames) - heart contours over time
- TOS: (126,) - time of stretch curve

We'll generate realistic-looking synthetic data that mimics this structure.
"""


class SyntheticCardiacDataset(Dataset):
    """
    Generates synthetic cardiac mask sequences and TOS curves
    
    How it works:
    1. Creates binary masks with circular/elliptical shapes (mimicking heart contours)
    2. Adds temporal variation (heart beating) across frames
    3. Generates TOS curves with smooth patterns
    
    This lets you test the full pipeline without real data.
    """
    
    def __init__(self, num_samples=100, image_size=128, num_frames=40, num_outputs=126):
        """
        Args:
            num_samples: Number of synthetic patient samples to generate
            image_size: Spatial dimensions (height, width) 
            num_frames: Number of time frames per sequence
            num_outputs: Length of TOS curve (126 values)
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_outputs = num_outputs
        
        # Pre-generate all data at initialization
        # In real scenario, you'd load from disk
        self.masks, self.tos_curves = self._generate_all_data()
    
    def _generate_single_mask_sequence(self, idx):
        """
        Generate a single cardiac mask sequence with temporal variation
        
        Simulates a beating heart:
        - Creates an ellipse that changes size over time
        - Adds some noise to make it realistic
        
        Returns:
            mask_sequence: (H, W, T) binary mask
        """
        # Use idx as seed for reproducibility
        np.random.seed(idx)
        
        # Initialize sequence
        mask_sequence = np.zeros((self.image_size, self.image_size, self.num_frames))
        
        # Center of the "heart"
        center_x = self.image_size // 2 + np.random.randint(-10, 10)
        center_y = self.image_size // 2 + np.random.randint(-10, 10)
        
        # Base size (will vary over time to simulate heartbeat)
        base_radius_x = np.random.randint(20, 35)
        base_radius_y = np.random.randint(20, 35)
        
        # Generate each frame
        for t in range(self.num_frames):
            # Simulate heartbeat: size varies sinusoidally over time
            # Peak at frame 0, shrink, expand again (systole/diastole cycle)
            phase = 2 * np.pi * t / self.num_frames
            size_variation = 1.0 + 0.3 * np.sin(phase)  # ±30% size variation
            
            radius_x = int(base_radius_x * size_variation)
            radius_y = int(base_radius_y * size_variation)
            
            # Create elliptical mask
            y, x = np.ogrid[:self.image_size, :self.image_size]
            ellipse_mask = ((x - center_x)**2 / radius_x**2 + 
                           (y - center_y)**2 / radius_y**2) <= 1
            
            mask_sequence[:, :, t] = ellipse_mask.astype(np.float32)
            
            # Add small random noise to make it more realistic
            noise = np.random.randn(self.image_size, self.image_size) * 0.05
            mask_sequence[:, :, t] = np.clip(mask_sequence[:, :, t] + noise, 0, 1)
        
        return mask_sequence
    
    def _generate_single_tos_curve(self, idx):
        """
        Generate a single TOS (Time of Stretch) curve
        
        Creates a smooth curve with:
        - Overall trend (linear or polynomial)
        - Periodic components (simulating cardiac cycles)
        - Small random variations
        
        Returns:
            tos_curve: (126,) array of continuous values
        """
        np.random.seed(idx + 10000)  # Different seed from masks
        
        # Time points for TOS curve
        t = np.linspace(0, 1, self.num_outputs)
        
        # Base trend (slight upward or downward)
        trend = np.random.randn() * 0.5 * t + np.random.randn() * 0.2
        
        # Add periodic component (simulating cardiac cycles)
        # Frequency: 2-4 cycles over the curve
        frequency = np.random.uniform(2, 4)
        periodic = 0.3 * np.sin(2 * np.pi * frequency * t + np.random.uniform(0, 2*np.pi))
        
        # Add smaller high-frequency variation
        high_freq = 0.1 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
        
        # Combine components
        tos_curve = trend + periodic + high_freq
        
        # Add small random noise
        tos_curve += np.random.randn(self.num_outputs) * 0.05
        
        # Normalize to reasonable range (e.g., -1 to 1)
        tos_curve = (tos_curve - tos_curve.mean()) / (tos_curve.std() + 1e-8)
        
        return tos_curve.astype(np.float32)
    
    def _generate_all_data(self):
        """
        Pre-generate all synthetic data samples
        
        Returns:
            masks: List of mask sequences
            tos_curves: List of TOS curves
        """
        print(f"Generating {self.num_samples} synthetic cardiac samples...")
        
        masks = []
        tos_curves = []
        
        for i in range(self.num_samples):
            mask = self._generate_single_mask_sequence(i)
            tos = self._generate_single_tos_curve(i)
            
            masks.append(mask)
            tos_curves.append(tos)
        
        print(f"Generated {len(masks)} mask sequences and {len(tos_curves)} TOS curves")
        return masks, tos_curves
    
    def __len__(self):
        """Return the total number of samples"""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            mask: (1, H, W) tensor - using single frame for now
            tos: (126,) tensor - target TOS curve
        """
        # Get the full sequence
        mask_sequence = self.masks[idx]  # (H, W, T)
        tos_curve = self.tos_curves[idx]  # (126,)
        
        # For now, use a single representative frame (middle frame)
        # Later, you can modify this to:
        # - Average across all frames
        # - Use max projection
        # - Feed temporal sequence to model
        middle_frame = self.num_frames // 2
        mask_frame = mask_sequence[:, :, middle_frame]
        
        # Convert to PyTorch tensors with float32 type
        # Add channel dimension: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask_frame).float().unsqueeze(0)
        tos_tensor = torch.from_numpy(tos_curve).float()
        
        return mask_tensor, tos_tensor


def create_data_loaders(train_size=80, val_size=20, batch_size=8):
    """
    Create train and validation data loaders
    
    Purpose: Split synthetic data into training and validation sets,
    and create PyTorch DataLoaders for batching.
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        batch_size: Batch size for training
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Create datasets
    train_dataset = SyntheticCardiacDataset(num_samples=train_size)
    val_dataset = SyntheticCardiacDataset(num_samples=val_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0  # Set to 0 for synthetic data (can increase for real data)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=0
    )
    
    return train_loader, val_loader


# === Test the synthetic data generator ===
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Synthetic Data Generator")
    print("=" * 60)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_size=80,
        val_size=20,
        batch_size=8
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Get a sample batch
    masks, tos_curves = next(iter(train_loader))
    
    print(f"\n--- Sample Batch ---")
    print(f"Mask batch shape: {masks.shape}")
    print(f"TOS batch shape: {tos_curves.shape}")
    print(f"Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"TOS value range: [{tos_curves.min():.3f}, {tos_curves.max():.3f}]")
    
    # Visualize one sample
    print(f"\n--- Single Sample Statistics ---")
    single_mask = masks[0, 0].numpy()  # First sample, remove channel dim
    single_tos = tos_curves[0].numpy()
    
    print(f"Mask shape: {single_mask.shape}")
    print(f"Mask non-zero pixels: {np.count_nonzero(single_mask)}")
    print(f"TOS curve shape: {single_tos.shape}")
    print(f"TOS mean: {single_tos.mean():.3f}, std: {single_tos.std():.3f}")
    
    print("\n" + "=" * 60)
    print("Data generation successful!")
    print("=" * 60)