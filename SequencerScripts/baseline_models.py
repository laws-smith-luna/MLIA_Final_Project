"""
Baseline Models for Cardiac TOS Prediction
===========================================

Purpose: Implement simple baseline models to compare against Sequencer.
This proves that Sequencer's complexity is actually beneficial.

Baselines implemented:
1. SimpleLinear - Linear regression (simplest possible)
2. SimpleMLP - Multi-layer perceptron (fully connected)
3. SimpleCNN - Convolutional neural network
4. SimpleLSTM - LSTM without spatial processing

Each baseline predicts 126 TOS values from cardiac mask images.
"""

import torch
import torch.nn as nn


class SimpleLinear(nn.Module):
    """
    Linear Regression Baseline
    
    Purpose: Simplest possible model - just a linear mapping
    Architecture: Flatten input → Linear layer → Output
    
    This is the absolute baseline. If Sequencer can't beat this,
    something is seriously wrong.
    """
    
    def __init__(self, image_size=128, in_channels=1, num_outputs=126):
        super().__init__()
        
        input_dim = image_size * image_size * in_channels
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_outputs)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            predictions: (batch, num_outputs)
        """
        x = self.flatten(x)  # (batch, C*H*W)
        x = self.linear(x)   # (batch, num_outputs)
        return x


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron Baseline
    
    Purpose: Simple fully-connected network with hidden layers
    Architecture: Flatten → Linear → ReLU → Linear → ReLU → Linear
    
    This tests whether simple non-linearity is enough, without
    spatial processing like CNNs or Sequencer.
    """
    
    def __init__(self, image_size=128, in_channels=1, num_outputs=126, 
                 hidden_dims=[512, 256]):
        super().__init__()
        
        input_dim = image_size * image_size * in_channels
        
        layers = []
        layers.append(nn.Flatten())
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))  # Regularization
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], num_outputs))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network Baseline
    
    Purpose: Standard CNN architecture for comparison
    Architecture: Conv blocks → Global Avg Pool → Linear
    
    This tests whether standard convolutional processing is enough,
    without the bidirectional LSTM spatial processing of Sequencer.
    """
    
    def __init__(self, in_channels=1, num_outputs=126):
        super().__init__()
        
        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            # Block 1: (1, 128, 128) → (32, 64, 64)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: (32, 64, 64) → (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Global average pooling: (256, 8, 8) → (256,)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            predictions: (batch, num_outputs)
        """
        x = self.conv_blocks(x)    # (batch, 256, 8, 8)
        x = self.global_pool(x)    # (batch, 256, 1, 1)
        x = self.fc(x)             # (batch, num_outputs)
        return x


class SimpleLSTM(nn.Module):
    """
    LSTM Baseline
    
    Purpose: Standard LSTM without 2D spatial processing
    Architecture: Treat each row as a sequence → LSTM → Linear
    
    This tests whether LSTM helps, but without the bidirectional
    horizontal+vertical processing that Sequencer uses.
    """
    
    def __init__(self, in_channels=1, num_outputs=126, hidden_dim=128):
        super().__init__()
        
        # Each row of the image is treated as a sequence
        # Input at each timestep is (width * channels)
        self.lstm = nn.LSTM(
            input_size=128 * in_channels,  # width * channels
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Map LSTM output to predictions
        # BiLSTM outputs 2*hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            predictions: (batch, num_outputs)
        """
        batch, channels, height, width = x.shape
        
        # Reshape to treat rows as sequences
        # (batch, channels, height, width) → (batch, height, width*channels)
        x = x.permute(0, 2, 1, 3).reshape(batch, height, -1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x)  # (batch, height, 2*hidden_dim)
        
        # Take output from last timestep
        last_out = lstm_out[:, -1, :]  # (batch, 2*hidden_dim)
        
        # Predict
        predictions = self.fc(last_out)  # (batch, num_outputs)
        
        return predictions


# Model registry for easy access
BASELINE_MODELS = {
    'linear': SimpleLinear,
    'mlp': SimpleMLP,
    'cnn': SimpleCNN,
    'lstm': SimpleLSTM,
}


def get_model(model_name, **kwargs):
    """
    Factory function to create models by name
    
    Args:
        model_name: One of 'linear', 'mlp', 'cnn', 'lstm', 'sequencer'
        **kwargs: Model-specific arguments
    
    Returns:
        model: Initialized model
    """
    if model_name == 'sequencer':
        from sequencer_model import Sequencer2D_S
        return Sequencer2D_S(**kwargs)
    
    elif model_name in BASELINE_MODELS:
        return BASELINE_MODELS[model_name](**kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: {list(BASELINE_MODELS.keys())} or 'sequencer'")


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# === Test all models ===
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Baseline Models")
    print("=" * 70)
    
    # Test input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 128)
    
    # Test all models
    models = {
        'Linear': SimpleLinear(),
        'MLP': SimpleMLP(),
        'CNN': SimpleCNN(),
        'LSTM': SimpleLSTM(),
    }
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Expected output: ({batch_size}, 126)\n")
    
    for name, model in models.items():
        output = model(dummy_input)
        params = count_parameters(model)
        
        print(f"{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {params:,}")
        print(f"  ✓ Passed" if output.shape == (batch_size, 126) else "  ✗ Failed")
        print()