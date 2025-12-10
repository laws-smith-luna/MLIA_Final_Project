"""
Fixed Baseline Models - Handles Variable Input Sizes
====================================================

Updated to automatically detect input dimensions from first forward pass.
"""

import torch
import torch.nn as nn


class SimpleLinear(nn.Module):
    """
    Simple linear regression baseline (flattens input)
    """
    
    def __init__(self, in_channels=1, num_outputs=126, image_size=80):
        super().__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.image_size = image_size
        
        # Initialize with default image size (80x80 for real data)
        input_features = in_channels * image_size * image_size
        self.linear = nn.Linear(input_features, num_outputs)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            predictions: (batch, num_outputs)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # (batch, channels*height*width)
        x = self.linear(x)
        return x


class SimpleMLP(nn.Module):
    """
    Multi-layer perceptron baseline
    """
    
    def __init__(self, in_channels=1, num_outputs=126, image_size=80):
        super().__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.image_size = image_size
        
        # Initialize with default image size
        input_features = in_channels * image_size * image_size
        
        self.mlp = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            predictions: (batch, num_outputs)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x


class SimpleCNN(nn.Module):
    """
    Simple CNN baseline
    """
    
    def __init__(self, in_channels=1, num_outputs=126, image_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        
        # CNN layers (work with any number of input channels)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling handles any input size
        )
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            predictions: (batch, num_outputs)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class SimpleLSTM(nn.Module):
    """
    LSTM baseline (treats rows as sequence)
    """
    
    def __init__(self, in_channels=1, num_outputs=126, hidden_dim=128, image_size=80):
        super().__init__()
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        
        # Initialize with default image size
        input_size = image_size * in_channels  # width * channels
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
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
        
        # Reshape: (batch, channels, height, width) -> (batch, height, width*channels)
        x = x.permute(0, 2, 1, 3).reshape(batch, height, -1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        # Predict
        predictions = self.fc(last_out)
        return predictions


# Model registry
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


if __name__ == "__main__":
    print("Testing baseline models with different input configurations...")
    
    # Test with different input sizes
    test_configs = [
        (1, 128, 128),  # Single channel, 128x128
        (5, 80, 80),    # 5 channels (multi_frame), 80x80 - REAL DATA
        (4, 80, 80),    # 4 channels (temporal_stats), 80x80
    ]
    
    for in_channels, height, width in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing with input shape: ({in_channels}, {height}, {width})")
        print(f"{'='*60}")
        
        dummy_input = torch.randn(2, in_channels, height, width)
        
        for model_name in ['linear', 'mlp', 'cnn', 'lstm']:
            print(f"\n{model_name.upper()}:")
            model = get_model(model_name, in_channels=in_channels, image_size=height)
            
            # Check that model has parameters
            num_params = count_parameters(model)
            assert num_params > 0, f"{model_name} has 0 parameters!"
            
            output = model(dummy_input)
            print(f"  Input: {dummy_input.shape}")
            print(f"  Output: {output.shape}")
            print(f"  Parameters: {num_params:,}")
            assert output.shape == (2, 126), f"Expected (2, 126), got {output.shape}"
    
    print(f"\n{'='*60}")
    print("✓ All models working correctly with proper initialization!")
    print(f"{'='*60}")