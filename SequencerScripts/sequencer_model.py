import torch
import torch.nn as nn

"""
Sequencer2D-S Architecture for Cardiac Regression
==================================================

Purpose: Predict TOS (Time of Stretch) curves from cardiac mask sequences

Architecture Overview:
1. Patch Embedding: Convert input into tokens (like words in a sentence)
2. BiLSTM2D Blocks: Process spatial relationships in both directions
3. Regression Head: Combine features and predict 126 continuous values

Key Innovation: Unlike transformers that use attention, Sequencer uses
bidirectional LSTMs to capture spatial patterns in images.
"""


class BiLSTM2D(nn.Module):
    """
    Bidirectional LSTM that processes images in 2D
    
    Purpose: Capture relationships between image patches by processing
    them in both horizontal and vertical directions separately, then
    combining the results.
    
    How it works:
    1. Process each row left-to-right and right-to-left (horizontal LSTM)
    2. Process each column top-to-bottom and bottom-to-top (vertical LSTM)
    3. Concatenate both outputs and project back to original dimension
    
    This is the core building block that replaces self-attention in transformers.
    """
    
    def __init__(self, dim, hidden_dim):
        """
        Args:
            dim: Number of channels in input/output (e.g., 192)
            hidden_dim: Hidden dimension for LSTM (e.g., 48)
                       BiLSTM outputs 2*hidden_dim (forward + backward)
        """
        super().__init__()
        
        # Horizontal LSTM: processes each row of the image
        # bidirectional=True means it goes both left->right and right->left
        self.horizontal_lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Vertical LSTM: processes each column of the image
        self.vertical_lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection layer: combines horizontal and vertical outputs
        # Input: 4*hidden_dim (2 from horizontal BiLSTM + 2 from vertical BiLSTM)
        # Output: dim (back to original channel dimension)
        self.projection = nn.Linear(4 * hidden_dim, dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, height, width, channels)
        
        Returns:
            Output tensor of same shape (batch, height, width, channels)
        """
        batch, height, width, channels = x.shape
        
        # === Process Vertically ===
        # Reshape to process each column: (batch*width, height, channels)
        # Think of this as: for each column, treat height as sequence length
        x_for_vertical = x.permute(0, 2, 1, 3).reshape(batch * width, height, channels)
        
        # Process with vertical LSTM
        vertical_out, _ = self.vertical_lstm(x_for_vertical)
        
        # Reshape back: (batch, width, height, 2*hidden_dim) -> (batch, height, width, 2*hidden_dim)
        vertical_out = vertical_out.reshape(batch, width, height, -1).permute(0, 2, 1, 3)
        
        # === Process Horizontally ===
        # Reshape to process each row: (batch*height, width, channels)
        # Think of this as: for each row, treat width as sequence length
        x_for_horizontal = x.reshape(batch * height, width, channels)
        
        # Process with horizontal LSTM
        horizontal_out, _ = self.horizontal_lstm(x_for_horizontal)
        
        # Reshape back: (batch*height, width, 2*hidden_dim) -> (batch, height, width, 2*hidden_dim)
        horizontal_out = horizontal_out.reshape(batch, height, width, -1)
        
        # === Combine Both Directions ===
        # Concatenate horizontal and vertical outputs: (batch, height, width, 4*hidden_dim)
        combined = torch.cat([vertical_out, horizontal_out], dim=-1)
        
        # Project back to original dimension: (batch, height, width, channels)
        output = self.projection(combined)
        
        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network)
    
    Purpose: Add non-linearity and capacity to the model after LSTM processing
    
    Architecture: Linear -> GELU -> Linear
    Uses expansion_ratio to temporarily increase dimension (like in transformers)
    """
    
    def __init__(self, dim, expansion_ratio=3):
        """
        Args:
            dim: Input and output dimension
            expansion_ratio: How much to expand in the hidden layer (default: 3x)
        """
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Smooth activation function (better than ReLU for deep networks)
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SequencerBlock(nn.Module):
    """
    Single Sequencer Block
    
    Purpose: One processing unit that combines BiLSTM2D and MLP
    
    Structure (similar to Transformer blocks):
    1. Layer Norm -> BiLSTM2D -> Add residual connection
    2. Layer Norm -> MLP -> Add residual connection
    
    The residual connections (x + layer(x)) help gradients flow during training.
    """
    
    def __init__(self, dim, hidden_dim, mlp_ratio=3):
        """
        Args:
            dim: Channel dimension
            hidden_dim: Hidden dimension for BiLSTM
            mlp_ratio: Expansion ratio for MLP
        """
        super().__init__()
        
        # Normalization layers (normalize across channel dimension)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Main processing layers
        self.bilstm = BiLSTM2D(dim, hidden_dim)
        self.mlp = MLP(dim, mlp_ratio)
    
    def forward(self, x):
        """
        Args:
            x: (batch, height, width, channels)
        """
        # BiLSTM path with residual connection
        x = x + self.bilstm(self.norm1(x))
        
        # MLP path with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchEmbedding(nn.Module):
    """
    Convert input into patches (tokens)
    
    Purpose: Transform raw input into a sequence of feature vectors
    Similar to how text is broken into words, images are broken into patches
    
    Uses convolution with stride to both:
    1. Extract features
    2. Downsample the spatial dimensions
    """
    
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale masks)
            embed_dim: Output channel dimension
            patch_size: Size of convolutional kernel
            stride: Downsampling factor
        """
        super().__init__()
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=stride,
            padding=patch_size // 2  # Keep spatial dimensions manageable
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) - standard Conv2d format
        Returns:
            (batch, height', width', embed_dim) - Sequencer format (channels last)
        """
        x = self.proj(x)
        # Permute to channels-last format: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x


class Sequencer2D_S(nn.Module):
    """
    Full Sequencer2D-S Model for Regression
    
    Purpose: Predict 126 TOS values from cardiac mask sequences
    
    Architecture stages:
    Stage 1: Patch embed (7x7, stride 7) + 4 Sequencer blocks
    Stage 2: Patch embed (2x2, stride 2) + 3 Sequencer blocks  
    Stage 3: Point-wise linear (1x1 conv) + 8 Sequencer blocks
    Stage 4: Point-wise linear (1x1 conv) + 3 Sequencer blocks
    Final: Layer Norm -> Global Average Pool -> Linear -> 126 outputs
    """
    
    def __init__(self, in_channels=1, num_outputs=126):
        """
        Args:
            in_channels: Input channels (1 for grayscale masks)
            num_outputs: Number of regression outputs (126 for TOS)
        """
        super().__init__()
        
        # === Stage 1: Initial embedding ===
        # Downsample 7x with 7x7 patches, output 192 channels
        self.stage1_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=192,
            patch_size=7,
            stride=7
        )
        
        # 4 Sequencer blocks with dimension 192, hidden dim 48
        self.stage1_blocks = nn.ModuleList([
            SequencerBlock(dim=192, hidden_dim=48, mlp_ratio=3)
            for _ in range(4)
        ])
        
        # === Stage 2: Further downsampling ===
        # Downsample 2x with 2x2 patches, output 384 channels
        self.stage2_embed = PatchEmbedding(
            in_channels=192,
            embed_dim=384,
            patch_size=2,
            stride=2
        )
        
        # 3 Sequencer blocks with dimension 384, hidden dim 96
        self.stage2_blocks = nn.ModuleList([
            SequencerBlock(dim=384, hidden_dim=96, mlp_ratio=3)
            for _ in range(3)
        ])
        
        # === Stage 3: No spatial downsampling ===
        # Point-wise linear (1x1 conv) keeps spatial size, maintains 384 channels
        self.stage3_downsample = nn.Conv2d(384, 384, kernel_size=1)
        
        # 8 Sequencer blocks
        self.stage3_blocks = nn.ModuleList([
            SequencerBlock(dim=384, hidden_dim=96, mlp_ratio=3)
            for _ in range(8)
        ])
        
        # === Stage 4: No spatial downsampling ===
        # Point-wise linear maintains 384 channels
        self.stage4_downsample = nn.Conv2d(384, 384, kernel_size=1)
        
        # 3 Sequencer blocks
        self.stage4_blocks = nn.ModuleList([
            SequencerBlock(dim=384, hidden_dim=96, mlp_ratio=3)
            for _ in range(3)
        ])
        
        # === Regression Head ===
        # Final layer norm
        self.norm = nn.LayerNorm(384)
        
        # Linear layer to produce final predictions
        # Input: 384 features, Output: 126 TOS values
        self.head = nn.Linear(384, num_outputs)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, height, width)
               For cardiac masks: (batch, 1, H, W)
        
        Returns:
            predictions: (batch, 126) - TOS curve predictions
        """
        # Stage 1: (B, 1, H, W) -> (B, H/7, W/7, 192)
        x = self.stage1_embed(x)
        for block in self.stage1_blocks:
            x = block(x)
        
        # Convert back to channels-first for conv: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Stage 2: (B, 192, H/7, W/7) -> (B, H/14, W/14, 384)
        x = self.stage2_embed(x)
        for block in self.stage2_blocks:
            x = block(x)
        
        # Convert for stage 3 downsampling
        x = x.permute(0, 3, 1, 2)
        
        # Stage 3: (B, 384, H/14, W/14) -> (B, H/14, W/14, 384)
        x = self.stage3_downsample(x)
        x = x.permute(0, 2, 3, 1)
        for block in self.stage3_blocks:
            x = block(x)
        
        # Convert for stage 4 downsampling
        x = x.permute(0, 3, 1, 2)
        
        # Stage 4: (B, 384, H/14, W/14) -> (B, H/14, W/14, 384)
        x = self.stage4_downsample(x)
        x = x.permute(0, 2, 3, 1)
        for block in self.stage4_blocks:
            x = block(x)
        
        # === Regression Head ===
        # Normalize: (B, H/14, W/14, 384)
        x = self.norm(x)
        
        # Global Average Pooling: average across spatial dimensions
        # (B, H/14, W/14, 384) -> (B, 384)
        x = x.mean(dim=[1, 2])
        
        # Final linear projection to TOS predictions
        # (B, 384) -> (B, 126)
        predictions = self.head(x)
        
        return predictions


# === Quick test to verify architecture ===
if __name__ == "__main__":
    # Create model
    model = Sequencer2D_S(in_channels=1, num_outputs=126)
    
    # Test with dummy input (batch=2, channels=1, height=128, width=128)
    dummy_input = torch.randn(2, 1, 128, 128)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (2, 126)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")