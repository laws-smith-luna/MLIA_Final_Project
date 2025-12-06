import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path

"""
Training Pipeline for Sequencer Regression
===========================================

Purpose: Train the Sequencer model to predict TOS curves from cardiac masks

Components:
1. Loss function (Mean Squared Error for regression)
2. Optimizer (AdamW - adaptive learning rate)
3. Training loop (forward pass, compute loss, backprop, update weights)
4. Validation loop (evaluate on held-out data)
5. Checkpointing (save best model)
6. Metrics tracking (loss curves, predictions)
"""


class Trainer:
    """
    Handles the complete training process
    
    Purpose: Encapsulates all training logic in one place for easy use
    
    Key responsibilities:
    - Run training epochs
    - Validate periodically
    - Save checkpoints
    - Track metrics
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints'
    ):
        """
        Args:
            model: Sequencer model instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Initial learning rate
            device: 'cuda' or 'cpu'
            checkpoint_dir: Where to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # === Loss Function ===
        # MSE (Mean Squared Error) for regression
        # Measures: How far are predictions from ground truth?
        # Lower is better
        self.criterion = nn.MSELoss()
        
        # === Optimizer ===
        # AdamW: Adaptive learning rate with weight decay (regularization)
        # Weight decay prevents overfitting by penalizing large weights
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.05  # L2 regularization strength
        )
        
        # === Learning Rate Scheduler ===
        # Reduces learning rate when validation loss plateaus
        # Purpose: Fine-tune learning at the end of training
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',         # Minimize validation loss
            factor=0.5,         # Reduce LR by half
            patience=5,         # Wait 5 epochs before reducing
            verbose=True
        )
        
        # === Metrics Tracking ===
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
    
    def train_epoch(self):
        """
        Train for one complete epoch
        
        Process:
        1. Set model to training mode
        2. For each batch:
           - Forward pass: get predictions
           - Compute loss
           - Backward pass: compute gradients
           - Update weights
        3. Return average loss
        
        Returns:
            avg_loss: Average training loss for this epoch
        """
        self.model.train()  # Enable dropout, batch norm training mode
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (masks, tos_targets) in enumerate(self.train_loader):
            # Move data to GPU/CPU
            masks = masks.to(self.device)
            tos_targets = tos_targets.to(self.device)
            
            # === Forward Pass ===
            # Get model predictions
            predictions = self.model(masks)
            
            # Compute loss: how far are we from targets?
            loss = self.criterion(predictions, tos_targets)
            
            # === Backward Pass ===
            # Zero out gradients from previous iteration
            self.optimizer.zero_grad()
            
            # Compute gradients via backpropagation
            loss.backward()
            
            # Update model weights
            self.optimizer.step()
            
            # === Track Statistics ===
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Return average loss across all batches
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """
        Evaluate model on validation set
        
        Purpose: Check how well model generalizes to unseen data
        
        Key difference from training:
        - No gradient computation (faster, less memory)
        - No weight updates
        - Model in eval mode (disables dropout)
        
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()  # Disable dropout, batch norm in eval mode
        
        total_loss = 0.0
        num_batches = 0
        
        # Disable gradient computation for validation
        with torch.no_grad():
            for masks, tos_targets in self.val_loader:
                # Move data to device
                masks = masks.to(self.device)
                tos_targets = tos_targets.to(self.device)
                
                # Forward pass only
                predictions = self.model(masks)
                
                # Compute loss
                loss = self.criterion(predictions, tos_targets)
                
                # Track statistics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Save model checkpoint
        
        Purpose: Save model weights so you can resume training or use
        the model later for inference
        
        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        print(f"  Saved checkpoint to {latest_path}")
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ⭐ New best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a saved checkpoint to resume training
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epochs_trained = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Loaded checkpoint from epoch {self.epochs_trained}")
    
    def train(self, num_epochs, save_every=5):
        """
        Main training loop
        
        Purpose: Train for multiple epochs, validate, and track progress
        
        Args:
            num_epochs: How many epochs to train
            save_every: Save checkpoint every N epochs
        """
        print("=" * 60)
        print(f"Starting Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        for epoch in range(self.epochs_trained, self.epochs_trained + num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.epochs_trained + num_epochs}")
            print("-" * 60)
            
            # === Training Phase ===
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # === Validation Phase ===
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # === Update Learning Rate ===
            # Scheduler reduces LR if validation loss plateaus
            self.scheduler.step(val_loss)
            
            # === Logging ===
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Time:       {epoch_time:.2f}s")
            print(f"  LR:         {current_lr:.6f}")
            
            # === Checkpointing ===
            # Check if this is the best model so far
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint periodically or if best
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        # Save final training history
        self._save_training_history()
    
    def _save_training_history(self):
        """Save training metrics to JSON file for analysis"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses)
        }
        
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved training history to {history_path}")


def test_model_prediction(model, val_loader, device='cpu', num_samples=3):
    """
    Test model predictions on a few samples
    
    Purpose: Sanity check - visualize predictions vs ground truth
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
        num_samples: Number of samples to test
    """
    model.eval()
    model.to(device)
    
    print("\n" + "=" * 60)
    print("Testing Model Predictions")
    print("=" * 60)
    
    with torch.no_grad():
        masks, tos_targets = next(iter(val_loader))
        masks = masks.to(device)
        tos_targets = tos_targets.to(device)
        
        predictions = model(masks)
        
        # Show predictions for first few samples
        for i in range(min(num_samples, len(predictions))):
            pred = predictions[i].cpu().numpy()
            target = tos_targets[i].cpu().numpy()
            
            # Compute metrics
            mse = ((pred - target) ** 2).mean()
            mae = abs(pred - target).mean()
            
            print(f"\nSample {i + 1}:")
            print(f"  MSE:  {mse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  Pred range: [{pred.min():.2f}, {pred.max():.2f}]")
            print(f"  True range: [{target.min():.2f}, {target.max():.2f}]")


# === Example Usage ===
if __name__ == "__main__":
    # This shows how to use the training pipeline
    print("Training Pipeline Ready!")
    print("\nTo use this in practice:")
    print("1. Import your model: from sequencer_model import Sequencer2D_S")
    print("2. Import data loaders: from synthetic_data import create_data_loaders")
    print("3. Create trainer: trainer = Trainer(model, train_loader, val_loader)")
    print("4. Start training: trainer.train(num_epochs=50)")