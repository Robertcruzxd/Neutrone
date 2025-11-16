#!/usr/bin/env python3
"""
Train a neural network to estimate drone angle from event camera data.

This model takes event data (x, y coordinates) and predicts the drone's angle
relative to a reference position.
"""

import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


class EventDataset(Dataset):
    """Dataset for event camera frames with angle labels."""
    
    def __init__(self, frames: List[Dict], max_events: int = 10000):
        """
        Args:
            frames: List of frame metadata from dataset
            max_events: Maximum number of events to use per frame (padding/truncation)
        """
        self.frames = []
        self.max_events = max_events
        
        # Filter out frames that are out of frame or don't have valid data
        for frame in frames:
            stats = frame.get('event_stats', {})
            if (not stats.get('out_of_frame', False) and 
                'x_mean' in stats and 
                'angle_deg' in frame):
                self.frames.append(frame)
        
        print(f"Loaded {len(self.frames)} valid frames out of {len(frames)} total")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        stats = frame['event_stats']
        
        # Extract angle (target)
        angle = torch.tensor(frame['angle_deg'], dtype=torch.float32)
        
        # Create feature vector from statistics
        features = torch.tensor([
            stats['x_mean'],
            stats['y_mean'],
            stats['x_std'],
            stats['y_std'],
            stats['num_filtered_events'],
            stats.get('tracking_confidence', 0.5),
            stats.get('num_propellers', 0)
        ], dtype=torch.float32)
        
        # Add propeller positions if available (up to 4 propellers)
        propeller_features = []
        propellers = stats.get('propellers', [])
        for i in range(4):
            if i < len(propellers):
                propeller_features.extend([propellers[i]['x'], propellers[i]['y']])
            else:
                propeller_features.extend([0.0, 0.0])  # Padding
        
        features = torch.cat([features, torch.tensor(propeller_features, dtype=torch.float32)])
        
        return features, angle


class AngleEstimationModel(nn.Module):
    """Neural network for estimating drone angle from event features."""
    
    def __init__(self, input_size: int = 15):
        super().__init__()
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Angle regression head
        self.angle_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        features = self.feature_net(x)
        angle = self.angle_head(features)
        return angle.squeeze()


def normalize_features(features: torch.Tensor, stats: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """Normalize features to 0 mean and unit variance."""
    if stats is None:
        mean = features.mean(dim=0)
        std = features.std(dim=0) + 1e-8
        stats = {'mean': mean, 'std': std}
    else:
        mean = stats['mean']
        std = stats['std']
    
    normalized = (features - mean) / std
    return normalized, stats


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the angle estimation model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, angles in train_loader:
            features, angles = features.to(device), angles.to(device)
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, angles)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, angles in val_loader:
                features, angles = features.to(device), angles.to(device)
                predictions = model(features)
                loss = criterion(predictions, angles)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/best_angle_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Best Val: {best_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu'):
    """Evaluate model and compute metrics."""
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, angles in test_loader:
            features = features.to(device)
            predictions = model(features)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(angles.numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    print(f"\nTest Set Evaluation:")
    print(f"  Mean Absolute Error: {mae:.2f}°")
    print(f"  Root Mean Squared Error: {rmse:.2f}°")
    
    return all_predictions, all_targets


def plot_results(train_losses, val_losses, predictions, targets, output_dir: Path):
    """Plot training results and predictions."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('True Angle (degrees)')
    plt.ylabel('Predicted Angle (degrees)')
    plt.title('Predictions vs Ground Truth')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150)
    print(f"\nSaved training results to {output_dir / 'training_results.png'}")
    plt.close()


def main():
    # Load dataset
    print("Loading dataset...")
    with open('drone_dataset/moving/metadata.json') as f:
        moving_data = json.load(f)
    
    # Create dataset
    full_dataset = EventDataset(moving_data['frames'])
    
    # Split into train/val/test (70/15/15)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split: {train_size} train, {val_size} val, {test_size} test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = AngleEstimationModel(input_size=15)  # 7 basic features + 8 propeller coords
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        train_loader, val_loader, model,
        epochs=100,
        lr=0.001
    )
    
    # Load best model
    checkpoint = torch.load('models/best_angle_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on test set
    predictions, targets = evaluate_model(model, test_loader)
    
    # Plot results
    plot_results(train_losses, val_losses, predictions, targets, Path('models'))
    
    print("\n✓ Training complete!")
    print(f"  Model saved to: models/best_angle_model.pth")
    print(f"  Results saved to: models/training_results.png")


if __name__ == '__main__':
    main()
