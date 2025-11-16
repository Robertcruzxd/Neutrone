#!/usr/bin/env python3
"""
Train a neural network for complete drone tracking:
- Drone position (x, y)
- Number of propellers (0-4)
- Propeller positions (up to 4)
- Drone angle
- Tracking confidence

This is a multi-task learning model that learns all aspects of drone tracking.
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


class DroneTrackingDataset(Dataset):
    """Dataset for complete drone tracking with all labels."""
    
    def __init__(self, frames: List[Dict]):
        """
        Args:
            frames: List of frame metadata from dataset
        """
        self.frames = []
        
        # Filter valid frames
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
        
        # Input: Raw event statistics
        features = torch.tensor([
            stats['x_mean'],
            stats['y_mean'],
            stats['x_std'],
            stats['y_std'],
            stats['num_filtered_events'],
            stats.get('num_total_events', stats['num_filtered_events'] * 2),
            stats.get('tracking_confidence', 0.5),
        ], dtype=torch.float32)
        
        # Targets (multiple outputs)
        targets = {}
        
        # 1. Drone position (regression)
        targets['position'] = torch.tensor([
            stats['x_mean'],
            stats['y_mean']
        ], dtype=torch.float32)
        
        # 2. Number of propellers (classification: 0-4)
        num_props = min(stats.get('num_propellers', 0), 4)
        targets['num_propellers'] = torch.tensor(num_props, dtype=torch.long)
        
        # 3. Propeller positions (regression, 4 propellers × 2 coords = 8 values)
        propeller_coords = []
        propellers = stats.get('propellers', [])
        for i in range(4):
            if i < len(propellers):
                propeller_coords.extend([propellers[i]['x'], propellers[i]['y']])
            else:
                propeller_coords.extend([0.0, 0.0])  # Padding
        targets['propellers'] = torch.tensor(propeller_coords, dtype=torch.float32)
        
        # 4. Propeller validity mask (which propellers are real vs padding)
        propeller_mask = torch.zeros(4, dtype=torch.float32)
        propeller_mask[:len(propellers)] = 1.0
        targets['propeller_mask'] = propeller_mask
        
        # 5. Drone angle (regression)
        targets['angle'] = torch.tensor(frame['angle_deg'], dtype=torch.float32)
        
        # 6. Tracking confidence (regression)
        targets['confidence'] = torch.tensor(
            stats.get('tracking_confidence', 0.5), 
            dtype=torch.float32
        )
        
        return features, targets


class DroneTrackerModel(nn.Module):
    """
    Multi-task neural network for complete drone tracking.
    
    Outputs:
    - position: (x, y) centroid
    - num_propellers: Classification (0-4)
    - propellers: 8 values (4 propellers × (x,y))
    - angle: Drone angle in degrees
    - confidence: Tracking confidence (0-1)
    """
    
    def __init__(self, input_size: int = 7):
        super().__init__()
        
        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Task-specific heads
        
        # Position head (x, y)
        self.position_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # Propeller count classifier (0-4)
        self.num_props_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 5 classes: 0, 1, 2, 3, 4 propellers
        )
        
        # Propeller positions (8 values)
        self.propellers_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # 4 propellers × 2 coords
        )
        
        # Angle regression head
        self.angle_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Confidence regression head
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Confidence is 0-1
        )
    
    def forward(self, x):
        # Extract shared features
        features = self.backbone(x)
        
        # Generate all predictions
        outputs = {
            'position': self.position_head(features),
            'num_propellers': self.num_props_head(features),
            'propellers': self.propellers_head(features),
            'angle': self.angle_head(features).squeeze(-1),
            'confidence': self.confidence_head(features).squeeze(-1)
        }
        
        return outputs


def compute_loss(predictions, targets, propeller_mask):
    """
    Compute multi-task loss with appropriate weighting.
    
    Loss components:
    - Position: MSE loss on (x,y)
    - Num propellers: Cross-entropy classification
    - Propellers: MSE on valid propeller positions only
    - Angle: MSE loss
    - Confidence: MSE loss
    """
    loss_dict = {}
    
    # Position loss (L2)
    loss_dict['position'] = nn.functional.mse_loss(
        predictions['position'], 
        targets['position']
    )
    
    # Propeller count loss (cross-entropy)
    loss_dict['num_propellers'] = nn.functional.cross_entropy(
        predictions['num_propellers'],
        targets['num_propellers']
    )
    
    # Propeller positions loss (only for valid propellers)
    # Reshape: (batch, 8) -> (batch, 4, 2)
    pred_props = predictions['propellers'].view(-1, 4, 2)
    target_props = targets['propellers'].view(-1, 4, 2)
    mask = propeller_mask.unsqueeze(-1).expand_as(pred_props)  # (batch, 4, 2)
    
    # Masked MSE
    propeller_diff = (pred_props - target_props) ** 2
    masked_diff = propeller_diff * mask
    loss_dict['propellers'] = masked_diff.sum() / (mask.sum() + 1e-8)
    
    # Angle loss
    loss_dict['angle'] = nn.functional.mse_loss(
        predictions['angle'],
        targets['angle']
    )
    
    # Confidence loss
    loss_dict['confidence'] = nn.functional.mse_loss(
        predictions['confidence'],
        targets['confidence']
    )
    
    # Weighted total loss
    total_loss = (
        loss_dict['position'] * 1.0 +
        loss_dict['num_propellers'] * 2.0 +  # Classification is important
        loss_dict['propellers'] * 3.0 +       # Propellers are KEY
        loss_dict['angle'] * 1.5 +
        loss_dict['confidence'] * 0.5
    )
    
    loss_dict['total'] = total_loss
    return total_loss, loss_dict


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 150,
    lr: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the complete drone tracking model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_components = {k: 0.0 for k in ['position', 'num_propellers', 'propellers', 'angle', 'confidence']}
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            predictions = model(features)
            loss, loss_dict = compute_loss(predictions, targets, targets['propeller_mask'])
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            for k in train_loss_components.keys():
                train_loss_components[k] += loss_dict[k].item()
        
        train_loss /= len(train_loader)
        for k in train_loss_components.keys():
            train_loss_components[k] /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components = {k: 0.0 for k in ['position', 'num_propellers', 'propellers', 'angle', 'confidence']}
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                predictions = model(features)
                loss, loss_dict = compute_loss(predictions, targets, targets['propeller_mask'])
                
                val_loss += loss.item()
                for k in val_loss_components.keys():
                    val_loss_components[k] += loss_dict[k].item()
        
        val_loss /= len(val_loader)
        for k in val_loss_components.keys():
            val_loss_components[k] /= len(val_loader)
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
                'train_loss': train_loss,
            }, 'models/best_drone_tracker.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
            print(f"  Losses: pos={val_loss_components['position']:.3f} "
                  f"props={val_loss_components['propellers']:.3f} "
                  f"count={val_loss_components['num_propellers']:.3f} "
                  f"angle={val_loss_components['angle']:.3f} "
                  f"conf={val_loss_components['confidence']:.3f}")
    
    return train_losses, val_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu'):
    """Evaluate model and compute metrics for all tasks."""
    model = model.to(device)
    model.eval()
    
    all_preds = {
        'position': [], 'num_propellers': [], 'propellers': [],
        'angle': [], 'confidence': []
    }
    all_targets = {
        'position': [], 'num_propellers': [], 'propellers': [],
        'angle': [], 'confidence': []
    }
    all_masks = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            predictions = model(features)
            
            # Collect predictions
            all_preds['position'].append(predictions['position'].cpu().numpy())
            all_preds['num_propellers'].append(
                predictions['num_propellers'].argmax(dim=1).cpu().numpy()
            )
            all_preds['propellers'].append(predictions['propellers'].cpu().numpy())
            all_preds['angle'].append(predictions['angle'].cpu().numpy())
            all_preds['confidence'].append(predictions['confidence'].cpu().numpy())
            
            # Collect targets
            all_targets['position'].append(targets['position'].cpu().numpy())
            all_targets['num_propellers'].append(targets['num_propellers'].cpu().numpy())
            all_targets['propellers'].append(targets['propellers'].cpu().numpy())
            all_targets['angle'].append(targets['angle'].cpu().numpy())
            all_targets['confidence'].append(targets['confidence'].cpu().numpy())
            all_masks.append(targets['propeller_mask'].cpu().numpy())
    
    # Concatenate all batches
    for k in all_preds.keys():
        all_preds[k] = np.concatenate(all_preds[k])
        all_targets[k] = np.concatenate(all_targets[k])
    all_masks = np.concatenate(all_masks)
    
    # Compute metrics
    print(f"\n{'='*70}")
    print("Test Set Evaluation:")
    print(f"{'='*70}")
    
    # Position metrics
    pos_error = np.linalg.norm(all_preds['position'] - all_targets['position'], axis=1)
    print(f"\n📍 Position Tracking:")
    print(f"  Mean Error: {np.mean(pos_error):.2f} pixels")
    print(f"  Median Error: {np.median(pos_error):.2f} pixels")
    print(f"  Max Error: {np.max(pos_error):.2f} pixels")
    
    # Propeller count accuracy
    prop_count_acc = np.mean(all_preds['num_propellers'] == all_targets['num_propellers'])
    print(f"\n🔢 Propeller Count:")
    print(f"  Accuracy: {prop_count_acc*100:.1f}%")
    
    # Propeller position errors (only for valid propellers)
    pred_props = all_preds['propellers'].reshape(-1, 4, 2)
    target_props = all_targets['propellers'].reshape(-1, 4, 2)
    prop_errors = []
    for i in range(len(pred_props)):
        for j in range(4):
            if all_masks[i, j] > 0.5:  # Valid propeller
                error = np.linalg.norm(pred_props[i, j] - target_props[i, j])
                prop_errors.append(error)
    
    if prop_errors:
        print(f"\n🚁 Propeller Positions:")
        print(f"  Mean Error: {np.mean(prop_errors):.2f} pixels")
        print(f"  Median Error: {np.median(prop_errors):.2f} pixels")
    
    # Angle metrics
    angle_errors = np.abs(all_preds['angle'] - all_targets['angle'])
    print(f"\n📐 Angle Estimation:")
    print(f"  MAE: {np.mean(angle_errors):.2f}°")
    print(f"  RMSE: {np.sqrt(np.mean(angle_errors**2)):.2f}°")
    print(f"  Median Error: {np.median(angle_errors):.2f}°")
    
    # Confidence metrics
    conf_error = np.abs(all_preds['confidence'] - all_targets['confidence'])
    print(f"\n💯 Confidence Prediction:")
    print(f"  MAE: {np.mean(conf_error):.3f}")
    
    print(f"\n{'='*70}\n")
    
    return all_preds, all_targets


def plot_results(train_losses, val_losses, predictions, targets, output_dir: Path):
    """Plot comprehensive training results."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_losses, label='Train Loss', alpha=0.7)
    ax1.plot(val_losses, label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Position tracking
    ax2 = plt.subplot(2, 3, 2)
    pos_errors = np.linalg.norm(predictions['position'] - targets['position'], axis=1)
    ax2.hist(pos_errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(pos_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(pos_errors):.1f}px')
    ax2.set_xlabel('Position Error (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Position Tracking Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Propeller count confusion
    ax3 = plt.subplot(2, 3, 3)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets['num_propellers'], predictions['num_propellers'])
    im = ax3.imshow(cm, cmap='Blues')
    ax3.set_xlabel('Predicted Count')
    ax3.set_ylabel('True Count')
    ax3.set_title('Propeller Count Confusion Matrix')
    ax3.set_xticks(range(5))
    ax3.set_yticks(range(5))
    plt.colorbar(im, ax=ax3)
    
    # 4. Angle predictions
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(targets['angle'], predictions['angle'], alpha=0.4, s=20)
    ax4.plot([0, 90], [0, 90], 'r--', lw=2)
    ax4.set_xlabel('True Angle (°)')
    ax4.set_ylabel('Predicted Angle (°)')
    ax4.set_title('Angle Estimation')
    ax4.grid(True, alpha=0.3)
    
    # 5. Angle error distribution
    ax5 = plt.subplot(2, 3, 5)
    angle_errors = np.abs(predictions['angle'] - targets['angle'])
    ax5.hist(angle_errors, bins=50, alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(angle_errors), color='red', linestyle='--',
                label=f'MAE: {np.mean(angle_errors):.1f}°')
    ax5.set_xlabel('Angle Error (°)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Angle Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Confidence prediction
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(targets['confidence'], predictions['confidence'], alpha=0.4, s=20)
    ax6.plot([0, 1], [0, 1], 'r--', lw=2)
    ax6.set_xlabel('True Confidence')
    ax6.set_ylabel('Predicted Confidence')
    ax6.set_title('Confidence Estimation')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drone_tracker_results.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved training results to {output_dir / 'drone_tracker_results.png'}")
    plt.close()


def main():
    # Load all datasets from drone_dataset directory
    print("Loading datasets...")
    dataset_dir = Path('drone_dataset')
    all_frames = []
    
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}")
        print("Run ./scripts/generate_drone_dataset.sh first to create training data")
        return
    
    # Find all subdirectories with metadata.json
    for subdir in dataset_dir.iterdir():
        if subdir.is_dir():
            metadata_file = subdir / 'metadata.json'
            if metadata_file.exists():
                print(f"  Loading {subdir.name}...")
                with open(metadata_file) as f:
                    data = json.load(f)
                    all_frames.extend(data['frames'])
                print(f"    Added {len(data['frames'])} frames")
    
    if not all_frames:
        print("❌ Error: No training data found in drone_dataset/")
        print("Run ./scripts/generate_drone_dataset.sh first")
        return
    
    print(f"\nTotal frames loaded: {len(all_frames)}")
    
    # Create dataset
    full_dataset = DroneTrackingDataset(all_frames)
    
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
    model = DroneTrackerModel(input_size=7)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Train model
    print("\n" + "="*70)
    print("Starting Training - Complete Drone Tracker")
    print("="*70)
    train_losses, val_losses = train_model(
        train_loader, val_loader, model,
        epochs=150,
        lr=0.001
    )
    
    # Load best model
    checkpoint = torch.load('models/best_drone_tracker.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on test set
    predictions, targets = evaluate_model(model, test_loader)
    
    # Plot results
    plot_results(train_losses, val_losses, predictions, targets, Path('models'))
    
    print("✓ Training complete!")
    print(f"  Model saved to: models/best_drone_tracker.pth")
    print(f"  Results saved to: models/drone_tracker_results.png")


if __name__ == '__main__':
    main()
