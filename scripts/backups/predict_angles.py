#!/usr/bin/env python3
"""
Use trained model to predict drone angles from new event camera recordings.
"""

import json
import numpy as np
from pathlib import Path
import torch
import argparse
import sys
sys.path.append(str(Path(__file__).parent))

from train_angle_model import AngleEstimationModel, EventDataset
from generate_dataset import DatasetGenerator


def load_model(model_path: str, device: str = 'cpu'):
    """Load trained model."""
    model = AngleEstimationModel(input_size=15)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_from_recording(
    model: torch.nn.Module,
    dat_file: Path,
    output_file: Path = None,
    delta_ms: float = 50.0,
    device: str = 'cpu'
):
    """
    Process a new .dat recording and predict angles for each frame.
    
    Args:
        model: Trained angle estimation model
        dat_file: Path to .dat event recording
        output_file: Optional path to save predictions JSON
        delta_ms: Time interval between frames (milliseconds)
        device: Device to run inference on
    """
    from evio.source.dat_file import DatFileSource
    from evio.core.pacer import Pacer
    import numpy as np
    
    # Helper function to extract event window
    def get_window(event_words, time_order, win_start, win_stop):
        event_indexes = time_order[win_start:win_stop]
        words = event_words[event_indexes].astype(np.uint32, copy=False)
        x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
        y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
        return x_coords, y_coords
    
    print(f"\n{'='*60}")
    print(f"Processing: {dat_file}")
    print(f"Frame interval: {delta_ms}ms")
    print(f"{'='*60}\n")
    
    # Load source
    src = DatFileSource(
        str(dat_file),
        width=1280,
        height=720,
        window_length_us=int(delta_ms * 1000)
    )
    
    # Load source
    src = DatFileSource(
        str(dat_file),
        width=1280,
        height=720,
        window_length_us=int(delta_ms * 1000)
    )
    
    # Create temporary dataset generator for detection
    temp_dataset = DatasetGenerator(
        output_dir=Path("/tmp/temp_predictions"),
        delta_ms=delta_ms
    )
    
    predictions = []
    frame_idx = 0
    
    # Process frames
    for batch_range in src.ranges():
        t_center_us = (batch_range.start_ts_us + batch_range.end_ts_us) / 2
        t_seconds = t_center_us / 1e6
        
        # Extract window
        x_coords, y_coords = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop
        )
        
        # Detect drone region
        filtered_x, filtered_y, drone_info = temp_dataset._detect_drone_region(x_coords, y_coords)
        
        # Check if drone is in frame
        if drone_info.get('out_of_frame', False):
            predictions.append({
                'frame_id': frame_idx,
                'timestamp_s': t_seconds,
                'status': 'out_of_frame',
                'angle_deg': None,
                'confidence': 0.0
            })
            print(f"Frame {frame_idx:3d} @ {t_seconds:6.2f}s: OUT OF FRAME")
            frame_idx += 1
            continue
        
        if len(filtered_x) == 0:
            predictions.append({
                'frame_id': frame_idx,
                'timestamp_s': t_seconds,
                'status': 'no_detection',
                'angle_deg': None,
                'confidence': 0.0
            })
            print(f"Frame {frame_idx:3d} @ {t_seconds:6.2f}s: NO DETECTION")
            frame_idx += 1
            continue
        
        # Prepare features for model
        x_mean = np.mean(filtered_x)
        y_mean = np.mean(filtered_y)
        x_std = np.std(filtered_x)
        y_std = np.std(filtered_y)
        num_events = len(filtered_x)
        confidence = drone_info.get('tracking_confidence', 0.5)
        num_props = drone_info.get('num_propellers', 0)
        
        # Propeller features
        propellers = drone_info.get('propellers', [])
        propeller_features = []
        for i in range(4):
            if i < len(propellers):
                propeller_features.extend([propellers[i][0], propellers[i][1]])
            else:
                propeller_features.extend([0.0, 0.0])
        
        # Create feature tensor
        features = torch.tensor([
            x_mean, y_mean, x_std, y_std, num_events, confidence, num_props
        ] + propeller_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict angle
        with torch.no_grad():
            angle_pred = model(features).item()
        
        predictions.append({
            'frame_id': frame_idx,
            'timestamp_s': t_seconds,
            'status': 'detected',
            'angle_deg': angle_pred,
            'confidence': confidence,
            'num_propellers': num_props,
            'centroid': {'x': float(x_mean), 'y': float(y_mean)}
        })
        
        print(f"Frame {frame_idx:3d} @ {t_seconds:6.2f}s: Angle = {angle_pred:6.2f}° "
              f"(conf={confidence:.2f}, props={num_props})")
        
        frame_idx += 1
    
    # Save predictions
    if output_file:
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, 'w') as f:
            json.dump({
                'source_file': str(dat_file),
                'delta_ms': delta_ms,
                'total_frames': len(predictions),
                'predictions': predictions
            }, f, indent=2)
        print(f"\n✓ Saved predictions to {output_file}")
    
    # Print summary
    detected = sum(1 for p in predictions if p['status'] == 'detected')
    out_of_frame = sum(1 for p in predictions if p['status'] == 'out_of_frame')
    no_detection = sum(1 for p in predictions if p['status'] == 'no_detection')
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total frames: {len(predictions)}")
    print(f"  Detected: {detected} ({detected/len(predictions)*100:.1f}%)")
    print(f"  Out of frame: {out_of_frame} ({out_of_frame/len(predictions)*100:.1f}%)")
    print(f"  No detection: {no_detection} ({no_detection/len(predictions)*100:.1f}%)")
    
    if detected > 0:
        angles = [p['angle_deg'] for p in predictions if p['status'] == 'detected']
        print(f"  Angle range: {min(angles):.1f}° to {max(angles):.1f}°")
        print(f"  Mean angle: {np.mean(angles):.1f}°")
    print(f"{'='*60}\n")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict drone angles from event recordings')
    parser.add_argument('dat_file', type=Path, help='Input .dat file')
    parser.add_argument('--model', type=Path, default='models/best_angle_model.pth',
                       help='Path to trained model (default: models/best_angle_model.pth)')
    parser.add_argument('--output', type=Path, help='Output JSON file for predictions')
    parser.add_argument('--delta', type=float, default=50.0,
                       help='Time interval between frames in ms (default: 50.0)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use: cpu or cuda (default: cpu)')
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.dat_file.exists():
        print(f"Error: Input file not found: {args.dat_file}")
        return 1
    
    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        print(f"Please train a model first using: python scripts/train_angle_model.py")
        return 1
    
    # Load model
    print(f"Loading model from {args.model}...")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    model = load_model(str(args.model), device=device)
    
    # Set default output path
    output_file = args.output or args.dat_file.parent / f"{args.dat_file.stem}_predictions.json"
    
    # Run predictions
    predictions = predict_from_recording(
        model=model,
        dat_file=args.dat_file,
        output_file=output_file,
        delta_ms=args.delta,
        device=device
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
