#!/usr/bin/env python3
"""
Real-time visualization of AI-powered drone tracking.

This script processes .dat recordings and visualizes:
- Drone detection and tracking
- Individual propeller positions
- Estimated angle
- Tracking confidence
- Event camera feed
"""

import json
import numpy as np
from pathlib import Path
import torch
import argparse
import cv2
import sys
sys.path.append(str(Path(__file__).parent))

from train_drone_tracker import DroneTrackerModel
from generate_dataset import DatasetGenerator


def load_tracker_model(model_path: str, device: str = 'cpu'):
    """Load trained drone tracker model."""
    model = DroneTrackerModel(input_size=7)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def create_visualization(
    frame: np.ndarray,
    drone_info: dict,
    predictions: dict,
    frame_idx: int,
    timestamp: float,
    width: int = 1280,
    height: int = 720
):
    """
    Create comprehensive visualization overlay.
    
    Args:
        frame: Base event frame (grayscale)
        drone_info: Detection info from DatasetGenerator
        predictions: AI model predictions
        frame_idx: Frame number
        timestamp: Timestamp in seconds
        width, height: Frame dimensions
    """
    # Convert grayscale to BGR for colored overlays
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Check if drone detected
    if drone_info.get('out_of_frame', False):
        # Out of frame indicator
        cv2.putText(vis, "DRONE OUT OF FRAME", (width//2 - 200, height//2),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(vis, f"Frame {frame_idx} | t={timestamp:.2f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis
    
    # Get predicted values
    pos_x, pos_y = predictions['position']
    num_props = predictions['num_propellers']
    angle = predictions['angle']
    confidence = predictions['confidence']
    propellers = predictions['propellers'].reshape(4, 2)
    
    # 1. Draw drone bounding box
    bbox = drone_info.get('bbox')
    if bbox:
        x1, y1, x2, y2 = bbox
        # Draw bbox with color based on confidence
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    
    # 2. Draw drone centroid (predicted)
    center = (int(pos_x), int(pos_y))
    cv2.circle(vis, center, 8, (0, 255, 255), -1)  # Yellow
    cv2.circle(vis, center, 12, (0, 255, 255), 2)
    
    # 5. Overlay statistics panel (top-left)
    panel_height = 180
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (350, panel_height), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    
    y_offset = 25
    line_height = 25
    
    cv2.putText(vis, f"Frame: {frame_idx} | Time: {timestamp:.2f}s", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(vis, f"Position: ({int(pos_x)}, {int(pos_y)})", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    y_offset += line_height
    
    cv2.putText(vis, f"Propellers: {num_props}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 1)
    y_offset += line_height
    
    # Confidence bar
    conf_text = f"Confidence: {confidence:.2f}"
    cv2.putText(vis, conf_text, (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += line_height
    
    # Draw confidence bar
    bar_width = 200
    bar_height = 15
    bar_x, bar_y = 10, y_offset - 10
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (100, 100, 100), -1)
    conf_bar_width = int(bar_width * confidence)
    conf_color = (0, int(255 * confidence), int(255 * (1 - confidence)))
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + conf_bar_width, bar_y + bar_height),
                 conf_color, -1)
    
    return vis


def visualize_tracking(
    model: torch.nn.Module,
    dat_file: Path,
    output_video: Path = None,
    delta_ms: float = 50.0,
    device: str = 'cpu',
    show_live: bool = True,
    save_frames: bool = False
):
    """
    Process recording and visualize AI tracking in real-time.
    
    Args:
        model: Trained drone tracker model
        dat_file: Path to .dat event recording
        output_video: Optional path to save output video
        delta_ms: Time interval between frames (milliseconds)
        device: Device to run inference on
        show_live: Show live OpenCV window
        save_frames: Save individual frames
    """
    from evio.source.dat_file import DatFileSource
    import numpy as np
    
    print(f"\n{'='*70}")
    print(f"🚁 AI-Powered Drone Tracking Visualization")
    print(f"{'='*70}")
    print(f"Recording: {dat_file}")
    print(f"Frame interval: {delta_ms}ms")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Helper function
    def get_window(event_words, time_order, win_start, win_stop):
        event_indexes = time_order[win_start:win_stop]
        words = event_words[event_indexes].astype(np.uint32, copy=False)
        x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
        y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
        return x_coords, y_coords
    
    # Load source
    src = DatFileSource(
        str(dat_file),
        width=1280,
        height=720,
        window_length_us=int(delta_ms * 1000)
    )
    
    # Create dataset generator for detection
    temp_dataset = DatasetGenerator(
        output_dir=Path("/tmp/temp_vis"),
        delta_ms=delta_ms
    )
    
    # Setup video writer if requested
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1000.0 / delta_ms  # Convert ms to fps
        video_writer = cv2.VideoWriter(
            str(output_video), fourcc, fps, (1280, 720)
        )
        print(f"📹 Recording to: {output_video}")
    
    # Setup frame saving if requested
    frames_dir = None
    if save_frames:
        frames_dir = dat_file.parent / f"{dat_file.stem}_tracking_frames"
        frames_dir.mkdir(exist_ok=True)
        print(f"💾 Saving frames to: {frames_dir}")
    
    frame_idx = 0
    
    print("\nProcessing... (Press 'q' to quit, 'p' to pause)\n")
    
    paused = False
    
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
        
        # Create event visualization (base frame)
        event_frame = np.zeros((720, 1280), dtype=np.uint8)
        if len(x_coords) > 0:
            # Clip coordinates to frame bounds
            x_valid = np.clip(x_coords, 0, 1279)
            y_valid = np.clip(y_coords, 0, 719)
            event_frame[y_valid, x_valid] = 255
        
        # Detect drone region
        filtered_x, filtered_y, drone_info = temp_dataset._detect_drone_region(
            x_coords, y_coords
        )
        
        # Prepare features for AI model
        if drone_info.get('out_of_frame', False) or len(filtered_x) == 0:
            # Out of frame - use dummy predictions
            predictions = {
                'position': np.array([640.0, 360.0]),
                'num_propellers': 0,
                'propellers': np.zeros((4, 2)),
                'angle': 0.0,
                'confidence': 0.0
            }
            detection_confidence = 0.0
        else:
            # Extract features
            x_mean = np.mean(filtered_x)
            y_mean = np.mean(filtered_y)
            x_std = np.std(filtered_x)
            y_std = np.std(filtered_y)
            num_events = len(filtered_x)
            num_total = len(x_coords)
            
            # Get detection confidence (from detection algorithm, not AI model)
            detection_confidence = drone_info.get('tracking_confidence', 0.5)
            
            # Create feature tensor
            features = torch.tensor([
                x_mean, y_mean, x_std, y_std, 
                num_events, num_total, detection_confidence
            ], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Run AI inference
            with torch.no_grad():
                outputs = model(features)
                predictions = {
                    'position': outputs['position'][0].cpu().numpy(),
                    'num_propellers': outputs['num_propellers'][0].argmax().item(),
                    'propellers': outputs['propellers'][0].cpu().numpy(),
                    'angle': outputs['angle'][0].item(),
                    # Use detection confidence, not model prediction (model just echoes input)
                    'confidence': detection_confidence
                }
        
        # Create visualization
        vis_frame = create_visualization(
            event_frame, drone_info, predictions,
            frame_idx, t_seconds
        )
        
        # Save to video
        if video_writer:
            video_writer.write(vis_frame)
        
        # Save frame
        if frames_dir:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), vis_frame)
        
        # Show live
        if show_live:
            cv2.imshow('AI Drone Tracking', vis_frame)
            
            # Handle key presses
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopped by user")
                break
            elif key == ord('p'):
                paused = not paused
                status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
                print(f"\r{status}", end='', flush=True)
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"screenshot_{frame_idx:06d}.png"
                cv2.imwrite(screenshot_path, vis_frame)
                print(f"\n📸 Screenshot saved: {screenshot_path}")
        
        # Progress
        if frame_idx % 20 == 0:
            status = f"Frame {frame_idx:4d} | t={t_seconds:6.2f}s | "
            if not drone_info.get('out_of_frame', False):
                status += f"Props={predictions['num_propellers']} | "
                status += f"Conf={predictions['confidence']:.2f}"
            else:
                status += "OUT OF FRAME"
            print(status)
        
        frame_idx += 1
    
    # Cleanup
    if video_writer:
        video_writer.release()
        print(f"\n✓ Video saved: {output_video}")
    
    if show_live:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print(f"✓ Processed {frame_idx} frames")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize AI-powered drone tracking on event recordings'
    )
    parser.add_argument('dat_file', type=Path, help='Input .dat file')
    parser.add_argument('--model', type=Path, default='models/best_drone_tracker.pth',
                       help='Path to trained model (default: models/best_drone_tracker.pth)')
    parser.add_argument('--output', type=Path, 
                       help='Output video file (e.g., tracking.mp4)')
    parser.add_argument('--delta', type=float, default=50.0,
                       help='Frame interval in ms (default: 50.0)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda (default: cpu)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable live display window')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save individual frames')
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.dat_file.exists():
        print(f"❌ Error: Input file not found: {args.dat_file}")
        return 1
    
    if not args.model.exists():
        print(f"❌ Error: Model not found: {args.model}")
        print(f"Train a model first: python scripts/train_drone_tracker.py")
        return 1
    
    # Load model
    print(f"Loading AI model from {args.model}...")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    model = load_tracker_model(str(args.model), device=device)
    print(f"✓ Model loaded on {device}")
    
    # Set default output video path
    output_video = args.output or args.dat_file.parent / f"{args.dat_file.stem}_tracking.mp4"
    
    # Run visualization
    visualize_tracking(
        model=model,
        dat_file=args.dat_file,
        output_video=output_video,
        delta_ms=args.delta,
        device=device,
        show_live=not args.no_display,
        save_frames=args.save_frames
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
