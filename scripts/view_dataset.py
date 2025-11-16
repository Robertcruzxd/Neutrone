import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class DatasetViewer:
    """Interactive viewer for generated dataset with manual annotation support."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.frames_dir = self.dataset_dir / "frames"
        self.metadata_path = self.dataset_dir / "metadata.json"
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.frames = self.metadata["frames"]
        self.current_idx = 0
        self.modified = False
        
        print(f"Loaded dataset: {dataset_dir}")
        print(f"Total frames: {len(self.frames)}")
        print(f"Label: {self.metadata['recording_info']['label']}")
        
    def draw_overlay(self, frame: np.ndarray, frame_info: dict) -> np.ndarray:
        """Draw info overlay on frame."""
        overlay = frame.copy()
        
        # Frame info
        frame_id = frame_info["frame_id"]
        timestamp = frame_info["timestamp_s"]
        label = frame_info["label"]
        angle = frame_info["angle_deg"]
        
        # Event stats
        event_stats = frame_info.get("event_stats", {})
        num_events = event_stats.get("num_events", 0)
        
        # Draw background box for text
        cv2.rectangle(overlay, (5, 5), (400, 140), (0, 0, 0), -1)
        
        # Frame info text
        y_pos = 25
        texts = [
            f"Frame: {frame_id}/{len(self.frames)-1}",
            f"Time: {timestamp:.3f}s",
            f"Label: {label}",
            f"Angle: {angle:.1f}°",
            f"Events: {num_events}",
        ]
        
        for text in texts:
            cv2.putText(overlay, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_pos += 20
        
        # Controls
        cv2.putText(overlay, "Controls:", (10, y_pos + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        y_pos += 25
        controls = [
            "Arrow keys: Navigate",
            "A: Set angle",
            "S: Save changes",
            "Q: Quit"
        ]
        for ctrl in controls:
            cv2.putText(overlay, ctrl, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            y_pos += 15
        
        # Event centroid visualization
        if event_stats:
            cx = int(event_stats.get("x_mean", 0))
            cy = int(event_stats.get("y_mean", 0))
            x_std = int(event_stats.get("x_std", 0))
            y_std = int(event_stats.get("y_std", 0))
            
            # Draw centroid
            cv2.circle(overlay, (cx, cy), 5, (0, 255, 255), 2)
            
            # Draw bounding box if available (from propeller-based detection)
            if 'bbox' in event_stats:
                bbox = event_stats['bbox']
                x1 = int(bbox['x_min'])
                y1 = int(bbox['y_min'])
                x2 = int(bbox['x_max'])
                y2 = int(bbox['y_max'])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw propeller positions if detected
            if 'propellers' in event_stats:
                for i, prop in enumerate(event_stats['propellers']):
                    px = int(prop['x'])
                    py = int(prop['y'])
                    cv2.circle(overlay, (px, py), 8, (255, 0, 255), 2)
                    cv2.putText(overlay, f"P{i+1}", (px + 10, py - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Draw lines connecting propellers to show drone shape
                if len(event_stats['propellers']) >= 2:
                    propellers = event_stats['propellers']
                    for i in range(len(propellers)):
                        p1 = propellers[i]
                        p2 = propellers[(i + 1) % len(propellers)]
                        cv2.line(overlay, 
                                (int(p1['x']), int(p1['y'])),
                                (int(p2['x']), int(p2['y'])),
                                (255, 165, 0), 1)
            else:
                # Fallback: Draw spread ellipse if no propellers detected
                if x_std > 0 and y_std > 0:
                    cv2.ellipse(overlay, (cx, cy), (x_std * 2, y_std * 2), 
                               0, 0, 360, (255, 255, 0), 1)
        
        # Blend overlay
        alpha = 0.7
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def load_frame(self, idx: int) -> np.ndarray:
        """Load frame by index."""
        frame_info = self.frames[idx]
        frame_path = self.frames_dir / frame_info["filename"]
        frame = cv2.imread(str(frame_path))
        return self.draw_overlay(frame, frame_info)
    
    def set_angle(self, idx: int, angle: float) -> None:
        """Set angle for current frame."""
        self.frames[idx]["angle_deg"] = angle
        self.modified = True
        print(f"Set angle for frame {idx} to {angle:.1f}°")
    
    def save_metadata(self) -> None:
        """Save modified metadata."""
        if self.modified:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Saved metadata to {self.metadata_path}")
            self.modified = False
        else:
            print("No changes to save")
    
    def run(self) -> None:
        """Run interactive viewer."""
        window_name = "Dataset Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print("\n" + "="*60)
        print("Interactive Dataset Viewer")
        print("="*60)
        print("Controls:")
        print("  Left/Right Arrow: Navigate frames")
        print("  A: Annotate angle (enter value)")
        print("  S: Save changes")
        print("  Q/ESC: Quit")
        print("="*60 + "\n")
        
        while True:
            # Load and display current frame
            frame = self.load_frame(self.current_idx)
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            elif key == 83 or key == ord('d'):  # Right arrow or D
                self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
            
            elif key == 81 or key == ord('a'):  # Left arrow or A
                if key == ord('a'):
                    # Annotate angle
                    print(f"\nCurrent angle: {self.frames[self.current_idx]['angle_deg']:.1f}°")
                    try:
                        angle_input = input("Enter new angle (0-90): ")
                        angle = float(angle_input)
                        if 0 <= angle <= 90:
                            self.set_angle(self.current_idx, angle)
                        else:
                            print("Angle must be between 0 and 90")
                    except ValueError:
                        print("Invalid input")
                else:
                    self.current_idx = max(self.current_idx - 1, 0)
            
            elif key == ord('s'):  # S - Save
                self.save_metadata()
            
            elif key == ord(' '):  # Space - Jump forward
                self.current_idx = min(self.current_idx + 10, len(self.frames) - 1)
            
            elif key == ord('b'):  # B - Jump backward
                self.current_idx = max(self.current_idx - 10, 0)
        
        cv2.destroyAllWindows()
        
        # Save on exit if modified
        if self.modified:
            save = input("\nSave changes before exiting? (y/n): ")
            if save.lower() == 'y':
                self.save_metadata()


def compare_datasets(idle_dir: str, moving_dir: str) -> None:
    """Compare idle and moving datasets side by side."""
    idle_dir = Path(idle_dir)
    moving_dir = Path(moving_dir)
    
    # Load metadata
    with open(idle_dir / "metadata.json") as f:
        idle_meta = json.load(f)
    with open(moving_dir / "metadata.json") as f:
        moving_meta = json.load(f)
    
    print("\n" + "="*60)
    print("Dataset Comparison")
    print("="*60)
    
    print(f"\nIdle Dataset:")
    print(f"  Frames: {len(idle_meta['frames'])}")
    print(f"  Reference centroid: {idle_meta['reference_features']['centroid']}")
    print(f"  Reference spread: {idle_meta['reference_features']['spread']}")
    
    print(f"\nMoving Dataset:")
    print(f"  Frames: {len(moving_meta['frames'])}")
    
    # Angle distribution
    angles = [f["angle_deg"] for f in moving_meta["frames"]]
    print(f"  Angle range: {min(angles):.1f}° - {max(angles):.1f}°")
    print(f"  Mean angle: {np.mean(angles):.1f}°")
    print(f"  Std angle: {np.std(angles):.1f}°")
    
    print("="*60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View and annotate event camera dataset"
    )
    parser.add_argument(
        "dataset_dir",
        help="Path to dataset directory (containing frames/ and metadata.json)"
    )
    parser.add_argument(
        "--compare",
        help="Compare with another dataset (e.g., idle vs moving)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_datasets(args.dataset_dir, args.compare)
    
    # Run viewer
    viewer = DatasetViewer(args.dataset_dir)
    viewer.run()


if __name__ == "__main__":
    main()