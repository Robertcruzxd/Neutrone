"""Visualize event activity map to debug drone detection."""

import numpy as np
import cv2
from evio.source.dat_file import DatFileSource

# Load first batch
src = DatFileSource(
    "src/evio/source/drone_idle.dat", 
    width=1280, 
    height=720, 
    window_length_us=10 * 1000
)

for batch_range in src.ranges():
    event_indexes = src.order[batch_range.start:batch_range.stop]
    words = src.event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities = ((words >> 28) & 0xF) > 0
    
    print(f"Total events: {len(x_coords)}")
    print(f"X range: [{np.min(x_coords)}, {np.max(x_coords)}]")
    print(f"Y range: [{np.min(y_coords)}, {np.max(y_coords)}]")
    
    # Create activity heatmap
    grid_size = 50
    x_bins = np.arange(0, 1280 + grid_size, grid_size)
    y_bins = np.arange(0, 720 + grid_size, grid_size)
    
    activity_map, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
    
    # Find max activity cell
    max_idx = np.unravel_index(np.argmax(activity_map), activity_map.shape)
    max_activity = activity_map[max_idx]
    center_x = x_bins[max_idx[0]] + grid_size / 2
    center_y = y_bins[max_idx[1]] + grid_size / 2
    
    print(f"\nMax activity: {max_activity} events in cell at ({center_x:.0f}, {center_y:.0f})")
    
    # Create visualization
    vis = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw events (sample for performance)
    for x, y, pol in zip(x_coords[::20], y_coords[::20], polarities[::20]):
        if 0 <= x < 1280 and 0 <= y < 720:
            color = (255, 255, 255) if pol else (100, 100, 100)
            cv2.circle(vis, (int(x), int(y)), 1, color, -1)
    
    # Overlay activity heatmap
    heatmap = np.zeros((720, 1280), dtype=np.float32)
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            x1, x2 = int(x_bins[i]), int(x_bins[i+1])
            y1, y2 = int(y_bins[j]), int(y_bins[j+1])
            if i < activity_map.shape[0] and j < activity_map.shape[1]:
                heatmap[y1:y2, x1:x2] = activity_map[i, j]
    
    # Normalize and colorize heatmap
    heatmap_norm = (heatmap / (np.max(heatmap) + 1e-6) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Blend
    vis = cv2.addWeighted(vis, 0.7, heatmap_color, 0.3, 0)
    
    # Mark max activity center
    cv2.circle(vis, (int(center_x), int(center_y)), 15, (0, 255, 0), 3)
    cv2.circle(vis, (int(center_x), int(center_y)), 200, (0, 255, 0), 2)  # ROI
    cv2.putText(vis, f"MAX: {int(max_activity)} events", 
                (int(center_x) - 80, int(center_y) - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show top 5 activity regions
    flat_activity = activity_map.flatten()
    top_5_indices = np.argsort(flat_activity)[-5:][::-1]
    
    print("\nTop 5 activity regions:")
    for rank, idx in enumerate(top_5_indices, 1):
        i, j = np.unravel_index(idx, activity_map.shape)
        cx = x_bins[i] + grid_size / 2
        cy = y_bins[j] + grid_size / 2
        activity = activity_map[i, j]
        print(f"  {rank}. ({cx:.0f}, {cy:.0f}): {int(activity)} events")
        
        # Draw on visualization
        if rank > 1:  # Skip #1 (already drawn in green)
            cv2.circle(vis, (int(cx), int(cy)), 10, (255, 0, 255), 2)
            cv2.putText(vis, f"#{rank}", (int(cx) + 15, int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Save
    cv2.imwrite("activity_heatmap.png", vis)
    print(f"\nSaved visualization to activity_heatmap.png")
    print("Green circle = highest activity (detected drone)")
    print("Magenta circles = other high-activity regions")
    
    break
