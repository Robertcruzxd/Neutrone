import json

# Check idle dataset
with open('drone_dataset/idle/metadata.json') as f:
    idle_meta = json.load(f)

print("=" * 60)
print("IDLE DRONE DATASET")
print("=" * 60)
print(f"\nReference centroid: ({idle_meta['reference_features']['centroid']['x']:.1f}, "
      f"{idle_meta['reference_features']['centroid']['y']:.1f})")
print(f"Reference spread: ({idle_meta['reference_features']['spread']['x']:.1f}, "
      f"{idle_meta['reference_features']['spread']['y']:.1f})")

print("\nFirst 3 frames:")
for i in range(min(3, len(idle_meta['frames']))):
    f = idle_meta['frames'][i]
    stats = f['event_stats']
    print(f"\nFrame {i}:")
    print(f"  Centroid: ({stats['x_mean']:.1f}, {stats['y_mean']:.1f})")
    print(f"  Angle: {f['angle_deg']:.2f}°")
    print(f"  Filtered events: {stats['num_filtered_events']}/{stats['num_events']}")
    
    if 'num_propellers' in stats:
        print(f"  Propellers detected: {stats['num_propellers']}")
        if 'propellers' in stats:
            for j, p in enumerate(stats['propellers']):
                print(f"    P{j+1}: ({p['x']:.1f}, {p['y']:.1f})")
    
    if 'bbox' in stats:
        bbox = stats['bbox']
        print(f"  BBox: [{bbox['x_min']:.0f}, {bbox['y_min']:.0f}] to [{bbox['x_max']:.0f}, {bbox['y_max']:.0f}]")

# Check moving dataset
with open('drone_dataset/moving/metadata.json') as f:
    moving_meta = json.load(f)

print("\n" + "=" * 60)
print("MOVING DRONE DATASET")
print("=" * 60)

angles = [f['angle_deg'] for f in moving_meta['frames']]
print(f"\nTotal frames: {len(moving_meta['frames'])}")
print(f"Angle range: {min(angles):.1f}° to {max(angles):.1f}°")
print(f"Mean angle: {sum(angles)/len(angles):.1f}°")

print("\nSample frames with different angles:")
# Show frames at different positions
for idx in [0, len(moving_meta['frames'])//4, len(moving_meta['frames'])//2, 
            3*len(moving_meta['frames'])//4, len(moving_meta['frames'])-1]:
    f = moving_meta['frames'][idx]
    stats = f['event_stats']
    print(f"\nFrame {idx} (t={f['timestamp_s']:.1f}s):")
    print(f"  Centroid: ({stats['x_mean']:.1f}, {stats['y_mean']:.1f})")
    print(f"  Angle: {f['angle_deg']:.2f}°")
    if 'num_propellers' in stats:
        print(f"  Propellers: {stats['num_propellers']}")
