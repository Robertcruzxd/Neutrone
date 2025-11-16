#!/usr/bin/env python3
"""
Test propeller detection improvements on existing dataset.
Shows frame-by-frame propeller counts and detection quality.
"""

import json
import sys
from pathlib import Path

def analyze_propeller_detection(metadata_path: str):
    """Analyze propeller detection quality."""
    with open(metadata_path) as f:
        data = json.load(f)
    
    frames = data['frames']
    
    # Statistics
    total_frames = len(frames)
    frames_with_props = []
    prop_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, '5+': 0}
    out_of_frame = 0
    
    print(f"\n{'='*70}")
    print(f"Propeller Detection Analysis: {metadata_path}")
    print(f"{'='*70}\n")
    
    for frame in frames:
        drone_info = frame.get('drone_info', {})
        
        if drone_info.get('out_of_frame'):
            out_of_frame += 1
            continue
        
        num_props = drone_info.get('num_propellers', 0)
        if num_props > 0:
            frames_with_props.append(frame['frame_id'])
        
        if num_props >= 5:
            prop_counts['5+'] += 1
        else:
            prop_counts[num_props] += 1
    
    # Print summary
    tracked_frames = total_frames - out_of_frame
    print(f"Total frames: {total_frames}")
    print(f"Out of frame: {out_of_frame} ({out_of_frame/total_frames*100:.1f}%)")
    print(f"Tracked frames: {tracked_frames} ({tracked_frames/total_frames*100:.1f}%)")
    print(f"\nPropeller Detection Distribution:")
    print(f"  0 propellers: {prop_counts[0]} frames ({prop_counts[0]/tracked_frames*100:.1f}%)")
    print(f"  1 propeller:  {prop_counts[1]} frames ({prop_counts[1]/tracked_frames*100:.1f}%)")
    print(f"  2 propellers: {prop_counts[2]} frames ({prop_counts[2]/tracked_frames*100:.1f}%)")
    print(f"  3 propellers: {prop_counts[3]} frames ({prop_counts[3]/tracked_frames*100:.1f}%)")
    print(f"  4 propellers: {prop_counts[4]} frames ({prop_counts[4]/tracked_frames*100:.1f}%)")
    print(f"  5+ propellers: {prop_counts['5+']} frames ({prop_counts['5+']/tracked_frames*100:.1f}%)")
    
    # Quality metrics
    good_detection = prop_counts[3] + prop_counts[4] + prop_counts['5+']
    print(f"\n✓ Good detection (3+ propellers): {good_detection} frames ({good_detection/tracked_frames*100:.1f}%)")
    
    weak_detection = prop_counts[1]
    print(f"⚠ Weak detection (1 propeller): {weak_detection} frames ({weak_detection/tracked_frames*100:.1f}%)")
    
    # Show sample frames with different propeller counts
    print(f"\n{'='*70}")
    print("Sample frames by propeller count:")
    print(f"{'='*70}")
    
    for count in [0, 1, 2, 3, 4]:
        sample_frames = [f['frame_id'] for f in frames 
                        if not f.get('drone_info', {}).get('out_of_frame') 
                        and f.get('drone_info', {}).get('num_propellers', 0) == count][:5]
        if sample_frames:
            print(f"{count} propellers: frames {sample_frames}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'total': total_frames,
        'tracked': tracked_frames,
        'out_of_frame': out_of_frame,
        'prop_counts': prop_counts,
        'good_detection_rate': good_detection/tracked_frames if tracked_frames > 0 else 0
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_propeller_detection.py <metadata.json>")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    analyze_propeller_detection(metadata_path)
