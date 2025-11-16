#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚁 Complete Drone Tracking AI - Quick Demo                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if model exists
if [ ! -f "models/best_drone_tracker.pth" ]; then
    echo "⚠️  Model not found. Training first..."
    echo ""
    python scripts/train_drone_tracker.py
    if [ $? -ne 0 ]; then
        echo "❌ Training failed"
        exit 1
    fi
    echo ""
fi

echo "✓ Model loaded"
echo ""

# Check if recording exists
RECORDING="src/evio/source/drone_moving.dat"
if [ ! -f "$RECORDING" ]; then
    echo "❌ Recording not found: $RECORDING"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Starting Live Visualization                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Controls:"
echo "  'q' - Quit"
echo "  'p' - Pause/Resume"
echo "  's' - Save screenshot"
echo ""
echo "Press any key to start..."
read -n 1 -s

# Run visualization
uv run python scripts/visualize_tracking.py "$RECORDING" \
    --output demo_tracking.mp4 \
    --delta 50

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Demo Complete!                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Output video: demo_tracking.mp4"
echo ""
echo "Next steps:"
echo "  1. View results: models/drone_tracker_results.png"
echo "  2. Play video: mpv demo_tracking.mp4"
echo "  3. Try your own: python scripts/visualize_tracking.py YOUR_FILE.dat"
echo ""
