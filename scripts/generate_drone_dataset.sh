#!/bin/bash

# Flexible script to generate drone dataset from any .dat recording
# Usage: ./generate_drone_dataset.sh [path/to/recording.dat] [label] [reference_metadata.json]
# 
# Examples:
#   ./generate_drone_dataset.sh my_flight.dat moving
#   ./generate_drone_dataset.sh recordings/test.dat test drone_dataset/idle/metadata.json
#   ./generate_drone_dataset.sh src/evio/source/drone_idle.dat idle

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "Drone Dataset Generation Pipeline"
echo "============================================"
echo ""

# Parse arguments
DAT_FILE="${1:-src/evio/source/drone_moving.dat}"
LABEL="${2:-unlabeled}"
REFERENCE_METADATA="${3:-}"

# Validate input file
if [ ! -f "$DAT_FILE" ]; then
    echo "❌ Error: File not found: $DAT_FILE"
    echo ""
    echo "Usage: $0 [recording.dat] [label] [reference_metadata.json]"
    echo ""
    echo "Examples:"
    echo "  $0 my_flight.dat moving"
    echo "  $0 recordings/test.dat test drone_dataset/idle/metadata.json"
    exit 1
fi

# Extract filename without extension for output directory
BASENAME=$(basename "$DAT_FILE" .dat)
OUTPUT_DIR="./drone_dataset/$BASENAME"

# Configuration
DELTA_MS=50  # Capture every 50ms for smoother tracking

echo -e "${BLUE}Input:${NC}"
echo "  Recording: $DAT_FILE"
echo "  Label: $LABEL"
echo "  Output: $OUTPUT_DIR"
if [ -n "$REFERENCE_METADATA" ]; then
    echo "  Reference: $REFERENCE_METADATA"
fi
echo ""
echo -e "${GREEN}Features:${NC}"
echo "  ✓ Density-based drone detection"
echo "  ✓ Background noise filtering"
echo "  ✓ Propeller tracking (improved!)"
echo "  ✓ Angle estimation"
echo ""

# Build command
CMD="uv run python scripts/generate_dataset.py \"$DAT_FILE\" --output \"$OUTPUT_DIR\" --delta $DELTA_MS --label \"$LABEL\""

# Add reference if provided
if [ -n "$REFERENCE_METADATA" ]; then
    if [ -f "$REFERENCE_METADATA" ]; then
        CMD="$CMD --reference \"$REFERENCE_METADATA\""
        echo -e "${YELLOW}Using reference for angle calculation${NC}"
    else
        echo -e "${YELLOW}Warning: Reference file not found: $REFERENCE_METADATA${NC}"
        echo -e "${YELLOW}Continuing without reference...${NC}"
    fi
fi

echo ""
echo "Processing..."
echo ""

# Execute
eval $CMD

echo ""
echo -e "${GREEN}✓ Dataset generated successfully!${NC}"
echo ""
echo "Output:"
echo "  Frames: $OUTPUT_DIR/frames/"
echo "  Metadata: $OUTPUT_DIR/metadata.json"
echo ""
echo "Next steps:"
echo "  1. Train AI: uv run python scripts/train_drone_tracker.py"
echo "  2. Visualize: uv run python scripts/visualize_tracking.py \"$DAT_FILE\""
echo ""
