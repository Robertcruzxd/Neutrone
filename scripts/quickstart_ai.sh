#!/bin/bash

echo "============================================"
echo "  Drone Angle Estimation - Quick Start"
echo "============================================"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
echo "----------------------------------------"

# Check if uv is available (better for Arch Linux)
if command -v uv &> /dev/null; then
    echo "✓ Using uv for package management"
    PYTHON_CMD="uv run python"
    PIP_CMD="uv pip install"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3 install"
fi

# Check if torch is installed
if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "⚠️  PyTorch not found. Installing dependencies..."
    echo ""
    echo "This will install: torch, torchvision, matplotlib"
    echo ""
    
    if command -v uv &> /dev/null; then
        uv pip install torch torchvision matplotlib
    else
        echo "Creating virtual environment..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install torch torchvision matplotlib
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        echo ""
        echo "Manual installation:"
        echo "  Option 1 (recommended): uv pip install torch torchvision matplotlib"
        echo "  Option 2: Create venv and install"
        exit 1
    fi
    echo "✓ Dependencies installed"
else
    echo "✓ PyTorch found"
fi
echo ""

# Step 1: Generate dataset
echo "Step 1/3: Generating labeled dataset..."
echo "----------------------------------------"
if [ ! -d "drone_dataset/moving" ]; then
    ./scripts/generate_drone_dataset.sh
    if [ $? -ne 0 ]; then
        echo "❌ Dataset generation failed"
        exit 1
    fi
else
    echo "✓ Dataset already exists (skipping)"
fi
echo ""

# Check dataset quality
echo "Dataset quality check:"
python3 check_tracking.py
echo ""

# Step 2: Train model
echo "Step 2/3: Training AI model..."
echo "----------------------------------------"
if [ ! -f "models/best_angle_model.pth" ]; then
    $PYTHON_CMD scripts/train_angle_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Training failed"
        exit 1
    fi
else
    echo "✓ Model already trained (skipping)"
    echo "  Delete models/best_angle_model.pth to retrain"
fi
echo ""

# Step 3: Run prediction demo
echo "Step 3/3: Running prediction on test recording..."
echo "----------------------------------------"
$PYTHON_CMD scripts/predict_angles.py src/evio/source/drone_moving.dat \
    --output predictions_demo.json \
    --delta 50 | head -30
echo ""
echo "... (output truncated)"
echo ""

# Summary
echo "============================================"
echo "✓ Quick start complete!"
echo "============================================"
echo ""
echo "Files created:"
echo "  📁 drone_dataset/          - Labeled training data"
echo "  🧠 models/best_angle_model.pth - Trained model"
echo "  📊 models/training_results.png - Training plots"
echo "  📝 predictions_demo.json   - Example predictions"
echo ""
echo "Next steps:"
echo "  1. View training results: open models/training_results.png"
echo "  2. View predictions: cat predictions_demo.json | jq ."
echo "  3. Predict on new files:"
echo "     python3 scripts/predict_angles.py YOUR_FILE.dat"
echo ""
echo "For more info, see: AI_TRAINING_GUIDE.md"
echo ""
