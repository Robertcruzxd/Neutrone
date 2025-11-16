# NEUTRONE 🚁⚡

**Neutralizing Drones Through Event-Based Intelligence**

*Developed by Yass from team Blacklist*  
*Junction Hackathon 2025 - Espoo, Finland*  
*SensoFusion Challenge - Military Applications*

---

## 🎯 Mission

NEUTRONE is a military-grade drone detection and neutralization system designed for hostile environments where traditional RGB cameras fail. Using cutting-edge **event camera technology** and **AI-powered tracking**, we provide:

- ⚡ **Microsecond-precision RPM detection** 
- 🎯 **Autonomous drone tracking** without manual labeling
- 🧠 **Real-time AI inference** (30+ FPS on CPU, 100+ FPS on GPU)
- 🌙 **Works in complete darkness** - event cameras detect motion, not light
- 💪 **Robust against countermeasures** - no RGB, no thermal signature needed

Built on the **evio** event camera library by [Ahti Helminen](https://github.com/helminen).

---

## 🔬 Technology Stack

### 1. **FFT-Based RPM Detection** 🎵

**The Problem:** Detect propeller rotation speed without knowing blade count or requiring calibration.

**The Solution:** Event cameras detect brightness changes at microsecond precision. When propellers spin, they create periodic patterns of events. We apply **Fast Fourier Transform (FFT)** to decompose this signal into frequency components—the dominant frequency directly corresponds to rotation rate.

**How It Works:**
```python
# Collect event counts over 2-second window
event_history = [(time, num_events), ...]

# Apply FFT to find dominant frequency
fft_result = rfft(event_counts)
peak_frequency = find_strongest_frequency(fft_result)  # in Hz

# Convert to RPM
rpm = peak_frequency × 60
```

**Example:** A propeller at 1200 RPM creates a 20 Hz oscillation (1200 ÷ 60). FFT detects this frequency regardless of blade count—we measure the **rotation frequency of the entire propeller**, not individual blades.

**Performance:**
- ✅ No calibration needed
- ✅ Confidence scoring based on signal strength
- ✅ Handles varying speeds with fast updates

---

### 2. **Autonomous Dataset Generation** 📊

**The Problem:** Manual labeling of drone positions, propeller locations, and angles is time-consuming and error-prone.

**The Solution:** Fully automated dataset generation that exploits the unique signature of spinning propellers—extremely dense, tightly clustered black pixels that create distinct connected regions.

**Algorithm:**

1. **Propeller-First Detection**
   - Scan frame with 8px grid to find ultra-dense event clusters
   - Validate using multi-level tightness metrics (10-25 events within 8-12px)
   - Apply non-maximum suppression to keep best 4-6 propeller candidates

2. **Drone Inference**
   - Group propellers by proximity (20-120px separation)
   - Score configurations: **20x bonus for 4 separated propellers**
   - Reject distant objects (airplanes, birds) via size filtering
   - Filter sensor noise zones automatically

3. **Temporal Tracking**
   - Maintain velocity prediction and confidence scoring (0.0-1.0)
   - Dynamic bounding box (50-100px) based on propeller positions
   - Handle out-of-frame cases gracefully

4. **Angle Estimation**
   - Analyze propeller layout aspect ratio
   - Face-on (0°) = circular, tilted = elongated
   - Estimate viewing angle 0-90° automatically

**Output:** For each frame (50ms intervals), saves:
- Drone centroid (x, y)
- 4 propeller positions (x, y each)
- Bounding box coordinates
- Estimated angle
- Tracking confidence
- PNG visualization + metadata.json

**No manual annotation required!**

---

### 3. **Multi-Task AI Tracker** 🧠

**The Problem:** Real-time drone tracking requires simultaneous position, propeller, and angle estimation with millisecond latency.

**The Solution:** A multi-task neural network that learns all aspects of drone tracking from automatically labeled datasets.

**Architecture:**
```
Input (7 features: position, spread, events, confidence)
    ↓
Shared Backbone: 256 → 128 → 64 neurons (ReLU + Dropout)
    ↓
├─→ Position Head: x, y coordinates
├─→ Propeller Count Head: 0-4 classification (90.7% accuracy)
├─→ Propeller Positions Head: Up to 4 propellers (x, y each)
├─→ Angle Head: 0-90° estimation (~12.7° MAE)
└─→ Confidence Head: 0-1 tracking quality

Total: 58K parameters (optimized for real-time)
```

**Why Multi-Task Learning?**
- Shared backbone learns general drone features
- Tasks help each other: propeller detection → position accuracy
- Better performance than separate models
- Faster inference (single forward pass)

**Training:**
- Weighted multi-task loss (balances all 5 objectives)
- Adam optimizer + learning rate scheduling
- Early stopping on validation loss
- Trains on all datasets in `drone_dataset/` automatically

**Performance:**
- 🚀 **Real-time capable:** 30 FPS on CPU, 100+ FPS on GPU
- 🎯 **90.7% propeller count accuracy**
- 📐 **~12.7° angle error**
- 💡 **50ms inference time** on modest hardware, and can go much lower.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone repository
cd neutrone

# Install dependencies (uses uv)
uv sync
```

### Complete Workflow

```bash
# 1️⃣ Calculate RPM from fan/propeller recording
uv run python scripts/play_dat_rpm.py YOUR_RECORDING.dat

# 2️⃣ Generate training dataset from drone recording
./scripts/generate_drone_dataset.sh YOUR_RECORDING.dat label_name

# 3️⃣ Train AI model (uses all datasets in drone_dataset/)
uv run python scripts/train_drone_tracker.py

# 4️⃣ Track drone in new recording with visualization
uv run python scripts/visualize_tracking.py YOUR_RECORDING.dat --output result.mp4
```

---

## 📖 Detailed Usage Guide

### 🎵 RPM Detection

**Purpose:** Measure propeller rotation speed without knowing blade count.

**Script:** `scripts/play_dat_rpm.py`

```bash
# Basic usage
uv run python scripts/play_dat_rpm.py RECORDING.dat

# With options
uv run python scripts/play_dat_rpm.py RECORDING.dat \
    --window 10 \      # Event window duration in ms (default: 10)
    --speed 1.0 \      # Playback speed (default: 1.0)
    --history 2.0 \    # History duration for RPM calculation in seconds
    --update-interval 0.5  # RPM update interval in seconds
```

**Controls:**
- `q` - Quit
- `p` - Pause/Resume

**Output:**
- Real-time RPM display with confidence score
- Green text = high confidence (strong periodic signal)
- Red text = low confidence (noisy or varying speed)

**Example:**
```bash
# Detect constant speed fan RPM
uv run python scripts/play_dat_rpm.py src/evio/source/fan_const_rpm.dat

# Result displays: RPM: 1200.5 (confidence: 0.87)
```

---

### 📊 Dataset Generation

**Purpose:** Create labeled training data from raw drone recordings.

**Script:** `scripts/generate_drone_dataset.sh`

```bash
# Generate dataset from any .dat recording
./scripts/generate_drone_dataset.sh RECORDING.dat LABEL [REFERENCE.json]

# Examples:
./scripts/generate_drone_dataset.sh my_flight.dat test
./scripts/generate_drone_dataset.sh src/evio/source/drone_idle.dat idle
./scripts/generate_drone_dataset.sh src/evio/source/drone_moving.dat moving
```

**Arguments:**
1. `RECORDING.dat` - Path to event camera recording
2. `LABEL` - Dataset label (idle/moving/test/etc.)
3. `REFERENCE.json` - Optional: Reference metadata for angle calculation

**Output Structure:**
```
drone_dataset/
├── recording_name/
│   ├── frames/              # PNG visualizations
│   │   ├── frame_000000.png
│   │   ├── frame_000001.png
│   │   └── ...
│   └── metadata.json        # Training labels with all annotations
```

**What Gets Labeled (Automatically):**
- ✅ Drone centroid (x, y)
- ✅ 4 propeller positions (x, y each)
- ✅ Bounding box coordinates
- ✅ Viewing angle (0-90°)
- ✅ Tracking confidence (0-1)
- ✅ Propeller count (0-4)
- ✅ Event statistics

**Direct Script Usage:**
```bash
# For advanced usage, call generate_dataset.py directly
uv run python scripts/generate_dataset.py RECORDING.dat \
    --output drone_dataset/my_dataset \
    --delta 50 \       # Frame interval in ms
    --label custom \
    --reference drone_dataset/idle/metadata.json  # Optional
```

---

### 🧠 AI Training

**Purpose:** Train multi-task neural network on all available datasets.

**Script:** `scripts/train_drone_tracker.py`

```bash
# Train on all datasets in drone_dataset/
uv run python scripts/train_drone_tracker.py
```

**Training Process:**
1. Automatically finds all `metadata.json` files in `drone_dataset/*/`
2. Loads and combines all datasets
3. Splits data: 70% train, 15% validation, 15% test
4. Trains multi-task model for 150 epochs with early stopping
5. Saves best model to `models/best_drone_tracker.pth`
6. Generates training plots: `models/drone_tracker_results.png`

**Output:**
```
Loading datasets...
  Loading idle... Added 52 frames
  Loading moving... Added 355 frames
  Loading challenge... Added 200 frames
Total frames: 607

Training on cuda
Epoch 10/150: Train: 913191.97 | Val: 897244.69
Epoch 20/150: Train: 439794.24 | Val: 532722.36
...
Best model saved at epoch 144!

Test Set Evaluation:
📍 Position Tracking: Mean Error: 141.73 pixels
🔢 Propeller Count: Accuracy: 90.7%
🚁 Propeller Positions: Mean Error: 130.66 pixels
📐 Angle Estimation: MAE: 12.72°
💯 Confidence Prediction: MAE: 0.468

✓ Training complete!
```

**Model Files Created:**
- `models/best_drone_tracker.pth` - Trained weights (PyTorch checkpoint)
- `models/drone_tracker_results.png` - Comprehensive training metrics

---

### 🎥 Visualization & Real-Time Tracking

**Purpose:** Use trained AI to track drones in new recordings.

**Script:** `scripts/visualize_tracking.py`

```bash
# Live visualization (opens window)
uv run python scripts/visualize_tracking.py RECORDING.dat

# Create video without display (faster)
uv run python scripts/visualize_tracking.py RECORDING.dat \
    --output result.mp4 \
    --no-display

# Full options
uv run python scripts/visualize_tracking.py RECORDING.dat \
    --model models/best_drone_tracker.pth \  # Model path (default)
    --output tracking.mp4 \                  # Save video
    --delta 50 \                             # Frame interval in ms
    --device cuda \                          # Use GPU (or cpu)
    --save-frames                            # Export individual frames
```

**Controls (Live Mode):**
- `q` - Quit
- `p` - Pause/Resume
- `s` - Save screenshot

**Visualization Elements:**
- 🟡 **Yellow circle** - Drone centroid (AI predicted)
- 📦 **Bounding box** - Color-coded by tracking confidence:
  - 🟢 Green = High confidence (0.8-1.0) - excellent tracking
  - 🟡 Yellow = Medium (0.5-0.8) - good tracking
  - 🟠 Orange = Low (0.3-0.5) - uncertain
  - 🔴 Red = Very low (0.0-0.3) - poor tracking
- 📊 **Info Panel** - Frame number, timestamp, position, propeller count, confidence
- 📈 **Confidence Bar** - Visual tracking quality indicator

**Example:**
```bash
# Track drone and save video
uv run python scripts/visualize_tracking.py challenge.mp4 --output tracked.mp4

# Result: tracked.mp4 created with AI overlay
```

**Batch Processing:**
```bash
# Process multiple recordings automatically
for file in recordings/*.dat; do
    uv run python scripts/visualize_tracking.py "$file" \
        --output "results/$(basename $file .dat)_tracked.mp4" \
        --no-display
done
```

---

### 🔍 Dataset Inspection Tools

**View Dataset Frames:**
```bash
# Visualize generated dataset
uv run python scripts/view_dataset.py drone_dataset/moving

# Interactive viewer with arrow keys to navigate frames
```

**Quick Start Script:**
```bash
# Automated demo of entire pipeline
./scripts/quickstart_ai.sh

# Runs: dataset generation → training → visualization
```

---

## 📊 Performance Benchmarks

### Detection & Tracking

| Metric                   | Value          | Notes                            |
| ------------------------ | -------------- | -------------------------------- |
| Propeller Count Accuracy | **90.7%**      | 0-4 propeller classification     |
| Angle Estimation MAE     | **12.7°**      | 0-90° viewing angle              |
| Position Tracking MAE    | **142 px**     | Drone centroid in 1280×720 frame |
| Propeller Position MAE   | **131 px**     | Individual propeller locations   |
| Inference Speed (CPU)    | **20 FPS**     | Intel/AMD consumer CPU           |
| Inference Speed (GPU)    | **100+ FPS**   | NVIDIA GPU (CUDA)                |
| Model Size               | **58K params** | 0.23 MB on disk                  |
| Latency                  | **50 ms**      | End-to-end per frame             |

### RPM Detection

| Metric               | Value           | Notes                         |
| -------------------- | --------------- | ----------------------------- |
| RPM Range            | **60-3000 RPM** | Tested on fans and propellers |
| Frequency Resolution | **0.5 Hz**      | FFT bin size                  |
| Update Rate          | **2 Hz**        | Recalculates every 0.5s       |
| Signal Window        | **2 seconds**   | History duration              |
| Blade Count Required | **No**          | Works without calibration     |
| Confidence Scoring   | **Yes**         | Based on peak prominence      |

---

## 🏗️ Project Structure

```
evio/
├── scripts/
│   ├── play_dat_rpm.py              # FFT-based RPM detection
│   ├── generate_dataset.py          # Automated dataset generation
│   ├── generate_drone_dataset.sh    # Dataset generation wrapper
│   ├── train_drone_tracker.py       # Multi-task AI training
│   ├── visualize_tracking.py        # Real-time tracking visualization
│   ├── view_dataset.py              # Dataset inspection tool
│   ├── quickstart_ai.sh             # One-command demo
│   ├── backups/                     # Legacy angle-only models
│   └── debug/                       # Development utilities
├── drone_dataset/                   # Generated training data (gitignored)
│   ├── idle/                        # Idle drone reference
│   ├── moving/                      # Moving drone recordings
│   ├── challenge/                   # Test recordings
│   └── dataset_summary.json         # Combined statistics
├── models/                          # Trained models (gitignored)
│   ├── best_drone_tracker.pth       # Multi-task tracker
│   ├── drone_tracker_results.png    # Training metrics
│   └── best_angle_model.pth         # Legacy angle-only model
├── src/evio/                        # Event camera library (by Ahti Helminen)
│   ├── core/                        # Core processing modules
│   │   ├── mmap.py
│   │   ├── pacer.py
│   │   └── recording.py
│   └── source/                      # Data source handlers
│       └── dat_file.py              # .dat file reader
├── pyproject.toml                   # Python dependencies (uv)
├── README.md                        # This file
└── uv.lock                          # Dependency lock file
```

**Note:** Generated data and models are gitignored. Run scripts to create them locally.

---

## 🎓 Technical Deep Dives

### Why Event Cameras for Military Applications?

**Traditional RGB Cameras Fail When:**
- ❌ Low light / Night operations
- ❌ High-speed motion blur (propellers invisible)
- ❌ Extreme dynamic range (sun glare, shadows)
- ❌ Privacy concerns (always recording video)
- ❌ High bandwidth (1080p @ 60 FPS = 3 Gbps)

**Event Cameras Excel:**
- ✅ **Microsecond precision** - no motion blur, capture 300+ FPS equivalent
- ✅ **120dB dynamic range** - works in direct sunlight AND darkness
- ✅ **Low power** - only transmit changes, not full frames
- ✅ **Privacy-preserving** - only motion events, no faces/details
- ✅ **High temporal resolution** - perfect for fast propellers (3000+ RPM)

### Multi-Task Learning Benefits

Training a single model for all tasks provides:
- **Better generalization** - Shared features prevent overfitting
- **Task synergy** - Propeller detection helps position tracking
- **Faster inference** - One forward pass instead of five models
- **Smaller memory footprint** - 58K params vs. 5× separate models
- **Easier deployment** - Single model file to manage

### Propeller Detection Innovation

Our detection algorithm is **propeller-first**, not drone-first:

**Traditional Approach (Fails):**
1. Detect drone body → Find propellers
2. Problem: Drone body often invisible in events
3. Problem: Background noise confuses detectors

**Our Approach (Works):**
1. Find all ultra-dense event clusters (propeller signatures)
2. Score and group clusters into drone configurations
3. Heavily favor 3-4 separated propellers (20x bonus)
4. Reject merged blobs and distant objects

**Key Insight:** Spinning propellers create the most distinctive signature in event data—far more recognizable than the drone body itself.

---

## 🛠️ Troubleshooting

### RPM Detection Issues

**Problem:** RPM shows 0 or very low confidence
- ✅ Check propeller is spinning (needs motion)
- ✅ Increase history window: `--history 3.0`
- ✅ Ensure event camera is recording (.dat file has data)
- ✅ Try slower playback: `--speed 0.5`

**Problem:** RPM value fluctuates wildly
- ✅ Normal for varying speed (real-time tracking)
- ✅ For constant speed, low confidence indicates noisy signal
- ✅ Check if propeller fully visible in frame

### Dataset Generation Issues

**Problem:** Only 1 propeller detected consistently
- ✅ Normal at recording start/end (drone entering/leaving)
- ✅ Check detection quality: `jq '.frames[].event_stats.num_propellers' drone_dataset/*/metadata.json | sort | uniq -c`
- ✅ Should see 3-4 propellers in majority of frames

**Problem:** Tracking confidence always low
- ✅ Drone may be far from camera
- ✅ Check propeller signatures are visible
- ✅ Inspect frames visually: `uv run python scripts/view_dataset.py drone_dataset/your_dataset`

**Problem:** "Out of frame" for all frames
- ✅ Recording may be empty or corrupted
- ✅ Check .dat file size (should be >10 MB)
- ✅ Verify event camera was recording

### Training Issues

**Problem:** "No training data found"
- ✅ Run dataset generation first: `./scripts/generate_drone_dataset.sh`
- ✅ Check `drone_dataset/` has subdirectories with `metadata.json`
- ✅ Need minimum 200+ frames for training

**Problem:** Training loss not decreasing
- ✅ Check dataset diversity (multiple recordings recommended)
- ✅ Verify propeller detection quality in dataset
- ✅ Try more epochs (edit script to increase from 150)

**Problem:** GPU out of memory
- ✅ Training uses ~2 GB VRAM (should work on most GPUs)
- ✅ Reduce batch size if needed (edit script)
- ✅ Fall back to CPU (slower but works)

### Visualization Issues

**Problem:** Model file not found
- ✅ Train model first: `uv run python scripts/train_drone_tracker.py`
- ✅ Check `models/best_drone_tracker.pth` exists
- ✅ Ensure training completed successfully

**Problem:** Bounding box color doesn't change
- ✅ Should vary based on tracking confidence
- ✅ Green = good, Yellow = medium, Red = poor
- ✅ If always one color, check detection confidence in metadata

**Problem:** Video creation is slow
- ✅ Use `--no-display` flag (2-3x faster)
- ✅ Increase delta: `--delta 100` for lower FPS output
- ✅ Use GPU: `--device cuda` for inference speedup

---

## 🔮 Future Enhancements

### Immediate Next Steps
- [ ] **Multi-drone tracking** - Simultaneous tracking of multiple drones
- [ ] **Trajectory prediction** - Predict flight path for intercept
- [ ] **Threat classification** - Friend/foe identification

### Military Deployment Features
- [ ] **Edge deployment** - Jetson Nano, Raspberry Pi optimization
- [ ] **Sensor fusion** - Combine event camera + radar + acoustic
- [ ] **Counter-drone actions** - Integration with jamming/capture systems
- [ ] **Swarm detection** - Handle coordinated multi-drone attacks
- [ ] **C2 integration** - Command and control system connectivity

### Technical Improvements
- [ ] **TensorRT optimization** - 5-10x GPU inference speedup
- [ ] **ONNX export** - Cross-platform deployment
- [ ] **Kalman filtering** - Smoother trajectory tracking
- [ ] **3D pose estimation** - Full 6-DOF orientation
- [ ] **Online learning** - Adapt to new drone types in field

---

## 📜 License & Credits

### evio Library
This project builds on the **evio** event camera library by [Ahti Helminen](https://github.com/helminen).  
See `src/evio/` for library code and original license.

### NEUTRONE Components
Detection algorithms, AI models, and military applications developed by:
- **Team Blacklist**
- Junction Hackathon 2025, Espoo, Finland
- SensoFusion Challenge

---

## 🙏 Acknowledgments

- **Ahti Helminen** - evio slibrary and event camera expertise
- **Junction Organizers** - Hosting and supporting the hackathon
- **SensoFusion** - Event camera hardware and technical challenge

---

## 📞 Contact

**Team Blacklist**  
Junction Hackathon 2025  
SensoFusion Challenge - Lights, Camera, Reaction! 

For technical questions or collaboration:
- Repository: [github.com/YassWrld/Neutrone](https://github.com/YassWrld/Neutrone)

---

**NEUTRONE** - *Because traditional cameras can't keep up with modern drone threats.* 🚁⚡

*Built for Junction Hackathon 2025 | SensoFusion Challenge | Lights, Camera, Reaction!*
      │   ├── index_scheduler.py
      │   ├── mmap.py
      │   ├── pacer.py
      │   └── recording.py
      └── source/
          ├── dat_file.py
          ├── drone_moving.dat   # Sample event recordings
          └── drone_idle.dat
```

---

## Quick Start

### Installation
```bash
# Install UV package manager (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository-url>
cd propeless
uv sync
```

### Complete AI Pipeline (One Command)
```bash
# Generate dataset, train model, and run inference
./scripts/quickstart_ai.sh
```

This will:
1. Generate labeled training data from event recordings (407 frames, 89.2% smooth tracking)
2. Train PyTorch angle estimation model (~5-10 minutes)
3. Run demo prediction on test recording

### Individual Commands

#### Generate Dataset
```bash
uv run python scripts/generate_dataset.py src/evio/source/drone_moving.dat \
    --output drone_dataset/moving \
    --delta 50 \
    --label moving
```

#### Train Model
```bash
uv run python scripts/train_angle_model.py
# Outputs: models/best_angle_model.pth
#          models/training_results.png
```

#### Run Inference
```bash
./scripts/predict_angles.py src/evio/source/drone_moving.dat
# Outputs: drone_moving_predictions.json
```

#### Validate Tracking Quality
```bash
uv run python scripts/check_tracking.py drone_dataset/moving/metadata.json
```

### Visualize Dataset
```bash
uv run python scripts/view_dataset.py drone_dataset/moving
```

---

## Technical Details

### Propeller Detection Algorithm
- **Grid-based scanning**: 15px cells for dense event cluster detection
- **Density criteria**: within_8px≥8, within_15px≥15, tightness≥0.35
- **Dual-path detection**: Handles separated propellers (3-4 clusters) and merged propellers (super-dense blob)
- **Temporal tracking**: Velocity prediction for smooth tracking across frames
- **Out-of-frame handling**: Automatic detection when drone leaves camera view

### AI Model Architecture
```
Input: 15 features
  ├─ 7 core: x_mean, y_mean, x_std, y_std, num_events, confidence, num_propellers
  └─ 8 propeller coordinates: (x1,y1), (x2,y2), (x3,y3), (x4,y4)

Network: 
  Dense(128) → ReLU → Dropout(0.2) →
  Dense(64)  → ReLU → Dropout(0.2) →
  Dense(32)  → ReLU →
  Dense(1)   [angle output]

Training: MSE loss, Adam optimizer, ReduceLROnPlateau
Expected Performance: 3-5° mean absolute error
```

### Dataset Statistics
- **407 frames** total (50ms intervals, 20.35s recording)
- **353 tracked frames** (86.7%) with valid drone detection
- **54 out-of-frame** (13.3%) correctly identified
- **89.2% smooth tracking** (38 jumps over 353 frames)
- **50.4%** with 3+ propellers detected

---

## `.dat` File Encoding (evio format)

`evio` reads Prophesee Metavision-style DAT files, which store events as fixed-width binary records following a short ASCII header.

### Header
The file starts with text lines beginning with `%`, for example:

```
% Width 1280
% Height 720
% Format EVT3
```

After the header, two bytes appear:

- **event_type** — currently only stored in metadata (not interpreted by `evio`)
- **event_size** — must be `8`, meaning each event occupies 8 bytes

### Event Record Format (8 bytes)
The binary payload is interpreted as an array of structured records with dtype:

```python
_DTYPE_CD8 = np.dtype([("t32", "<u4"), ("w32", "<u4")])
```

Each event record is 8 bytes (64 bits):

- `t32` (upper 32 bits) is a little-endian `uint32` timestamp in microseconds.
- `w32` (lower 32 bits) packs polarity and coordinates as:

| Bits  | Meaning                              |
| ----- | ------------------------------------ |
| 31–28 | polarity (4 bits; > 0 → ON, 0 → OFF) |
| 27–14 | y coordinate (14 bits)               |
| 13–0  | x coordinate (14 bits)               |

This matches the decoder:

```python
packed_w32 = raw_events["w32"].astype(np.uint32, copy=False)

decoded_x = (packed_w32 & 0x3FFF).astype(np.uint16, copy=False)
decoded_y = ((packed_w32 >> 14) & 0x3FFF).astype(np.uint16, copy=False)
raw_polarity = ((packed_w32 >> 28) & 0xF).astype(np.uint8, copy=False)
decoded_polarity = (raw_polarity > 0).astype(np.int8, copy=False)
```

### Decoded Arrays in `evio`
`evio` exposes the following decoded NumPy arrays:

- `x_coords` — uint16 (from bits 0–13)
- `y_coords` — uint16 (from bits 14–27)
- `timestamps` — int64 (from `t32` promoted from uint32)
- `polarities` — int8 (0 for OFF, 1 for ON)

### Memory-Mapped Reading
`evio` uses a `numpy.memmap` view of the event region with `_DTYPE_CD8` and performs zero-copy decoding of the packed fields. This allows:

- fast slicing of large recordings
- stable real-time playback
- minimal memory use even with millions of events




## Credits

### Built with evio
This project is built on top of [evio](https://github.com/ahtihelminen/evio) by **Ahti Helminen** - a minimal Python library for standardized handling of event camera data. evio provides the foundation for reading `.dat` recordings and processing event streams.

### Junction Hackathon 2025
Created for Junction Hackathon 2025 - demonstrating the power of event cameras for drone tracking without traditional RGB imaging.

---

## License
MIT

## References
- [evio GitHub Repository](https://github.com/ahtihelminen/evio)
- [Prophesee Metavision SDK](https://docs.prophesee.ai/)
- Event Camera Technology: Neuromorphic vision sensors that detect pixel-level changes asynchronously

