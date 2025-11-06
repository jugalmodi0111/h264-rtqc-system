# H.264 Real-Time Quality Control System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based system for intelligent H.264 video encoding quality control. Given a target PSNR (Peak Signal-to-Noise Ratio), the system automatically predicts the optimal QP (Quantization Parameter) to achieve that quality level efficiently.

## 🎯 Overview

Traditional video encoding requires manual tuning or trial-and-error to achieve desired quality levels. This project uses a **3D Convolutional Neural Network** with **Conditional Group Normalization** to learn the complex relationship between video content, target quality (PSNR), and encoding parameters (QP).

### Key Features

- ✅ **Intelligent QP Prediction**: Automatically predicts optimal QP values for target PSNR levels
- ✅ **Content-Aware**: Uses 3D CNN (X3D architecture) to analyze spatial and temporal video features
- ✅ **PSNR-Conditioned Processing**: Conditional Group Normalization layers adapt network behavior based on target quality
- ✅ **Real H.264 Encoding**: Supports FFmpeg with OpenH264/libx264 codecs (with JPEG fallback)
- ✅ **Production Ready**: Includes training pipeline, evaluation metrics, and model persistence
- ✅ **Interactive Notebook**: Complete Jupyter notebook with explanations, visualizations, and examples

## 📊 Results

Training on lightweight sample videos demonstrates clear improvement:

| Metric | Before Training | After Training | Improvement |
|--------|----------------|----------------|-------------|
| **Mean QP Error** | 21.75 | 0.75 | **96.6%** reduction |
| **Mean PSNR Gap** | 10.73 dB | 3.64 dB | **66.1%** reduction |

The model learns to predict QP values much closer to optimal, achieving target PSNR levels more accurately.

## 🏗️ Architecture

### Model Components

1. **X3DBackbone** (3D CNN Feature Extractor)
   - 4 convolutional layers with 3D kernels (3×3×3)
   - Progressively increases channels: 3 → 16 → 32 → 64 → 128
   - Extracts spatiotemporal features from video clips
   - Output: 128-dimensional feature vector per frame

2. **ConditionalGroupNorm** (PSNR-Conditioned Normalization)
   - Adapts normalization parameters based on target PSNR
   - Learns to modulate features differently for quality levels
   - Improves model's ability to condition on quality targets

3. **H264QualityController** (Full Model)
   - Combines X3D backbone with PSNR conditioning
   - FC layers with dropout for classification
   - Outputs: 52 classes (QP values 0-51)
   - Total parameters: ~5.8 million

### Pipeline Flow

```
Input Video [B, 3, T, H, W] + Target PSNR
            ↓
    X3D Backbone (3D CNN)
            ↓
    Feature Vector [B, 128]
            ↓
Conditional Group Norm (PSNR-conditioned)
            ↓
    FC Layers + Dropout
            ↓
    Softmax(52 classes)
            ↓
    Predicted QP (0-51)
            ↓
H.264 Encode → Measure Actual PSNR
```

## 📁 Project Structure

```
openH264/
├── Untitled-1.ipynb          # Main interactive notebook with complete pipeline
├── assets/                    # Generated visualizations
│   ├── qp_psnr_curve.png     # QP vs PSNR relationship curve
│   ├── rtqc_pipeline.png     # System architecture diagram
│   └── psnr_vs_qp_*.png      # Content-specific PSNR measurements
├── data/
│   └── videos/                # Sample training videos
│       ├── sample-5s.mp4     # 5-second sample clip
│       └── sample-10s.mp4    # 10-second sample clip
├── models/                    # Saved model checkpoints
│   └── h264_quality_controller.pth
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- FFmpeg with OpenH264 or libx264 codec (optional, JPEG fallback available)
- CUDA-capable GPU (optional, CPU/MPS supported)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/h264-quality-control.git
   cd h264-quality-control
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg (optional but recommended)**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

### Quick Start

1. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook Untitled-1.ipynb
   ```

2. **Run the setup cells** to install packages and import libraries

3. **Explore the examples**:
   - Simple QP prediction tests
   - PSNR vs QP measurements for different content types
   - Training on sample videos
   - Before/after evaluation

4. **Train your own model**:
   - Add your videos to `data/videos/`
   - Run the training cells
   - Monitor loss and accuracy improvements

## 📖 Understanding the Notebook

The notebook is organized into educational modules:

### 1. Setup & Installation
- Package installation with Jupyter magic commands
- Library imports with detailed explanations

### 2. Model Architecture
- **X3DBackbone**: 3D CNN for video feature extraction
- **ConditionalGroupNorm**: PSNR-conditioned normalization
- **H264QualityController**: Complete end-to-end model

### 3. Training Pipeline
- **H264TrainingPipeline**: Handles encoding, PSNR measurement, training loops
- FFmpeg integration with automatic fallback to JPEG
- Binary search for optimal QP finding

### 4. Visualization & Analysis
- QP vs PSNR curves for understanding encoder behavior
- Pipeline flow diagrams
- PSNR measurement plots for different content types

### 5. Training & Evaluation
- Sample video dataset download
- Baseline evaluation (before training)
- Training loop with loss monitoring
- Post-training evaluation with improvement metrics

### 6. Model Persistence
- Save trained models
- Load and resume training

## 🔬 How It Works

### The H.264 Quality Problem

H.264 video encoding uses a **Quantization Parameter (QP)** to control quality:
- **QP = 0**: Best quality, largest file size
- **QP = 51**: Worst quality, smallest file size
- **Challenge**: Same QP produces different PSNR for different content

### Our Solution

Instead of manually tuning QP for each video, our model:

1. **Analyzes video content** using 3D convolutions (spatial + temporal)
2. **Considers target PSNR** through conditional normalization
3. **Predicts optimal QP** that achieves the target quality
4. **Learns from examples** via supervised training with optimal QP labels

### Training Process

1. **Input**: Video clip + target PSNR (e.g., 35 dB)
2. **Label Generation**: Binary search to find QP that achieves target PSNR
3. **Training**: Minimize cross-entropy loss between predicted and optimal QP
4. **Validation**: Measure QP error and PSNR gap on held-out videos

## 📈 Performance Metrics

### QP Error
Mean absolute difference between predicted QP and optimal QP:
```
|predicted_QP - optimal_QP|
```
Lower is better. After training: **0.75** (vs 21.75 baseline)

### PSNR Gap
Mean absolute difference between achieved PSNR and target PSNR:
```
|actual_PSNR - target_PSNR| (in dB)
```
Lower is better. After training: **3.64 dB** (vs 10.73 dB baseline)

## 🎨 Visualizations

The notebook generates several visualizations saved in `assets/`:

- **QP-PSNR Curve**: Shows typical relationship between QP and PSNR
- **Pipeline Diagram**: Visual representation of the system architecture
- **Content Analysis**: PSNR vs QP plots for gradient, checkerboard, noise, and solid patterns

## �️ Advanced Usage

### Custom Training

```python
# Create your dataset
dataset = SimpleVideoDataset(
    video_dir='path/to/videos',
    clip_length=16,
    num_clips_per_video=50
)

# Initialize model
model = H264QualityController(in_channels=3, num_frames=16)

# Train
pipeline = H264TrainingPipeline(model)
pipeline.train_model(train_loader, val_loader, epochs=10)
```

### Model Inference

```python
# Load trained model
model = H264QualityController(in_channels=3, num_frames=16)
model.load_state_dict(torch.load('models/h264_quality_controller.pth'))
model.eval()

# Predict QP for target PSNR
video_clip = torch.randn(1, 3, 16, 224, 224)  # [B, C, T, H, W]
target_psnr = torch.tensor([35.0])  # Target 35 dB

predicted_qp = model(video_clip, target_psnr)
qp_value = predicted_qp.argmax(dim=1).item()
print(f"Predicted QP: {qp_value}")
```

## 🧪 Testing Different Content Types

The system handles various content types differently:

- **Gradient patterns**: Smooth transitions, compress well
- **Checkerboard**: High frequency, harder to compress
- **Noise**: Random patterns, very difficult to compress
- **Solid colors**: Trivial compression

Run the measurement cells to see how QP affects PSNR for each type.

## 📦 Dependencies

Core libraries:
- **PyTorch 2.8.0**: Deep learning framework
- **OpenCV (cv2)**: Video/image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **scikit-image**: PSNR calculations
- **av (PyAV)**: Optional video I/O

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Multi-GPU training support
- [ ] Additional video codec support (VP9, AV1)
- [ ] Real-time inference optimization
- [ ] Larger benchmark datasets
- [ ] Rate-distortion optimization
- [ ] Temporal consistency constraints

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **X3D Architecture**: Inspired by Facebook AI Research's X3D networks
- **H.264 Standard**: ITU-T Recommendation H.264 / ISO/IEC 14496-10
- **Sample Videos**: Courtesy of samplelib.com
- **FFmpeg**: Open-source multimedia framework

## � References

1. Feichtenhofer, C. (2020). "X3D: Expanding Architectures for Efficient Video Recognition"
2. ITU-T H.264: "Advanced video coding for generic audiovisual services"
3. Wang, Z., et al. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity"

---

**Note**: This project is for educational and research purposes. For production video encoding systems, consider additional factors like rate control, scene detection, and bitrate constraints.
