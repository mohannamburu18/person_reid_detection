# Setup Guide for Person Re-Identification System

## Step-by-Step Installation

### 1. System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- CPU: Intel i5 or equivalent

**Recommended:**
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- CUDA 11.7+ and cuDNN

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv reid_env
source reid_env/bin/activate  # Linux/Mac
# OR
reid_env\Scripts\activate  # Windows

# Using conda
conda create -n reid_env python=3.9
conda activate reid_env
```

### 3. Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# For GPU support (CUDA 11.7)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu117

# For GPU support (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 4. Download Market-1501 Dataset

**Option 1: Direct Download**
- Link: https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view
- Extract to: `data/Market-1501/`

**Option 2: Using Command Line (Linux)**
```bash
# Install gdown
pip install gdown

# Download and extract
mkdir -p data
cd data
gdown --id 0B8-rUzbwVRk0c054eEozWG9COHM
unzip Market-1501-v15.09.15.zip
mv Market-1501-v15.09.15 Market-1501
cd ..
```

**Expected Structure:**
```
data/Market-1501/
├── bounding_box_test/  (19,732 images)
├── bounding_box_train/ (12,936 images)
├── query/              (3,368 images)
└── gt_bbox/            (25,259 images)
```

### 5. Download YOLO Model

The YOLOv8 model will download automatically on first run. To pre-download:

```bash
# Download YOLOv8m (medium, recommended)
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Or download YOLOv8s (small, faster)
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

### 6. Prepare Dataset

```bash
# Convert images to videos
python data_prep/images_to_video.py

# Verify videos were created
ls -lh data/fake_cctv_videos/
```

Expected output: `c1.mp4`, `c2.mp4`, `c3.mp4`, `c4.mp4`, `c5.mp4`, `c6.mp4`

### 7. Test Installation

```bash
# Quick test
python -c "
from reid.global_id_manager import GlobalIDManager
import torch
print('✓ ReID module loaded')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"

# Test YOLO
python -c "
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
print('✓ YOLO model loaded')
"
```

### 8. Run the System

```bash
# Run complete pipeline
python main.py

# Run with data preparation
python main.py --prepare-data
```

## Common Installation Issues

### Issue 1: PyTorch CUDA version mismatch

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.7
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu117

# For CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: OpenCV import error

```bash
# Uninstall all OpenCV versions
pip uninstall opencv-python opencv-python-headless opencv-contrib-python

# Reinstall clean version
pip install opencv-python==4.8.1.78
```

### Issue 3: Ultralytics installation fails

```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Install ultralytics
pip install ultralytics==8.0.221
```

### Issue 4: Out of memory error

**Solutions:**
1. Use smaller YOLO model: Change to `yolov8n.pt` in `run_pipeline.py`
2. Increase frame skip: Set `FRAME_SKIP = 3` or higher
3. Reduce gallery size: Set `MAX_GALLERY_SIZE = 5`
4. Disable deep features: Set `USE_DEEP_FEATURES = False`

## Performance Optimization

### For CPU-only systems

Edit `reid/global_id_manager.py`:
```python
# Force CPU
self.device = torch.device('cpu')
```

Edit `inference/run_pipeline.py`:
```python
# Optimize settings
FRAME_SKIP = 3
USE_DEEP_FEATURES = False
MIN_FRAMES_TO_CONFIRM = 3
```

### For GPU systems

```python
# Optimize settings
FRAME_SKIP = 1  # Process all frames
USE_DEEP_FEATURES = True
model = YOLO("yolov8l.pt")  # Use larger model
```

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list`)
- [ ] PyTorch with CUDA support (if GPU available)
- [ ] Market-1501 dataset downloaded and extracted
- [ ] Dataset structure verified
- [ ] YOLO model downloaded
- [ ] Fake CCTV videos generated
- [ ] Test imports successful
- [ ] System runs without errors

## Next Steps

1. Read `README.md` for usage guide
2. Review configuration options in `inference/run_pipeline.py`
3. Adjust thresholds based on your requirements
4. Run the pipeline: `python main.py`
5. Check output and logs

## Getting Help

If you encounter issues:
1. Check error messages carefully
2. Review this guide and README.md
3. Verify all dependencies are installed
4. Check CUDA/GPU availability
5. Try CPU-only mode first
6. Consult troubleshooting section in README.md

Happy Re-Identifying! 🎯
