# Person Re-Identification System for Railway Surveillance

Advanced multi-camera person re-identification system using deep learning features and YOLOv8 detection, optimized for Market-1501 dataset.

## 🎯 Features

- **Deep Learning ReID**: ResNet50-based feature extraction for accurate person matching
- **Multi-Camera Tracking**: Track persons across multiple camera views
- **Temporal & Spatial Constraints**: Smart filtering based on camera topology and timing
- **Gallery-based Matching**: Maintains multiple feature representations per person
- **Real-time Visualization**: Live bounding boxes and person IDs
- **Comprehensive Logging**: Movement paths and re-detection events

## 📁 Project Structure

```
person_reid_project/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/                        # Data directory
│   ├── Market-1501/            # Place Market-1501 dataset here
│   │   ├── bounding_box_test/  # Test images
│   │   └── query/              # Query images
│   └── fake_cctv_videos/       # Generated CCTV videos (auto-created)
│
├── reid/                        # ReID core module
│   ├── __init__.py
│   └── global_id_manager.py    # Global ID assignment & feature extraction
│
├── inference/                   # Pipeline execution
│   ├── __init__.py
│   └── run_pipeline.py         # Main detection & tracking pipeline
│
└── data_prep/                   # Data preparation
    └── images_to_video.py      # Convert images to videos
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For GPU support (recommended), install CUDA-enabled PyTorch
# Visit: https://pytorch.org/get-started/locally/
```

### 2. Download Market-1501 Dataset

Download from: [Market-1501 Dataset](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)

Extract to:
```
data/Market-1501/
├── bounding_box_test/
└── query/
```

### 3. Prepare Dataset

Convert images to fake CCTV videos:

```bash
python data_prep/images_to_video.py
```

Or use the integrated command:

```bash
python main.py --prepare-data
```

### 4. Run ReID Pipeline

```bash
python main.py
```

## ⚙️ Configuration

Edit configuration in `inference/run_pipeline.py`:

```python
# Performance settings
FRAME_SKIP = 2                    # Process every Nth frame
MIN_FRAMES_TO_CONFIRM = 5         # Frames for stable detection
LOST_THRESHOLD = 45               # Frames before marking as LOST

# Detection quality
MIN_DETECTION_CONF = 0.45         # Minimum confidence
MIN_PERSON_HEIGHT = 100           # Minimum bbox height
MIN_PERSON_WIDTH = 40             # Minimum bbox width

# ReID settings
USE_DEEP_FEATURES = True          # Use ResNet50 vs color histogram
REID_THRESHOLD = 0.65             # Similarity threshold (0-1)
MAX_GALLERY_SIZE = 10             # Features stored per person
```

### Threshold Tuning Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.5-0.6   | Very strict | Few false matches, may miss some |
| 0.65-0.7  | **Balanced** (recommended) | Good accuracy/recall trade-off |
| 0.75-0.85 | Lenient | More matches, some false positives |

### Feature Extraction Options

1. **Deep Features (ResNet50)** - Default, recommended
   - Higher accuracy
   - Requires GPU for real-time performance
   - Set `USE_DEEP_FEATURES = True`

2. **Color Histogram** - Fallback
   - Faster, CPU-friendly
   - Lower accuracy
   - Set `USE_DEEP_FEATURES = False`

## 📊 Output Examples

### Console Output
```
[CAMERA 1/3] Processing c1...
✓ [RE-DETECTED] Person 5 in c2 (visited 2 cameras)
✓ [RE-DETECTED] Person 12 in c3 (visited 3 cameras)
✗ [LOST] Person 8 (last seen in c1)

===== PERSON MOVEMENT SUMMARY =====
Total Unique Persons: 45
Cross-Camera Tracks: 23

--- Movement Paths ---
Person   5: c1 → c2
Person  12: c1 → c2 → c3
Person  23: c2 → c4 → c6
```

### Visual Output
- Green bounding boxes with person IDs
- Real-time camera feed display
- Active tracking count per frame

## 🔧 Advanced Usage

### Using Custom Videos

Replace fake CCTV videos with your own:

1. Place videos in `data/fake_cctv_videos/`
2. Name them: `c1.mp4`, `c2.mp4`, etc.
3. Run: `python main.py`

### Evaluation on Market-1501

For standard evaluation metrics (Rank-1, mAP):

```python
# See evaluation/eval_market1501.py (to be implemented)
# Requires ground truth from Market-1501 query/gallery split
```

### Camera Topology Configuration

Edit camera connections in `reid/global_id_manager.py`:

```python
self.camera_connections = {
    'c1': ['c2', 'c3'],        # c1 connects to c2 and c3
    'c2': ['c1', 'c4', 'c5'],  # c2 connects to c1, c4, c5
    # ... add your camera layout
}
```

## 🐛 Troubleshooting

### Issue: Low accuracy

**Solutions:**
1. Increase `REID_THRESHOLD` (0.7-0.75)
2. Use `USE_DEEP_FEATURES = True`
3. Adjust `MIN_FRAMES_TO_CONFIRM` (5-10)
4. Improve detection quality: increase `MIN_DETECTION_CONF`

### Issue: Too many false positives

**Solutions:**
1. Decrease `REID_THRESHOLD` (0.55-0.6)
2. Add more spatial constraints
3. Increase `MIN_FRAMES_TO_CONFIRM`

### Issue: Slow performance

**Solutions:**
1. Increase `FRAME_SKIP` (3-5)
2. Use smaller YOLO model: `yolov8n.pt`
3. Set `USE_DEEP_FEATURES = False`
4. Process fewer cameras at once

### Issue: CUDA out of memory

**Solutions:**
1. Reduce batch size in feature extraction
2. Use CPU: edit `global_id_manager.py`, force `self.device = 'cpu'`
3. Process one video at a time

## 📚 Key Improvements Over Basic Implementation

1. ✅ **Deep Learning Features**: ResNet50 vs simple color histograms
2. ✅ **Gallery-based Matching**: Multiple features per person
3. ✅ **Quality Filtering**: Size, aspect ratio, confidence checks
4. ✅ **Temporal Constraints**: Realistic travel time between cameras
5. ✅ **Spatial Constraints**: Camera topology awareness
6. ✅ **Smooth Tracking**: Bounding box smoothing reduces jitter
7. ✅ **Better Detection**: YOLOv8m with BoT-SORT tracker

## 🎓 Further Improvements

For production deployment, consider:

1. **Use Specialized ReID Models**: 
   - [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) OSNet
   - FastReID models
   
2. **Add Pose Information**: 
   - Body keypoints for better matching
   - Gait analysis

3. **Implement Re-ranking**: 
   - Query expansion
   - k-reciprocal encoding

4. **Add Database Persistence**:
   - Save features to database
   - Historical analysis

## 📄 License

MIT License - Feel free to use for academic and commercial purposes.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review configuration options
3. Open a GitHub issue with logs

---

**Developed for Railway Surveillance Project - Final Year B.Tech CSE**
