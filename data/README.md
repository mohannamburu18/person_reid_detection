# Data Directory

## Directory Structure

```
data/
├── Market-1501/              # Place Market-1501 dataset here
│   ├── bounding_box_test/   # 19,732 test images
│   ├── bounding_box_train/  # 12,936 train images (optional)
│   ├── query/               # 3,368 query images (for evaluation)
│   └── gt_bbox/             # 25,259 ground truth (optional)
│
└── fake_cctv_videos/        # Auto-generated videos
    ├── c1.mp4
    ├── c2.mp4
    ├── c3.mp4
    ├── c4.mp4
    ├── c5.mp4
    └── c6.mp4
```

## Market-1501 Dataset

### Download

**Direct Link:** https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view

**Using Command Line:**
```bash
# Install gdown
pip install gdown

# Download dataset
cd data
gdown --id 0B8-rUzbwVRk0c054eEozWG9COHM

# Extract
unzip Market-1501-v15.09.15.zip
mv Market-1501-v15.09.15 Market-1501

# Clean up
rm Market-1501-v15.09.15.zip
```

### Dataset Information

**Market-1501** is a large-scale person re-identification dataset:
- 32,668 images of 1,501 identities
- 6 cameras (5 high-resolution, 1 low-resolution)
- Captured in front of a supermarket
- Each person captured by at least 2 cameras

**Statistics:**
- Training set: 12,936 images of 751 identities
- Test set: 19,732 images of 750 identities
- Query set: 3,368 images
- Average of 17.2 images per identity

### Image Naming Convention

Images follow the format: `XXXX_cY_ZZZZ_WW.jpg`

Where:
- `XXXX`: Person ID (0001-1501)
  - 0000: Junk images (background/distractors)
  - -1: False detections
- `cY`: Camera ID (c1, c2, c3, c4, c5, c6)
- `ZZZZ`: Sequence/track number
- `WW`: Frame number within sequence

**Examples:**
- `0001_c1s1_0010_01.jpg`: Person 1, Camera 1, Sequence 1, Frame 10
- `0342_c5s2_0156_03.jpg`: Person 342, Camera 5, Sequence 2, Frame 156

### Folders Required

**Minimum (for this project):**
- ✅ `bounding_box_test/` - Required for generating videos

**Optional:**
- `bounding_box_train/` - For training custom models
- `query/` - For standard evaluation
- `gt_bbox/` - Ground truth for mAP calculation

## Generating Videos

Once you've placed Market-1501 in the data directory, generate videos:

```bash
# Option 1: Run separately
python data_prep/images_to_video.py

# Option 2: Integrated with main pipeline
python main.py --prepare-data
```

This will:
1. Scan `bounding_box_test/` folder
2. Group images by camera ID
3. Create one video per camera
4. Save to `fake_cctv_videos/`

**Expected output:**
- 6 videos (c1.mp4 through c6.mp4)
- ~5 FPS playback speed
- Total size: ~200-300 MB

## Using Your Own Videos

To use real CCTV footage instead:

1. Place your videos in `fake_cctv_videos/`
2. Name them: `c1.mp4`, `c2.mp4`, etc.
3. Run: `python main.py`

**Requirements:**
- Common formats: .mp4, .avi, .mov
- At least 2 cameras for cross-camera tracking
- Preferably with some person overlap between cameras

## Troubleshooting

### "No such file or directory: data/Market-1501"
- Download and extract Market-1501 dataset
- Ensure correct folder name (Market-1501, not Market-1501-v15.09.15)

### "No videos found"
- Run `python data_prep/images_to_video.py` first
- Check that `bounding_box_test/` contains .jpg images

### "No valid images found"
- Verify dataset structure matches expected format
- Check that images follow naming convention

### Videos are empty or corrupt
- Check available disk space
- Verify OpenCV is installed: `pip install opencv-python`
- Try deleting and regenerating videos

## Additional Resources

**Market-1501 Paper:**
"Scalable Person Re-identification: A Benchmark" (ICCV 2015)

**Dataset Homepage:**
https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html

**Citation:**
```bibtex
@inproceedings{zheng2015scalable,
  title={Scalable person re-identification: A benchmark},
  author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
  booktitle={ICCV},
  year={2015}
}
```
