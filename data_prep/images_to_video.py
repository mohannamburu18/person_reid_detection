import os
import cv2
from collections import defaultdict

"""
Market-1501 to Fake CCTV Video Converter

This script converts Market-1501 dataset images into simulated CCTV videos,
one video per camera. This allows testing of cross-camera person re-identification.

Dataset structure expected:
data/Market-1501/bounding_box_test/XXXX_cY_ZZZZ_WW.jpg
Where:
- XXXX: Person ID
- cY: Camera ID (c1, c2, c3, c4, c5, c6)
- ZZZZ: Sequence number
- WW: Frame number
"""

# Configuration
DATASET_DIR = "data/Market-1501/bounding_box_test"
OUTPUT_DIR = "data/fake_cctv_videos"
VIDEO_FPS = 5  # Frames per second for output videos
CODEC = "mp4v"  # Video codec

def create_fake_videos():
    """Convert Market-1501 images to camera-based videos"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("[INFO] Please download Market-1501 dataset and extract to data/Market-1501/")
        return

    # Group images by camera ID
    camera_images = defaultdict(list)
    
    print("[INFO] Scanning Market-1501 dataset...")
    
    for img_name in os.listdir(DATASET_DIR):
        if not img_name.endswith(".jpg"):
            continue
        
        # Parse filename: XXXX_cY_ZZZZ_WW.jpg
        try:
            parts = img_name.split("_")
            if len(parts) < 4:
                continue
                
            cam_id = parts[1]  # e.g., 'c1', 'c2', etc.
            
            # Filter out junk images (-1 person ID)
            person_id = parts[0]
            if person_id == "-1" or person_id == "0000":
                continue
                
            camera_images[cam_id].append(img_name)
        except Exception as e:
            print(f"[WARNING] Could not parse filename: {img_name}")
            continue

    if not camera_images:
        print("[ERROR] No valid images found in dataset")
        return

    print(f"[INFO] Found {len(camera_images)} cameras")
    print(f"[INFO] Total images: {sum(len(imgs) for imgs in camera_images.values())}")

    # Create video for each camera
    for cam_id, images in sorted(camera_images.items()):
        print(f"\n[INFO] Processing camera {cam_id} ({len(images)} images)...")
        
        # Sort images by sequence and frame number
        images.sort()

        # Read first image to get dimensions
        first_img_path = os.path.join(DATASET_DIR, images[0])
        first_img = cv2.imread(first_img_path)
        
        if first_img is None:
            print(f"[WARNING] Could not read {first_img_path}, skipping camera {cam_id}")
            continue

        height, width, _ = first_img.shape
        video_path = os.path.join(OUTPUT_DIR, f"{cam_id}.mp4")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))

        # Write frames to video
        frame_count = 0
        for img_name in images:
            img_path = os.path.join(DATASET_DIR, img_name)
            frame = cv2.imread(img_path)
            
            if frame is not None:
                # Resize if dimensions don't match
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
                frame_count += 1

        out.release()
        print(f"[OK] Created video: {video_path} ({frame_count} frames)")

    print("\n" + "="*60)
    print(f"[SUCCESS] Generated {len(camera_images)} fake CCTV videos")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    create_fake_videos()
