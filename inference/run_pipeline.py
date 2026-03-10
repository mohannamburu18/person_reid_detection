from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict, deque
from reid.global_id_manager import GlobalIDManager

# ================= CONFIGURATION =================
FRAME_SKIP = 2                    # Process every 2nd frame (adjust for performance)
MIN_FRAMES_TO_CONFIRM = 5         # Frames needed for stable detection
LOST_THRESHOLD = 45               # Frames before marking as LOST
MIN_DETECTION_CONF = 0.45         # Minimum confidence for person detection
MIN_PERSON_HEIGHT = 100           # Minimum bounding box height
MIN_PERSON_WIDTH = 40             # Minimum bounding box width
MAX_ASPECT_RATIO = 4.5            # Maximum height/width ratio
MIN_ASPECT_RATIO = 1.5            # Minimum height/width ratio

# ReID parameters
USE_DEEP_FEATURES = True          # Use ResNet50 (True) or color histogram (False)
REID_THRESHOLD = 0.65             # Similarity threshold (lower = stricter)
MAX_GALLERY_SIZE = 10             # Features to store per person
# =================================================

class BBoxSmoother:
    """Smooth bounding box coordinates to reduce jitter"""
    def __init__(self, maxlen=5):
        self.history = defaultdict(lambda: deque(maxlen=maxlen))

    def smooth(self, gid, box):
        self.history[gid].append(box)
        if len(self.history[gid]) == 0:
            return box
        return np.mean(self.history[gid], axis=0).astype(int)

def is_valid_detection(box, frame_shape):
    """
    Filter out invalid detections based on size and aspect ratio
    
    Args:
        box: [x1, y1, x2, y2]
        frame_shape: (height, width, channels)
        
    Returns:
        Boolean indicating if detection is valid
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Check minimum size
    if width < MIN_PERSON_WIDTH or height < MIN_PERSON_HEIGHT:
        return False
    
    # Check aspect ratio (people are taller than wide)
    aspect_ratio = height / width if width > 0 else 0
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
    
    # Check if box is within frame boundaries
    if x1 < 0 or y1 < 0 or x2 > frame_shape[1] or y2 > frame_shape[0]:
        return False
    
    return True

def run_pipeline():
    """Main ReID pipeline execution"""
    
    # Setup paths
    video_dir = "data/fake_cctv_videos"
    videos = sorted([v for v in os.listdir(video_dir) if v.endswith(".mp4")])

    if not videos:
        print("[ERROR] No videos found in", video_dir)
        print("[INFO] Please run 'python data_prep/images_to_video.py' first")
        return

    print(f"[INFO] Found {len(videos)} camera videos")
    print(f"[INFO] Processing first 3 cameras for demo...")
    
    # Initialize models
    print("[INFO] Loading YOLO model...")
    model = YOLO("yolov8m.pt")  # Medium model for better accuracy
    
    print("[INFO] Initializing ReID manager...")
    reid = GlobalIDManager(
        threshold=REID_THRESHOLD,
        max_gallery_size=MAX_GALLERY_SIZE,
        use_deep_features=USE_DEEP_FEATURES
    )
    
    smoother = BBoxSmoother(maxlen=5)

    # Tracking state
    frame_count = 0
    seen_frames = defaultdict(int)  # global_id -> frame count
    last_seen = {}                   # global_id -> last frame number
    active_ids = set()               # Currently visible IDs
    
    print("\n" + "="*60)
    print("STARTING PERSON RE-IDENTIFICATION PIPELINE")
    print("="*60)

    # Process videos (first 3 cameras for demo)
    for video_idx, video_name in enumerate(videos[:3], 1):
        camera_id = video_name.split(".")[0]  # e.g., 'c1', 'c2', etc.
        video_path = os.path.join(video_dir, video_name)

        print(f"\n[CAMERA {video_idx}/3] Processing {camera_id}...")
        print(f"[INFO] Video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            continue
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] FPS: {fps}, Total frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Frame skipping for performance
            if frame_count % FRAME_SKIP != 0:
                continue

            # Detect and track persons
            results = model.track(
                frame,
                persist=True,
                classes=[0],           # Person class only
                conf=MIN_DETECTION_CONF,
                iou=0.4,               # Lower IoU for better separation
                tracker="botsort.yaml", # Better tracker
                verbose=False
            )

            if results[0].boxes.id is None:
                # No detections in this frame
                cv2.imshow("ReID Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            current_frame_ids = set()

            # Process each detection
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Validate detection quality
                if not is_valid_detection([x1, y1, x2, y2], frame.shape):
                    continue

                # Extract person crop
                person = frame[y1:y2, x1:x2]

                if person.size == 0:
                    continue
                
                # Check image quality (avoid too dark images)
                if np.mean(person) < 15:
                    continue

                # Extract ReID feature
                feature = reid.extract_feature(person)
                if feature is None:
                    continue
                
                # Assign global ID
                gid = reid.assign_global_id(feature, camera_id, frame_count)
                
                if gid is None:
                    continue

                # Update tracking statistics
                seen_frames[gid] += 1
                last_seen[gid] = frame_count
                current_frame_ids.add(gid)

                # Only show stable detections (confirmed persons)
                if seen_frames[gid] < MIN_FRAMES_TO_CONFIRM:
                    continue

                # Smooth bounding box
                smooth_box = smoother.smooth(gid, np.array([x1, y1, x2, y2]))

                # Draw bounding box
                color = (0, 255, 0)  # Green for tracked persons
                cv2.rectangle(
                    frame,
                    (smooth_box[0], smooth_box[1]),
                    (smooth_box[2], smooth_box[3]),
                    color,
                    2
                )

                # Draw label
                label = f"Person {gid}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Label background
                cv2.rectangle(
                    frame,
                    (smooth_box[0], smooth_box[1] - label_size[1] - 10),
                    (smooth_box[0] + label_size[0], smooth_box[1]),
                    color,
                    -1
                )
                
                # Label text
                cv2.putText(
                    frame,
                    label,
                    (smooth_box[0], smooth_box[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

                # Log re-detection across cameras
                if gid not in active_ids:
                    cameras_visited = len(reid.movements.get(gid, []))
                    if cameras_visited > 1:
                        print(f"✓ [RE-DETECTED] Person {gid} in {camera_id} "
                              f"(visited {cameras_visited} cameras)")
                    active_ids.add(gid)

            # Detect lost persons
            for gid in list(active_ids):
                if frame_count - last_seen.get(gid, 0) > LOST_THRESHOLD:
                    print(f"✗ [LOST] Person {gid} (last seen in {reid.last_seen_camera.get(gid, 'unknown')})")
                    active_ids.remove(gid)

            # Display info on frame
            info_text = f"Camera: {camera_id} | Frame: {frame_count} | Active: {len(current_frame_ids)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow("ReID Pipeline", frame)
            
            # Press ESC to exit
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap.release()
        print(f"[INFO] Finished processing {camera_id}")

    cv2.destroyAllWindows()

    # Print final statistics
    print("\n" + "="*60)
    print("PERSON MOVEMENT SUMMARY")
    print("="*60)
    
    stats = reid.get_statistics()
    print(f"\nTotal Unique Persons: {stats['total_persons']}")
    print(f"Cross-Camera Tracks: {stats['cross_camera_tracks']}")
    print(f"Average Gallery Size: {stats['avg_gallery_size']:.2f}")
    
    print("\n--- Movement Paths ---")
    for gid, cameras in sorted(reid.movements.items()):
        if len(cameras) > 1:
            path = " → ".join(cameras)
            print(f"Person {gid:3d}: {path}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
