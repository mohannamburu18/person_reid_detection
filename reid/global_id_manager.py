import cv2
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import torch
import torchvision.transforms as T
from torchvision.models import resnet50

class GlobalIDManager:
    def __init__(self, threshold=0.65, max_gallery_size=10, use_deep_features=True):
        """
        Enhanced Global ID Manager with deep learning features
        
        Args:
            threshold: Similarity threshold (lower = stricter matching)
            max_gallery_size: Number of features to store per person
            use_deep_features: Use ResNet50 (True) or HSV histogram (False)
        """
        self.db = {}  # global_id -> list of features (gallery)
        self.movements = {}  # global_id -> list of cameras
        self.last_seen_camera = {}  # global_id -> camera_id
        self.last_seen_time = {}  # global_id -> frame_number
        self.next_id = 1
        self.threshold = threshold
        self.max_gallery_size = max_gallery_size
        self.use_deep_features = use_deep_features
        
        # Camera topology for Market-1501 (adjust based on your setup)
        self.camera_connections = {
            'c1': ['c2', 'c3', 'c4'],
            'c2': ['c1', 'c3', 'c5'],
            'c3': ['c1', 'c2', 'c4', 'c6'],
            'c4': ['c1', 'c3', 'c5', 'c6'],
            'c5': ['c2', 'c4', 'c6'],
            'c6': ['c3', 'c4', 'c5']
        }
        
        if use_deep_features:
            self._init_deep_model()
        
    def _init_deep_model(self):
        """Initialize ResNet50 for feature extraction"""
        print("[INFO] Loading ResNet50 feature extractor...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet50
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove classification layer
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing for ResNet
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # Standard ReID size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"[INFO] Model loaded on {self.device}")
    
    def extract_feature(self, person_img):
        """
        Extract feature vector from person image
        
        Args:
            person_img: Cropped person image (BGR format)
            
        Returns:
            Normalized feature vector
        """
        if person_img.size == 0:
            return None
        
        # Check minimum size
        if person_img.shape[0] < 50 or person_img.shape[1] < 30:
            return None
        
        if self.use_deep_features:
            return self._extract_deep_feature(person_img)
        else:
            return self._extract_color_feature(person_img)
    
    def _extract_deep_feature(self, person_img):
        """Extract deep learning features using ResNet50"""
        try:
            # Convert BGR to RGB
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            
            # Preprocess and extract features
            img_tensor = self.transform(person_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.model(img_tensor)
                feature = feature.cpu().numpy().flatten()
                
                # L2 normalization
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm
                    
            return feature
        except Exception as e:
            print(f"[WARNING] Feature extraction failed: {e}")
            return None
    
    def _extract_color_feature(self, person_img):
        """Extract color histogram features (fallback method)"""
        try:
            hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
            
            # Multi-channel histogram
            hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Normalize
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            cv2.normalize(hist_v, hist_v)
            
            # Concatenate
            feature = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            
            return feature
        except Exception as e:
            print(f"[WARNING] Color feature extraction failed: {e}")
            return None
    
    def _calculate_similarity(self, feature1, feature2):
        """Calculate similarity score between two features"""
        if self.use_deep_features:
            # Cosine distance for deep features
            return cosine(feature1, feature2)
        else:
            # Euclidean distance for histograms
            return euclidean(feature1, feature2)
    
    def assign_global_id(self, feature, camera_id, frame_number=0):
        """
        Assign global ID to a person based on feature matching
        
        Args:
            feature: Feature vector
            camera_id: Camera identifier (e.g., 'c1', 'c2')
            frame_number: Current frame number for temporal constraints
            
        Returns:
            Global ID (integer)
        """
        if feature is None:
            return None
        
        best_id = None
        best_score = float('inf')
        
        # Match against all existing persons in database
        for gid, gallery_features in self.db.items():
            # Calculate minimum distance to gallery
            distances = [self._calculate_similarity(feature, gf) for gf in gallery_features]
            feat_score = min(distances)  # Best match in gallery
            
            # Apply spatial constraint (camera topology)
            last_cam = self.last_seen_camera.get(gid)
            if last_cam and last_cam != camera_id:
                # Check if cameras are connected
                if camera_id not in self.camera_connections.get(last_cam, []):
                    feat_score *= 1.3  # Penalize unlikely transitions
            
            # Apply temporal constraint (realistic travel time)
            last_time = self.last_seen_time.get(gid, 0)
            time_diff = frame_number - last_time
            
            if time_diff < 15:  # Too quick to move between cameras
                feat_score *= 1.2
            elif time_diff > 500:  # Too long, probably different person
                feat_score *= 1.1
            
            # Track best match
            if feat_score < best_score:
                best_score = feat_score
                best_id = gid
        
        # Decision: match existing person or create new ID
        if best_score < self.threshold:
            # Match found - update gallery
            self.db[best_id].append(feature)
            
            # Maintain gallery size limit
            if len(self.db[best_id]) > self.max_gallery_size:
                self.db[best_id].pop(0)  # Remove oldest feature
            
            # Update movement tracking
            if camera_id not in self.movements[best_id]:
                self.movements[best_id].append(camera_id)
            
            # Update temporal/spatial info
            self.last_seen_camera[best_id] = camera_id
            self.last_seen_time[best_id] = frame_number
            
            return best_id
        else:
            # No match - create new person
            gid = self.next_id
            self.db[gid] = [feature]
            self.movements[gid] = [camera_id]
            self.last_seen_camera[gid] = camera_id
            self.last_seen_time[gid] = frame_number
            self.next_id += 1
            
            return gid
    
    def get_statistics(self):
        """Get ReID statistics"""
        total_persons = len(self.db)
        total_movements = sum(len(cams) > 1 for cams in self.movements.values())
        
        return {
            'total_persons': total_persons,
            'cross_camera_tracks': total_movements,
            'avg_gallery_size': np.mean([len(g) for g in self.db.values()]) if self.db else 0
        }
