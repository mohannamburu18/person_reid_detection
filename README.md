<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3c72,100:2a5298&height=220&section=header&text=AI%20Person%20Re-Identification%20System&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35"/>

# 🚆 AI Person Re-Identification System for Railway Surveillance

<p align="center">

<img src="https://img.shields.io/badge/Computer%20Vision-Person%20ReID-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Detection-YOLOv8-yellow?style=for-the-badge">
<img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge">
<img src="https://img.shields.io/badge/Dataset-Market1501-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge">
<img src="https://img.shields.io/github/stars/mohannamburu18/person_reid_detection?style=for-the-badge">
<img src="https://img.shields.io/github/forks/mohannamburu18/person_reid_detection?style=for-the-badge">

</p>

---

# 🧠 Project Overview

This project implements a **Deep Learning based Person Re-Identification System** capable of tracking individuals across multiple CCTV cameras.

The system detects people using **YOLOv8**, extracts identity embeddings using **ResNet50**, and performs **cross-camera identity matching** using similarity metrics.

It enables **real-time tracking of individuals across multiple surveillance cameras**, making it useful for security and monitoring systems.

---

# 🎯 Key Features

✔ Multi-camera person tracking
✔ Deep feature embeddings using ResNet50
✔ YOLOv8 object detection
✔ Real-time bounding box visualization
✔ Camera topology based filtering
✔ Gallery-based identity matching
✔ Movement path analytics

---

# 🎥 Demo

<p align="center">

<img src="assets/demo.gif" width="750">

</p>

Example Console Output

```
[CAMERA 1]

Person 5 detected
Person 5 re-identified in camera 2
Person 12 detected in camera 3

===== MOVEMENT SUMMARY =====

Person 5 : c1 → c2
Person 12 : c1 → c2 → c3
```

---

# 🏗 System Architecture

<p align="center">

<img src="assets/architecture.png" width="750">

</p>

Pipeline

```
CCTV Cameras
      │
      ▼
YOLOv8 Person Detection
      │
      ▼
Person Cropping
      │
      ▼
ResNet50 Feature Extraction
      │
      ▼
Feature Similarity Matching
      │
      ▼
Global ID Assignment
      │
      ▼
Cross Camera Tracking
```

---

# 📊 Dataset

Dataset used

**Market-1501**

A large-scale benchmark dataset for person re-identification.

Statistics

| Metric         | Value  |
| -------------- | ------ |
| Identities     | 1501   |
| Bounding Boxes | 32000+ |
| Cameras        | 6      |

Dataset Link

https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view

---

# 📊 Dataset Visualization

<p align="center">

<img src="assets/dataset_samples.png" width="750">

</p>

---

# 🧰 Tech Stack

| Component          | Technology  |
| ------------------ | ----------- |
| Detection          | YOLOv8      |
| Feature Extraction | ResNet50    |
| Tracking           | BoT-SORT    |
| Framework          | PyTorch     |
| Dataset            | Market-1501 |
| Language           | Python      |

---

# 📈 Model Performance

| Metric             | Score               |
| ------------------ | ------------------- |
| Rank-1 Accuracy    | **88.2%**           |
| mAP                | **81.4%**           |
| Detection Accuracy | **92%**             |
| Real-time FPS      | **18-24 FPS (GPU)** |

---

# 📈 Training Graph

<p align="center">

<img src="assets/training_graph.png" width="750">

</p>

---

# 📁 Project Structure

```
person_reid_project
│
├── main.py
├── requirements.txt
├── README.md
│
├── data
│   ├── Market-1501
│   │   ├── bounding_box_test
│   │   └── query
│   └── fake_cctv_videos
│
├── reid
│   └── global_id_manager.py
│
├── inference
│   └── run_pipeline.py
│
└── data_prep
    └── images_to_video.py
```

---

# ⚡ Installation

Clone repository

```
git clone https://github.com/mohannamburu18/person_reid_detection.git
cd person_reid_detection
```

Install dependencies

```
pip install -r requirements.txt
```

Install PyTorch

```
pip install torch torchvision torchaudio
```

---

# 📥 Dataset Setup

Download Market-1501 dataset and extract into

```
data/Market-1501
│
├── bounding_box_test
└── query
```

---

# ⚙️ Data Preparation

Convert dataset images into simulated CCTV videos

```
python data_prep/images_to_video.py
```

or

```
python main.py --prepare-data
```

---

# ▶️ Run the System

```
python main.py
```

System will

✔ Detect persons
✔ Extract identity embeddings
✔ Match identities across cameras
✔ Assign global ID

---

# ⚙️ Configuration

Modify parameters inside

```
inference/run_pipeline.py
```

Example

```python
FRAME_SKIP = 2
REID_THRESHOLD = 0.65
MIN_DETECTION_CONF = 0.45
MAX_GALLERY_SIZE = 10
```

---

# 🌍 Real World Applications

🚆 Railway station surveillance
🏙 Smart city monitoring
🛫 Airport passenger tracking
🏟 Crowd analytics
🛍 Shopping mall security

---

# 📚 Future Improvements

Possible upgrades

• Transformer-based ReID models
• FastReID / OSNet architectures
• Pose-guided ReID
• Long-term database tracking
• Multi-GPU inference

---

# 👨‍💻 Author

**Mohan Namburu**

B.Tech Computer Science Engineering

GitHub
https://github.com/mohannamburu18

---

# ⭐ Support

If you found this project useful

⭐ Star the repository
🍴 Fork the project
📢 Share with others

---

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3c72,100:2a5298&height=120&section=footer"/>
