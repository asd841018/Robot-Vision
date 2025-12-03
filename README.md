# Robot Vision Course

> Assignments and lab exercises for the Robot Vision module

This repository contains all assignments and lab exercises for the Robot Vision course, covering core computer vision concepts from basic image processing to depth estimation and 3D reconstruction.

---

## Project Structure

```
robotvision/
├── AssignmentOne/          # Assignment 1: Image Stitching
├── Assignment2-task1-2/    # Assignment 2: Depth Estimation, Photometric Stereo & LLM Evaluation
├── RobotVision25/          # Weekly Lab Exercises
│   ├── Lab1/               # Image Basics & Edge Detection
│   ├── Lab2/               # Pinhole Camera Model & Projection
│   ├── Lab3/               # 3D Geometry & Camera Projection
│   └── Lab4/               # Camera Calibration & Distortion Correction
└── README.md
```

---

## Assignments

### Assignment 1: Image Stitching

**Directory**: `AssignmentOne/`

**Objective**: Implement panoramic image stitching with RANSAC algorithm

**Core Topics**:
- **Feature Detection**: SIFT, ORB, Harris Corner detectors
- **Feature Matching**: David Lowe's Ratio Test
- **Homography Estimation**: Robust estimation using RANSAC
- **Image Blending**: Linear Blending (Feathering)

**Key Files**:
| File | Description |
|------|-------------|
| `Assignment1_Guocheng.ipynb` | Main notebook with complete implementation and analysis |
| `stitch.py` | Core stitching class (`Stitcher`) |
| `blending.py` | Image blending methods (`Blender`) |
| `imgs/` | Test images (3 image pairs) |
| `results/` | Output results |

**Technical Highlights**:
- Complete image stitching pipeline implementation
- Support for multiple feature detector comparison
- Quantitative evaluation using reprojection error
- Handling of varying exposure and parallax issues

---

### Assignment 2: Depth Estimation & Photometric Stereo

**Directory**: `Assignment2-task1-2/`

#### Task 1: Depth Estimation Comparison (10%)

**Objective**: Compare the performance of different depth estimation methods

**Methods Compared**:
| Method | Description |
|--------|-------------|
| **Stereo Depth** | Stereo reconstruction from image pairs |
| **Monodepth (Depth Anything V2)** | Monocular depth estimation model |
| **LiDAR** | Simulated LiDAR scan data |

**Evaluation Metrics**:
- Qualitative: Depth map visualization, viewpoint reprojection
- Quantitative: L1 difference maps, error statistical analysis

**Resources**: 
- [Depth Anything V2](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2)

#### Task 2: RV and LLMs (10%)

**Objective**: Evaluate how effectively LLMs can answer Robot Vision curriculum questions

**Models Tested**:
- ChatGPT
- Anthropic Claude  
- DeepSeek

**Evaluation Criteria**:
- **Correctness** (1-5): Is the answer correct?
- **Explanation Quality** (1-5): Is the reasoning clear and detailed?
- **Consistency**: Does the model change its answer when challenged?

**Key Files**:
| File | Description |
|------|-------------|
| `Assignment2.ipynb` | Task 1 depth estimation implementation |
| `Part2-1.ipynb` | Task 2 LLM evaluation |
| `Depth-Anything-V2/` | Depth estimation model code |

---

## Weekly Labs

**Directory**: `RobotVision25/`

### Lab 1: Image Basics & Edge Detection
- Image loading and display (scikit-image)
- Sobel edge detection
- Convolution fundamentals

### Lab 2: Image Formation & Camera Fundamentals
- Pinhole camera model simulation
- Perspective projection
- Effect of focal length variation on projection

### Lab 3: 3D Geometry & Camera Projection
- Extrinsic matrix calculation
- Intrinsic matrix definition
- Point cloud rasterization
- Open3D point cloud processing

### Lab 4: Calibration & Photometric Image Formation
- Chessboard camera calibration
- Radial and tangential distortion correction
- OpenCV calibration functions
- Single view metrology

---

## Tech Stack

### Languages & Environment
- Python 3.x
- Jupyter Notebook

### Main Packages
```python
# Image Processing
opencv-python (cv2)
scikit-image (skimage)
PIL / Pillow

# Numerical Computing
numpy
scipy

# Visualization
matplotlib

# 3D Processing
open3d

# Deep Learning (Assignment 2)
torch
transformers
```

### Install Dependencies
```bash
pip install numpy scipy matplotlib opencv-python scikit-image pillow open3d
```

---

## Course Topics Overview

| Week | Topic | Lab/Assignment |
|------|-------|----------------|
| 1 | Image Basics, Edge Detection | Lab 1 |
| 2 | Pinhole Camera, Perspective Projection | Lab 2 |
| 3 | 3D Geometry, Projection Matrices | Lab 3 |
| 4 | Camera Calibration, Distortion Correction | Lab 4 |
| - | Feature Matching, Image Stitching, RANSAC | Assignment 1 |
| - | Depth Estimation, Photometric Stereo | Assignment 2 |

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd robotvision
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Select a notebook to run**
   - Lab exercises: `RobotVision25/Lab*/`
   - Assignment 1: `AssignmentOne/Assignment1_Guocheng.ipynb`
   - Assignment 2: `Assignment2-task1-2/Assignment2.ipynb`

---

## Notes

- Some packages may require additional installation (e.g., `open3d`, deep learning models)
- In the vLab environment, install missing packages using:
  ```python
  %pip install <package-name>
  ```
- Remember to restart the kernel after installing new packages

---

## 📄 License

This project is for academic purposes only.
