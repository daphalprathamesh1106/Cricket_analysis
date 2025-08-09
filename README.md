# Cricket_analysis
AI model for Cricket biomechanics analysis using MediaPipe, OpenCV, and 3D skeleton visualizations. 

##Dataset: https://drive.google.com/drive/folders/1ACgf4r5BjZjPluHfeK5vKbrNbEtQib6A?usp=drive_link


# Cricket Multi-Role Motion Analysis with 3D Skeletons

## Overview
This project analyzes cricket player performance from videos using **pose estimation** and **biomechanics calculations**.  
It automatically detects and extracts **Batting**, **Bowling**, **Follow-through**, and **Fielding** sequences,  
then produces both:
- **Biomechanical metrics** (stance angles, release dynamics, etc.)
- **3D skeleton visualizations** (animation + export to video)

**Frameworks Used:**  
- [MediaPipe](https://github.com/google/mediapipe) for 2D/3D keypoint extraction  
- [OpenCV](https://opencv.org/) for video processing  
- [Matplotlib](https://matplotlib.org/) for 3D skeleton rendering  
- [NumPy](https://numpy.org/) for biomechanics math  

---

## Key Features

### 1. **Batting Mechanics**
- Detects **stance**  
- Tracks **trigger movement**  
- Calculates **bat angle**  
- Measures **shot timing** & **selection patterns**  

### 2. **Bowling Mechanics**
- **Run-up consistency** (stride length stability)  
- **Load-up position** detection  
- **Front foot landing** accuracy  
- **Release dynamics** (elbow, wrist, shoulder angles)  

### 3. **Follow-through**
- **Balance stability** after execution  
- **Wrist motion** tracking  
- **Shoulder–hip torque** measurement  

### 4. **Fielding Mechanics**
- **Anticipation reaction time** (from ball trajectory cues)  
- **Dive mechanics**  
- **Throwing angle and arm extension**  


