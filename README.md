# Cricket_analysis
AI model for Cricket biomechanics analysis using MediaPipe, OpenCV, and 3D skeleton visualizations. 

Dataset: https://drive.google.com/drive/folders/1ACgf4r5BjZjPluHfeK5vKbrNbEtQib6A?usp=drive_link

Download these videos and then provide the path for each of them in the code. 

Due to large size of output files, I have uploaded them on drive: https://drive.google.com/drive/folders/1WdhLsw1cwDY9IqbiLDX77xzQcfvrBeXb?usp=sharing


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


**Analysis Workflow**
1. Pose Detection & Keypoint Extraction
Library: MediaPipe Pose

Detects 33 body landmarks per frame with (x, y, z) coordinates & visibility score.

Stores results in CSV format for analysis.

2️. Action Classification
Uses heuristic rules on landmark positions:

Bowling: High wrist position at release.

Batting: Stable stance, symmetric knees.

Follow-through: Large rotation + forward motion.

Fielding: Extended arms, crouch posture.

3️. Biomechanical Analysis
Batting → Stance angle, bat angle, pre-shot movement

Bowling → Run-up stride length, elbow extension at release

Follow-through → Stability, upper-lower body torque

Fielding → Dive form, throwing arm extension

4️. 3D Skeleton Visualization
Plots detected landmarks in 3D using Matplotlib.

Connects joints to form a moving skeleton.

Compiles frames into a 3D motion video.

5️. Output
CSV files: Store action-specific biomechanical data.

MP4 video: 3D skeleton animation for motion replay.


