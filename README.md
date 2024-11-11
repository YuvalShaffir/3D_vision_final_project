# 3D Vision for Object Detection and Distance Estimation
![image](https://github.com/user-attachments/assets/f1ac6558-592c-4bf2-8589-5a3619dfc59c)

This project is designed to estimate the distance of a tennis ball from a single monocular camera by detecting the ball's size and position in the image. The process uses a combination of computer vision techniques, camera calibration, and machine learning to achieve accurate distance estimation.

### Writen by:
- Yuval Shaffir
- Michael Malka

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Calibration Process](#calibration-process)
- [Model Details](#model-details)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Contributing](#contributing)

---

## Project Overview

### Goal
Estimate the distance of an object—in this case, a tennis ball—from a monocular camera. This project uses a pinhole camera model with a geometric approach, combined with machine learning, to predict object distance accurately.

### Applications
- Sports analytics
- Robotics
- Cost-effective distance estimation solutions

### Key Techniques
- **Camera Calibration**: Using chessboard patterns to obtain intrinsic camera parameters.
- **Ellipse Detection**: Detecting and enhancing the ball’s appearance in images to extract key geometric features.
- **Distance Estimation**: Using Support Vector Regression (SVR) and geometric features to predict distance.

---

## Features
- **Camera Calibration**: Computes intrinsic camera parameters and distortion coefficients.
- **Ellipse Detection**: Locates the ball, detects its radius, and measures relevant features.
- **Distance Prediction**: Uses a machine learning model to predict object distance from the camera based on detected features.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YuvalShaffir/3D-Vision-Object-Detection.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
##  Usage
Run the main script to detect the ball and estimate its distance from the camera in test images.
  ```bash
  run calibrate.py
  ```

## Usage 
### Calibration Process
- Chessboard Pattern Setup: Capture images of a chessboard pattern in different positions.
Detect Corners: Use findChessboardCornersSB for corner detection and 3D-2D correspondence.
- Compute Camera Parameters: Use calibrateCamera to obtain the camera matrix and distortion coefficients.
Output Validation: Check re-projection error to validate calibration accuracy.

#### Center ball height VS radius ratio:
![myplot](https://github.com/user-attachments/assets/de2da406-19ad-40dc-bbc3-b923e51dbeac)

#### Ball distance from the center of the camera VS radius ratio:
![myplot2](https://github.com/user-attachments/assets/f3e65684-a9e9-4ee4-a4f3-25d16de2d07c)


## Model Details
This project uses Support Vector Regression (SVR) with polynomial features to handle non-linear relationships between object distance and features.

Key features include:
- Distance from the center of the image
- Radius
- aspect ratio
- area
- orientation angle of the detected ellipse

## Prediction
The model predicts the ball’s radius when it is at the image center, allowing for accurate distance estimation based on changes in apparent size.

## Results
The model provides reliable distance predictions for objects across varying distances and perspectives. Polynomial feature transformation and SVR improve accuracy by capturing complex feature relationships.

```bash
aspect ratio: 1.0547831330924
 area: 19320.0
Ball radius in direction of camera center: 92.76949310302722
Ellipse: Center = (1508.703125, 2054.5849609375), Width = 175.90249633789062, Height = 185.5389862060547, Angle = 84.2403335571289
predicted center radius: 86.29117272713827
Distance to the object from the camera: 109.29 cm
Real distance is: 100 cm


aspect ratio: 1.054053173632564
 area: 28804.5
Ball radius in direction of camera center: 98.74616241455077
Ellipse: Center = (1456.949462890625, 1527.8621826171875), Width = 187.3646697998047, Height = 197.49232482910156, Angle = 145.79238891601562
predicted center radius: 91.89240128088103
Distance to the object from the camera: 102.62 cm
Real distance is: 100 cm


aspect ratio: 1.024585640919084
 area: 33601.5
Ball radius in direction of camera center: 134.37966918945304
Ellipse: Center = (1264.3277587890625, 1523.717041015625), Width = 262.3102722167969, Height = 268.75933837890625, Angle = 172.87974548339844
predicted center radius: 126.74295473052373
Distance to the object from the camera: 74.41 cm
Real distance is: 70 cm

```

## Future Work
Real-time processing and application to video streams
Testing with different camera setups and object types
Enhancing robustness under varying lighting conditions and occlusions

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.
   
