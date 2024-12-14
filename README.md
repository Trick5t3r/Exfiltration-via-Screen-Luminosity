# Air-Gapped Data Transmission via Screen Brightness and Color Variations

## Introduction

This project focuses on exfiltrating data from an air-gapped computer (one isolated from all networks such as Internet and Bluetooth). Our approach leverages variations in screen brightness and colors to encode and transmit data. The variations are captured by a controlled camera (e.g., a compromised security camera or one with long-range zoom capabilities).

This repository contains the code, methods, and results of our study.

---

## Objectives

1. **Data Exfiltration**: Transmit binary data by modulating the brightness and colors of the screen.
2. **Signal Capture**: Detect variations using an external camera.
3. **Signal Processing**: Decode the captured data into the original binary signal.
4. **Perspective:Environmental Adaptation**: Compensate for external interferences and environmental changes during the transmission.

---

## Methods

### Signal Generation
- **Binary Encoding**: Information is transmitted as binary values represented by screen brightness or color variations.
- **Color Channel Modulation**: Each pixel's RGB values are adjusted slightly to transmit data discreetly without visible changes to the human eye.

### Signal Capture
- **Rectangle Detection**: Detects the screen area in the camera feed using edge detection and contour analysis.
- **YOLOv2 Detection**: Enhances detection by identifying screens (`tvmonitor`) using the YOLOv2 deep learning model.

### Signal Processing
- **Naive Approach**: Spike detection based on strong variations in brightness.
- **Least Squares Optimization**: Fitting the signal to minimize errors and accurately reconstruct the transmitted binary data.

---

## Key Results

- Successful signal reconstruction in controlled environments using both naive and optimized methods.
- Developed a robust method for tracking and analyzing screen regions over time, even with environmental noise.

---

## Challenges

1. **Automatic Brightness Adjustments**: Modern cameras and monitors often auto-adjust their brightness and colors, complicating data consistency.
2. **Blue Light Filters**: These filters alter the displayed colors, introducing noise into the captured signal.
3. **Environmental Interference**: External light sources and reflections affect signal reliability.

---

## Improvements and Future Work

- **Environmental Understanding**: Develop an algorithm to analyze and adapt to environmental changes dynamically.
- **Error Correction**: Integrate error correction techniques inspired by TCP protocols to enhance reliability.
- **Machine Learning**: Train models for signal denoising, leveraging the repetitive nature of environmental noise.

---

## Repository Contents

- **`/src`**: Contains Python scripts for signal generation, data capture, and processing.
- **`/src/files_under_test`**: Experimental scripts for testing and refining signal processing ideas.
- **`/yolo_files`**: Includes pre-trained YOLOv2 weights for object detection.
- **`/records`**: Contains video files used for testing the algorithms.
- **`README.md`**: Provides an overview and documentation of the project.

---

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/Trick5t3r/Exfiltration-via-Screen-Luminosity.git
   cd Exfiltration-via-Screen-Luminosity
   pip install -r requirements.txt```

2. Run this code on the target
   ```bash
   cd src```
   ```bash
   python screen_leak_detection_person.py```

3. Use a camera and run this code
   ```bash
    python track_screen_yolo.py ```