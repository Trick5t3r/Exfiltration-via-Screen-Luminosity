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
- **`/records`**: Contains video and csv files used for testing the algorithms.
- **`README.md`**: Provides an overview and documentation of the project.

---

## How to Run the Code

0. Clone the repository:
   ```bash
   git clone https://github.com/Trick5t3r/Exfiltration-via-Screen-Luminosity.git
   cd Exfiltration-via-Screen-Luminosity
   pip install -r requirements.txt
   ```

1. Get the weigths of YOLOv3
- Go to "https://pjreddie.com/darknet/yolo/" and download "yolov3-tiny.weights" (too big for github)
- Put it in the "yolo_files" folder

2. Run this code on the target :
   ```bash
   python src/screen_leak_detection_person.py
   ```

4. Use a camera and run this code
   ```bash
    python src/track_screen_yolo_lstq.py
   ```

---

# Alternatives

## Algorithm Descriptions

### 1. **`track_screen_yolo_naive`**
This algorithm implements a method based on the second derivative of the signal.

### 2. **`track_screen_yolo_lstq`**
This algorithm implements a method using least squares (`lstq`).

### 3. **`screen_tracker_yolo_to_csv`**
This algorithm extracts information from a video and saves it into a CSV file.

### 4. **`test_reconstruction_decalage`**
This algorithm tests the reconstruction of a signal using three noisy and shifted signals.

### 5. **`screen_leak_brightness`**
This algorithm leaks a message by exploiting the brightness of the screen.

### 6. **`screen_leak_detection_person`**
This algorithm leaks a message by capturing a screenshot and halts the process if a person is detected in front of the screen.

## Test Files in `/records`
In the `/records` folder, you will find test files named `final_00x`, which are used to test the algorithm.  
To run a test, simply set `video_path` to a specific file, e.g., `"records/final_003.mov"`.

## CSV Files in `/records/luminosity_signal`

In the folder `/records/luminosity_signal`, we have stored **CSV files** containing the extracted brightness data from our test videos.  
These files are provided so you can test the **`message_decoder`** algorithm.




