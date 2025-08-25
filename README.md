# Face Recognition Project

A comprehensive face recognition system built with Python and OpenCV that can detect and recognize faces in images and real-time video streams.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Training Your Own Model](#training-your-own-model)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a face recognition system using OpenCV's Haar Cascade classifiers and Local Binary Pattern Histograms (LBPH) face recognizer. The system can:
- Detect faces in static images
- Recognize faces from a trained dataset
- Perform real-time face recognition using webcam
- Detect eyes and other facial features

## ✨ Features

- **Face Detection**: Detect faces in images using Haar Cascade classifiers
- **Face Recognition**: Recognize faces using LBPH (Local Binary Pattern Histograms) algorithm
- **Real-time Processing**: Live face recognition through webcam
- **Eye Detection**: Additional feature to detect eyes within detected faces
- **Training Module**: Train the system with your own dataset
- **Multiple Cascade Support**: Includes various Haar cascade files for different detection tasks

## 📁 Project Structure

```
face_recognition/
├── README.md                          # This file
├── face_detection.py                  # Basic face detection in static images
├── face_Recognition.py                # Face recognition functions (has syntax errors)
├── FaceRecognition.py                 # Main face recognition module
├── tester.py                         # Test face recognition on static images
├── trainer.py                        # Train the face recognition model
├── video_FaceRecognition.py          # Video-based face recognition functions
├── video.py                          # Real-time face and eye detection
├── videoTester.py                    # Real-time face recognition testing
├── haarcascade_eye.xml               # Haar cascade for eye detection
├── __pycache__/                      # Python cache files
└── harcasscade_files/                # Additional cascade files and examples
    ├── face_detection.py             # Alternative face detection implementation
    ├── faces_detection.py            # Simple face detection
    ├── live_cam_detector.py          # Live camera face detection
    ├── pedastian_detector.py         # Pedestrian detection (incomplete)
    ├── IMG-20191001-WA0115.jpg       # Sample test image
    └── [various .xml files]          # Additional Haar cascade classifiers
```

## 🔧 Prerequisites

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy
- A webcam (for real-time features)

## 🚀 Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd face_recognition
   ```

2. **Install required packages**
   ```bash
   pip install opencv-python
   pip install opencv-contrib-python
   pip install numpy
   ```

3. **Download Haar Cascade files**
   - The project requires `haarcascade_frontalface_default.xml`
   - Download from [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades)
   - Place in the project root directory

## 🎮 Usage

### Basic Face Detection
```bash
python face_detection.py
```
Detects faces in a static image and displays the result with bounding boxes.

### Real-time Face and Eye Detection
```bash
python video.py
```
Opens webcam and detects faces and eyes in real-time. Press 'ESC' to exit.

### Train Face Recognition Model
1. **Prepare your dataset**:
   - Create a directory structure: `photo/person_id/images.jpg`
   - Each person should have a unique numeric ID as folder name
   - Place multiple photos of each person in their respective folders

2. **Train the model**:
   ```bash
   python trainer.py
   ```
   This will create a `trainingData.yml` file containing the trained model.

### Test Face Recognition (Static Images)
```bash
python tester.py
```
Tests the trained model on a static image and displays the recognized person's name.

### Real-time Face Recognition
```bash
python videoTester.py
```
Performs real-time face recognition using webcam. Press 'q' to quit.

## 📄 File Descriptions

### Core Modules

- **`FaceRecognition.py`**: Main module containing all face recognition functions
  - `faceDetection()`: Detects faces in images
  - `labels_for_training_data()`: Processes training dataset
  - `train_classifier()`: Trains the LBPH face recognizer
  - `draw_rect()`: Draws bounding boxes around faces
  - `put_text()`: Adds person names to detected faces

- **`tester.py`**: Tests face recognition on static images
  - Loads a pre-trained model from `trainingData.yml`
  - Recognizes faces in test images
  - Displays confidence scores and predicted names

- **`videoTester.py`**: Real-time face recognition using webcam
  - Captures video from default camera
  - Performs face recognition on each frame
  - Shows live results with name labels

### Detection Modules

- **`face_detection.py`**: Basic face detection in static images
- **`video.py`**: Real-time face and eye detection without recognition
- **`trainer.py`**: Alternative training script (has some syntax issues)

### Additional Files

- **`video_FaceRecognition.py`**: Contains functions for video-based face recognition
- **`face_Recognition.py`**: Alternative face recognition functions (contains syntax errors)

## 🎓 Training Your Own Model

1. **Prepare Dataset**:
   ```
   training_data/
   ├── 0/          # Person ID 0
   │   ├── img1.jpg
   │   ├── img2.jpg
   │   └── ...
   ├── 1/          # Person ID 1
   │   ├── img1.jpg
   │   └── ...
   └── 2/          # Person ID 2
       └── ...
   ```

2. **Update Paths**:
   - Modify the dataset path in training scripts
   - Update the name dictionary in `tester.py` and `videoTester.py`:
   ```python
   name = {0: "Person1", 1: "Person2", 2: "Person3"}
   ```

3. **Train the Model**:
   - Run the training script to generate `trainingData.yml`
   - This file contains the trained face recognition model

## ⚙️ Configuration

### Important Paths to Update

Before running the scripts, update these paths according to your system:

1. **Haar Cascade Path** (in multiple files):
   ```python
   face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   ```

2. **Training Data Path** (in `tester.py`, `trainer.py`):
   ```python
   faces, faceID = fr.labels_for_training_data('path/to/your/training/data')
   ```

3. **Test Image Path** (in `tester.py`):
   ```python
   test_img = cv2.imread('path/to/your/test/image.jpg')
   ```

4. **Model Save/Load Path**:
   ```python
   face_recognizer.read('trainingData.yml')
   ```

### Recognition Parameters

- **Confidence Threshold**: Adjust in `tester.py` and `videoTester.py`
  ```python
  if confidence < 60:  # Lower values = more strict recognition
  ```

- **Detection Parameters**: Modify in `faceDetection()` function
  ```python
  faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=4)
  ```

## 🔧 Troubleshooting

### Common Issues

1. **"File not found" errors**:
   - Ensure all file paths are correct for your system
   - Download required Haar cascade XML files
   - Check image file paths and formats

2. **"No module named cv2"**:
   ```bash
   pip install opencv-python opencv-contrib-python
   ```

3. **Empty training data**:
   - Ensure your training images contain clear, single faces
   - Check that face detection is working on your training images
   - Verify directory structure and naming conventions

4. **Poor recognition accuracy**:
   - Add more training images per person (recommended: 20-50 images)
   - Ensure good lighting and clear faces in training data
   - Adjust confidence threshold values
   - Use consistent image quality and angles

5. **Camera not working**:
   - Check if camera is being used by another application
   - Try different camera indices: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

### Code Issues Found

Some files contain syntax errors that need to be fixed:

- **`face_Recognition.py`**: Contains syntax errors and incomplete functions
- **`trainer.py`**: Has import and function call errors
- **`live_cam_detector.py`**: Contains syntax errors in the loop condition and variable names

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Fix any existing bugs or add new features
4. Test your changes thoroughly
5. Submit a pull request

### Areas for Improvement

- Fix syntax errors in several files
- Add error handling and validation
- Implement better face detection confidence scoring
- Add support for multiple face recognition algorithms
- Create a GUI interface
- Add face enrollment features
- Implement face verification vs identification modes

## 📝 Notes

- The system works best with clear, front-facing images
- Good lighting conditions improve recognition accuracy
- Multiple training images per person enhance performance
- The LBPH algorithm is robust to lighting changes but sensitive to pose variations

## 📜 License

This project is for educational purposes. Please ensure you have proper permissions when using facial recognition technology and comply with local privacy laws and regulations.

---

**Created by**: Asthir Kushwaha  
**Last Updated**: August 2025  
**Version**: 1.0
