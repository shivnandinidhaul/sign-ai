#### SignAI

SignAI is a real-time American Sign Language (ASL) fingerspelling recognition system built using Python, OpenCV, MediaPipe, and machine learning.

It enables real-time classification of static ASL alphabet gestures using a standard webcam, without requiring specialized hardware or cloud processing.

Communication barriers can arise when Deaf and hard-of-hearing individuals use sign language in environments where others are not familiar with it, as well as makes sign language more accessible to understand. SignAI aims to address this by enabling real-time gesture recognition using lightweight, on-device machine learning.

### Overview

SignAI captures live video from a webcam, detects hand landmarks using MediaPipe, and predicts ASL alphabet gestures using a trained machine learning model. The prediction is displayed in real time on the screen.

The system is designed to be lightweight, fast, and runnable on standard laptops.

### Features
Real-time ASL fingerspelling recognition using webcam input
MediaPipe-based hand landmark detection (21 key points per hand)
Machine learning model for alphabet classification
Live prediction overlay in real time
Fully offline processing (no cloud dependency)
Lightweight design suitable for low-resource devices
How It Works
Webcam captures live hand movement
MediaPipe extracts 21 hand landmarks per frame
Landmark coordinates are converted into feature vectors
A trained ML model predicts the corresponding ASL letter
Prediction is displayed instantly on screen


### Tech Stack
Python
OpenCV
MediaPipe
Scikit-learn
NumPy
Pandas
Project Structure
sign-ai/
├── collect.py      # Dataset collection using webcam
├── train.py        # Model training pipeline
├── predict.py      # Real-time inference
├── utils.py        # Helper functions
├── data/           # Collected landmark dataset
├── model.pkl       # Trained machine learning model
└── README.md

### Dataset

The dataset was created using MediaPipe hand landmark extraction.

Each sample consists of 21 hand landmarks (x, y, z coordinates) representing static ASL fingerspelling gestures.

### Current Scope

This project currently supports static ASL alphabet recognition.

Dynamic gestures such as J and Z are not included, as they require temporal (sequence-based) motion modeling rather than single-frame classification.

### Future Improvements
Dynamic gesture recognition (J and Z) using sequence models
Word and sentence formation from letter predictions
Sign language to text/speech conversion
Web deployment using Streamlit or Flask
Larger and more diverse dataset collection


### How to Run

pip install -r requirements.txt
python collect.py
python train.py
python predict.py
