# sign-ai
Real-time ASL fingerspelling recognition using MediaPipe hand tracking and machine learning.
# SignAI

SignAI is a real-time American Sign Language (ASL) fingerspelling recognition system built using Python, OpenCV, MediaPipe, and machine learning.

The application uses a standard webcam to detect hand landmarks, classify static ASL fingerspelling gestures, and display predicted letters in real time.

#Problem

Communication barriers can arise when sign language users interact with people who do not understand sign language. Many existing solutions require specialized hardware, large datasets, or cloud-based processing.

SignAI explores a lightweight, accessible approach using only a webcam and computer vision.

Features

* Real-time webcam-based recognition
* MediaPipe hand tracking
* 21-hand-landmark extraction
* Machine learning classification
* Live ASL fingerspelling prediction
* No specialized hardware required

How It Works

1. A webcam captures the user's hand.
2. MediaPipe extracts 21 hand landmarks.
3. Landmark coordinates are converted into numerical feature vectors.
4. A machine learning model predicts the corresponding ASL fingerspelling letter.
5. The prediction is displayed on screen in real time.

Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* Pandas
* NumPy

Project Structure

```text
sign-ai/
│
├── collect.py      # Dataset collection
├── train.py        # Model training
├── predict.py      # Real-time prediction
├── utils.py        # Utility functions
├── data/           # Collected landmark datasets
├── model.pkl       # Trained model
└── README.md
```

Dataset

The training dataset was created by collecting MediaPipe hand landmark coordinates for static ASL fingerspelling gestures.

Each sample contains the x, y, and z coordinates of 21 detected hand landmarks.

Current Scope

This prototype focuses on static ASL fingerspelling gestures.

Dynamic gestures such as J and Z are not currently supported because they require motion-based recognition rather than single-frame classification.

Future Improvements

* Dynamic gesture recognition (J and Z)
* Word and phrase formation
* Sign-language-to-text conversion
* Sign-language-to-speech output
* Web deployment using Streamlit
* Larger and more diverse training datasets

Demo

