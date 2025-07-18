# Smart CCTV – AI-Powered Surveillance System (Raspberry Pi 5)

Smart CCTV is an AI-powered real-time security camera system built for Raspberry Pi 5. It performs person/face detection using OpenCV's deep learning model and serves the video feed live via a Flask web interface. Designed for home security, surveillance, or educational STEM projects.

## Features
- Real-time person/face detection using OpenCV DNN (SSD-based model)
- Live video streaming via Flask app
- Confidence-based detection with bounding boxes and labels
- Lightweight, real-time performance on Raspberry Pi 5

## Installation
```bash
pip install opencv-python flask numpy
```

For Raspberry Pi dependencies:
```bash
sudo apt install libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5
```

## Run
```bash
python3 smart_cctv.py
```

Then open browser: `http://<your-raspberry-pi-ip>:5001/`

## Model Files
The app automatically downloads:
- `deploy.prototxt` – Model architecture
- `res10_300x300_ssd_iter_140000.caffemodel` – Pre-trained face detection model

## Screenshots
Live demo on browser with real-time detection.
