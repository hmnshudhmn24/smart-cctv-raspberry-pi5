# smart_cctv.py - AI-Powered CCTV with Flask + OpenCV
import cv2
import time
import threading
import numpy as np
from flask import Flask, render_template_string, Response
import urllib.request

app = Flask(__name__)

prototxt_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
model_url = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel'
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'

urllib.request.urlretrieve(prototxt_url, prototxt_path)
urllib.request.urlretrieve(model_url, model_path)

HTML_TEMPLATE = """
<html>
<head><title>Smart CCTV</title></head>
<body><h1>Smart CCTV - Live Stream</h1>
<img src="{{ url_for('video_feed') }}"></body>
</html>
"""

CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

lock = threading.Lock()
output_frame = None

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
cap = cv2.VideoCapture(0)

def detect_and_draw(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"Face: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def capture_frames():
    global output_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = detect_and_draw(frame)
        with lock:
            output_frame = frame.copy()

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_app():
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5001, debug=False)

start_app()
