from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time
import threading

app = Flask(__name__)

# Load model
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "C:/Users/haris/OneDrive/Desktop/Hack/isl_gesture_recognition_mobilenet_model.h5",
    "C:/Users/haris/OneDrive/Desktop/model/labels.txt"
)

offset = 20
imgSize = 300
labels = ["Hello", "Thank you", "No", "Yes", "Love You", "Done", "Think"]

cap = cv2.VideoCapture(0)

# State tracking
last_spoken_time = 0
speaking_lock = threading.Lock()

# Safe background TTS
def speak_in_background(text):
    def run():
        with speaking_lock:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def gen_frames():
    global last_spoken_time
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                current_label = labels[index]

                # Speak every 2 seconds regardless of label change
                if time.time() - last_spoken_time > 2:
                    speak_in_background(current_label)
                    last_spoken_time = time.time()

                # Draw
                cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, current_label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            except Exception as e:
                print("Error in processing:", e)

        ret, buffer = cv2.imencode('.jpg', imgOutput)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
