from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import json
import os
import sqlite3
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

db_path = "detections.db"

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS movements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        camera_name TEXT,
                        detection TEXT)''')
    conn.commit()
    conn.close()

init_db()

with open("cameras.json", "r") as f:
    cameras = json.load(f)

def record_movement(camera_name, detection):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO movements (camera_name, detection) VALUES (?, ?)", (camera_name, detection))
    conn.commit()
    conn.close()

def get_camera_feed(camera_url):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Default to system camera
    return cap

model = YOLO("best.pt")

def generate_frames(camera_url, camera_name):
    cap = get_camera_feed(camera_url)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                record_movement(camera_name, "Object detected")
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template("index.html", cameras=cameras)

@app.route('/video_feed')
def video_feed():
    camera_name = request.args.get("camera", "Default Camera")
    camera_url = cameras.get(camera_name, 0)
    return Response(generate_frames(camera_url, camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)

    # Remove previous files
    for folder in [app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"]]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    
    file.save(file_path)
    results = model(file_path)
    results[0].save(filename=output_path)  # Ensure the file is saved with the correct filename

    return jsonify({
        "input": f"/uploads/{filename}",
        "output": f"/outputs/{filename}"
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == '__main__':
    app.run(debug=True)