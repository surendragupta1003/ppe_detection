from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import json
import os
import sqlite3
import shutil
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

# Load both models
model_det = YOLO("best.pt")  # Detection model
model_seg = YOLO("seg_best.pt")  # Segmentation model

def generate_frames(camera_url, camera_name, model_type="detection"):
    cap = get_camera_feed(camera_url)
    model = model_det if model_type == "detection" else model_seg
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model(frame, conf=0.25)  # Lower confidence threshold
        for result in results:
            if model_type == "segmentation":
                if result.masks is not None:
                    for mask in result.masks.xy:
                        for poly in mask:  # Loop through mask polygons
                            pts = poly.reshape((-1, 1, 2)).astype(int)
                            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            else:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    record_movement(camera_name, f"{label} detected")
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
    model_type = request.args.get("model", "detection")  # Choose between "detection" or "segmentation"
    return Response(generate_frames(camera_url, camera_name, model_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    
    # Delete only old files if they exist
    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    file.save(file_path)
    
    # Run both models and overlay results on the same image
    img = cv2.imread(file_path)
    
    results_det = model_det(file_path, conf=0.25)
    results_seg = model_seg(file_path, conf=0.25)
    
    # Apply detection results
    for result in results_det:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Apply segmentation results
    for result in results_seg:
        if result.masks is not None:
            for mask in result.masks.xy:
                for poly in mask:
                    pts = poly.reshape((-1, 1, 2)).astype(int)
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    
    cv2.imwrite(output_path, img)
    
    return jsonify({
        "input": f"/uploads/{filename}",
        "output": f"/outputs/{filename}",
        "detection_model_results": str(results_det),
        "segmentation_model_results": str(results_seg)
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.exists(output_path):
        return jsonify({"error": "Output file not found"}), 404
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == '__main__':
    app.run(debug=True)