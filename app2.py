from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import json
import os
import sqlite3
from ultralytics import YOLO
from dotenv import load_dotenv
import logging
import threading
import time
import queue  # Import the queue module

# Load environment variables
load_dotenv()
USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")

app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Load cameras from JSON
try:
    with open("cameras.json", "r") as f:
        cameras = json.load(f)
    logging.info("Cameras loaded from cameras.json")
except FileNotFoundError:
    logging.error("cameras.json not found.  Exiting.")
    exit()  # or handle the error more gracefully (e.g., provide a default camera)
except json.JSONDecodeError:
    logging.error("Invalid JSON in cameras.json.  Exiting.")
    exit()

def record_movement(camera_name, detection):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO movements (camera_name, detection) VALUES (?, ?)", (camera_name, detection))
        conn.commit()
        logging.info(f"Movement recorded: camera={camera_name}, detection={detection}")
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        conn.close()

def get_camera_feed(camera_ip):
    """Connects to the camera using IP and credentials."""
    global USERNAME, PASSWORD # Use global USERNAME and PASSWORD

    # Debugging: Print the variables
    logging.info(f"Camera IP: {camera_ip}")
    logging.info(f"Username: {USERNAME}")
    logging.info(f"Password: {PASSWORD}")

    #Strip white space
    camera_ip = camera_ip.strip()
    USERNAME = USERNAME.strip()
    PASSWORD = PASSWORD.strip()


    if not USERNAME or not PASSWORD:
        logging.error(f"Credentials missing for camera {camera_ip}.")
        return None  # or raise an exception if credentials are required

    url = f"rtsp://{USERNAME}:{PASSWORD}@{camera_ip}/live"  # EXACT URL /live  #####USE /LIVE
    logging.info(f"RTSP URL: {url}")


    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logging.warning(f"Failed to connect to {camera_ip}, trying local camera...")
        cap = cv2.VideoCapture(0)  # Fallback to default webcam
        if not cap.isOpened():
            logging.error("Failed to open default webcam.")
            return None # Return None if both fail

    return cap

# Load models
try:
    model_det = YOLO("best.pt")  # Detection model
    model_seg = YOLO("seg_best.pt")  # Segmentation model
    logging.info("YOLO models loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}. Exiting.")
    exit()
except Exception as e:
    logging.error(f"Error loading YOLO models: {e}. Exiting.")
    exit()

SEGMENTATION_CLASSES = {
    0: "Panel",
    1: "Restricted Passage",
    2: "Safe Passage"
}

# Global Queue to pass detection to main Thread.
detection_queue = queue.Queue()

def monitor_camera(camera_ip, camera_name):
    """Monitors a single camera for detections in a background thread."""
    cap = get_camera_feed(camera_ip)
    if cap is None:
        logging.error(f"Failed to get camera feed for {camera_name}.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            logging.warning(f"Failed to read frame from {camera_name}. Reconnecting...")
            cap.release()
            cap = get_camera_feed(camera_ip)
            if cap is None:
                logging.error(f"Failed to reconnect to {camera_name}. Monitoring stopped.")
                break
            continue

        try:
            results = model_det(frame, conf=0.25)  # Perform detection

            for result in results:
                for box in result.boxes:
                    label = model_det.names[int(box.cls[0])]
                    detection_queue.put((camera_name, label)) #Pass detection to main thread.

        except Exception as e:
            logging.error(f"Error processing frame from {camera_name}: {e}")

        # Small delay to avoid excessive CPU usage
        time.sleep(0.1)

    cap.release()
    logging.info(f"Monitoring stopped for {camera_name}.")

def start_camera_monitoring():
    """Starts monitoring threads for all cameras."""
    for ip, name in cameras.items():
        thread = threading.Thread(target=monitor_camera, args=(ip.strip(), name.strip()))
        thread.daemon = True  # Allow the main thread to exit even if this thread is running
        thread.start()
        logging.info(f"Started monitoring {name} in background.")

def initialize(): #Remove Decorator
    start_camera_monitoring()

    def process_detections():
         while True:
            try:
                camera_name, label = detection_queue.get(timeout = 5) #Timeout incase Queue never recieves data.
                record_movement(camera_name, f"{label} detected")
            except queue.Empty:
                pass #Queue is empty.
    thread = threading.Thread(target = process_detections)
    thread.daemon = True
    thread.start()



def generate_frames(camera_ip, camera_name, model_type="detection"):
    cap = get_camera_feed(camera_ip)
    if cap is None:
        logging.error("Failed to get camera feed.")
        return # Exit generator if camera feed failed

    model = model_det if model_type == "detection" else model_seg
    while True:  # Changed to a loop that checks success before processing
        success, frame = cap.read()
        if not success:
            logging.warning(f"Failed to read frame. Reconnecting to camera.")
            cap.release() #Release current capture
            cap = get_camera_feed(camera_ip) #Reconnect
            if cap is None:
                logging.error("Failed to reconnect to camera. Exiting frame generation.")
                break #Exit generator if reconnection fails

            success, frame = cap.read() #Read the frame again after reconnecting
            if not success:
                logging.error("Failed to read frame after reconnecting. Exiting frame generation.")
                break #Exit if second read fails.
            #break  # Exit loop if no more frames
            continue # Continue if reconnection was successful and frame was read

        try:
            results = model(frame, conf=0.25)  # Lower confidence threshold
            for result in results:
                if model_type == "segmentation" and result.masks is not None:
                    for mask, cls in zip(result.masks.xy, result.boxes.cls):
                        label = SEGMENTATION_CLASSES.get(int(cls), "Unknown")
                        color = (0, 255, 0) if label == "Safe Passage" else (0, 0, 255) if label == "Restricted Passage" else (255, 0, 0)
                        for poly in mask:
                            pts = poly.reshape((-1, 1, 2)).astype(int)
                            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                            cv2.putText(frame, label, (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = result.names[int(box.cls[0])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            break # Exit the generator on error

    #cap.release() # Moved release inside the camera feed function
    logging.info("Frame generation stopped.") #Log when generation stops

@app.route('/')
def index():
    return render_template("index.html", cameras=cameras)

@app.route('/video_feed')
def video_feed():
    camera_name = request.args.get("camera")  # Get camera name from request

    if not camera_name:
        return "❌ Camera name is required!", 400

    # Find the camera IP based on the name
    camera_ip = None
    for ip, name in cameras.items():
        if name == camera_name:
            camera_ip = ip
            break

        #Add Strip too
    camera_ip = camera_ip.strip()

    if not camera_ip:
        return "❌ Camera not found!", 404

    model_type = request.args.get("model", "detection")
    return Response(generate_frames(camera_ip, camera_name, model_type), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)

        # Basic filename sanitization (important for security)
        filename = os.path.basename(filename) #Removes directory components
        if not filename: #Check if the name is empty
             return jsonify({"error": "Invalid filename"}), 400 #Return an error if it is empty

        # Delete old files (handle exceptions)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError as e:
            logging.error(f"Error deleting file: {e}")
            return jsonify({"error": "Error deleting existing files."}), 500

        file.save(file_path)  # Save the uploaded file

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
                for mask, cls in zip(result.masks.xy, result.boxes.cls):
                    label = SEGMENTATION_CLASSES.get(int(cls), "Unknown")
                    color = (0, 255, 0) if label == "Safe Passage" else (0, 0, 255) if label == "Restricted Passage" else (255, 0, 0)
                    for poly in mask:
                        pts = poly.reshape((-1, 1, 2)).astype(int)
                        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                        cv2.putText(img, label, (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(output_path, img)

        return jsonify({
            "input": f"/uploads/{filename}",
            "output": f"/outputs/{filename}",
            "detection_model_results": str(results_det),
            "segmentation_model_results": str(results_seg)
        })
    except Exception as e:
        logging.exception("Error processing image upload:") #Log full traceback
        return jsonify({"error": str(e)}), 500 #Return the error to the client

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/outputs/<path:filename>')
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == '__main__':
    # Set default username and password if not in .env
    os.environ.setdefault("CAMERA_USERNAME", "admin")
    os.environ.setdefault("CAMERA_PASSWORD", "admin1234")

    initialize() # Call the initialize function directly.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader = False)