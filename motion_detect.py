import cv2
import numpy as np
import sqlite3
import datetime

def log_movement(camera_name):
    """Logs motion detection event into SQLite database"""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO motion_log (camera_name, timestamp) VALUES (?, ?)", (camera_name, timestamp))
    conn.commit()
    conn.close()

def detect_motion(frame, prev_frame, camera_name):
    """Detects motion and logs event if movement is detected"""
    if prev_frame is None:
        return frame  # First frame

    # Convert to grayscale & blur for better detection
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels (movement threshold)
    if np.count_nonzero(thresh) > 1000:
        log_movement(camera_name)
    
    return frame  # Return current frame for next comparison
