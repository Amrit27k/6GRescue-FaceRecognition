#!/usr/bin/env python3
"""Flask-based Face Recognition Inference Service for Jetson (System packages only)"""
import cv2
import numpy as np
import time
import os
import base64
import json
import logging
import subprocess
import threading
import queue
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, Response
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class JetsonFaceRecognition:
    def __init__(self, model_service_url=None, temp_dir="temp_frames"):
        # Model service URL
        self.model_service_url = model_service_url or os.environ.get('MODEL_SERVICE_URL', 'http://localhost:5000')
        logger.info(f"Using model service: {self.model_service_url}")
        
        # Temporary directory for frame capture
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Temp frame path
        self.temp_frame_path = os.path.join(self.temp_dir, "current_frame.jpg")
        
        # Output directory for saved frames
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Performance metrics
        self.fps = 0
        self.avg_inference_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.inference_times = []
        
        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480
        
        # Most recent processed frame and results for streaming
        self.last_frame = None
        self.last_results = []
        self.last_method = "none"
        
        # For camera streaming
        self.camera_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.camera_thread = None
        
        # Detect best camera method
        self.camera_method = self.detect_camera_method()
        self.opencv_camera_id = None
        if self.camera_method == "opencv":
            self.opencv_camera_id = self.find_opencv_camera()
        
        logger.info(f"Using camera method: {self.camera_method}")
        
    def detect_camera_method(self):
        """Detect the best available camera method"""
        # Check for CSI camera (Jetson)
        try:
            result = subprocess.run("gst-inspect-1.0 nvarguscamerasrc", 
                                  shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                if self.test_csi_camera():
                    return "csi"
        except:
            pass
        
        # Check for USB camera with GStreamer
        try:
            result = subprocess.run("gst-inspect-1.0 v4l2src", 
                                  shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and os.path.exists("/dev/video0"):
                if self.test_usb_camera():
                    return "usb"
        except:
            pass
        
        # Fall back to OpenCV
        if self.find_opencv_camera() is not None:
            return "opencv"
        
        logger.warning("No camera method detected!")
        return "none"
    
    def test_csi_camera(self):
        """Test if CSI camera works"""
        test_path = os.path.join(self.temp_dir, "test_csi.jpg")
        gst_cmd = (
            f"gst-launch-1.0 -e nvarguscamerasrc num-buffers=1 ! "
            f"'video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1' ! "
            f"nvvidconv ! jpegenc ! filesink location={test_path}"
        )
        
        try:
            result = subprocess.run(gst_cmd, shell=True, capture_output=True, text=True, timeout=10)
            success = os.path.exists(test_path) and os.path.getsize(test_path) > 0
            if os.path.exists(test_path):
                os.remove(test_path)
            return success
        except:
            return False
    
    def test_usb_camera(self):
        """Test if USB camera works"""
        test_path = os.path.join(self.temp_dir, "test_usb.jpg")
        gst_cmd = (
            f"gst-launch-1.0 -e v4l2src device=/dev/video0 num-buffers=1 ! "
            f"videoconvert ! jpegenc ! filesink location={test_path}"
        )
        
        try:
            result = subprocess.run(gst_cmd, shell=True, capture_output=True, text=True, timeout=10)
            success = os.path.exists(test_path) and os.path.getsize(test_path) > 0
            if os.path.exists(test_path):
                os.remove(test_path)
            return success
        except:
            return False
    
    def find_opencv_camera(self):
        """Find working OpenCV camera"""
        for camera_id in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        return camera_id
            except:
                pass
        return None
        
    def capture_frame(self):
        """Capture frame using the best available method"""
        if self.camera_method == "csi":
            return self.capture_frame_gstreamer_csi()
        elif self.camera_method == "usb":
            return self.capture_frame_gstreamer_usb()
        elif self.camera_method == "opencv":
            return self.capture_frame_opencv()
        else:
            logger.error("No camera method available")
            return None
            
    def capture_frame_gstreamer_csi(self):
        """Capture frame using CSI camera with GStreamer"""
        gst_cmd = (
            f"gst-launch-1.0 -e nvarguscamerasrc num-buffers=1 ! "
            f"'video/x-raw(memory:NVMM),width={self.frame_width},height={self.frame_height},framerate=30/1' ! "
            f"nvvidconv ! jpegenc ! filesink location={self.temp_frame_path}"
        )
        
        try:
            if os.path.exists(self.temp_frame_path):
                os.remove(self.temp_frame_path)
            
            process = subprocess.Popen(gst_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            timeout = 5
            start_time = time.time()
            
            while process.poll() is None and time.time() - start_time < timeout:
                time.sleep(0.01)
            
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                logger.warning("GStreamer CSI process timed out")
                return None
            
            if os.path.exists(self.temp_frame_path) and os.path.getsize(self.temp_frame_path) > 0:
                frame = cv2.imread(self.temp_frame_path)
                if frame is not None:
                    self.last_method = "csi"
                    return frame
            
            logger.warning("CSI camera capture failed")
            return None
                
        except Exception as e:
            logger.error(f"Error capturing CSI frame: {e}")
            return None
    
    def capture_frame_gstreamer_usb(self):
        """Capture frame using USB camera with GStreamer"""
        gst_cmd = (
            f"gst-launch-1.0 -e v4l2src device=/dev/video0 num-buffers=1 ! "
            f"videoconvert ! videoscale ! video/x-raw,width={self.frame_width},height={self.frame_height} ! "
            f"jpegenc ! filesink location={self.temp_frame_path}"
        )
        
        try:
            if os.path.exists(self.temp_frame_path):
                os.remove(self.temp_frame_path)
            
            process = subprocess.Popen(gst_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            timeout = 5
            start_time = time.time()
            
            while process.poll() is None and time.time() - start_time < timeout:
                time.sleep(0.01)
            
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                logger.warning("GStreamer USB process timed out")
                return None
            
            if os.path.exists(self.temp_frame_path) and os.path.getsize(self.temp_frame_path) > 0:
                frame = cv2.imread(self.temp_frame_path)
                if frame is not None:
                    self.last_method = "usb"
                    return frame
            
            logger.warning("USB camera capture failed")
            return None
                
        except Exception as e:
            logger.error(f"Error capturing USB frame: {e}")
            return None
    
    def capture_frame_opencv(self):
        """Capture frame using OpenCV"""
        if self.opencv_camera_id is None:
            logger.error("No OpenCV camera available")
            return None
        
        try:
            cap = cv2.VideoCapture(self.opencv_camera_id)
            if not cap.isOpened():
                logger.error("Failed to open OpenCV camera")
                return None
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            for _ in range(3):
                ret, frame = cap.read()
                time.sleep(0.1)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                self.last_method = "opencv"
                return frame
            else:
                logger.error("Failed to capture OpenCV frame")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing OpenCV frame: {e}")
            return None

    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade"""
        faces = []
        
        try:
            if not hasattr(self, 'detector'):
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.detector.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            for (x, y, w, h) in detected_faces:
                faces.append({
                    "box": [x, y, w, h],
                    "name": "Unknown",
                    "confidence": 0
                })
                
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            
        return faces
    
    def process_frame(self, frame):
        """Process a frame - detect and recognize faces"""
        start_time = time.time()
        
        faces = self.detect_faces(frame)
        processed_frame = frame.copy()
        
        for face in faces:
            x, y, w, h = face["box"]
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                result = self.recognize_face(face_roi)
                face["name"] = result.get("name", "Unknown")
                face["confidence"] = result.get("confidence", 0)
                face["person_id"] = result.get("person_id", None)
            except Exception as e:
                logger.error(f"Error recognizing face: {e}")
                face["name"] = "Error"
                face["confidence"] = 0
                
            color = (0, 255, 0) if face["name"] != "Unknown" else (0, 0, 255)
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{face['name']}"
            if face["confidence"] > 0:
                label += f" ({face['confidence']:.0f}%)"
            
            cv2.putText(processed_frame, label, (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.avg_inference_time = np.mean(self.inference_times[-30:]) * 1000
            self.frame_count = 0
            self.start_time = time.time()
        
        info_text = f"FPS: {self.fps:.1f} | Inference: {self.avg_inference_time:.1f}ms | Faces: {len(faces)} | {self.last_method.upper()}"
        cv2.putText(processed_frame, info_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return faces, processed_frame
        
    def recognize_face(self, face_roi):
        """Recognize face using model service"""
        try:
            _, img_encoded = cv2.imencode('.jpg', face_roi)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            data = {
                "instances": [
                    {"face_image": img_base64}
                ]
            }
            
            response = requests.post(
                f"{self.model_service_url}/invocations",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])
                if predictions:
                    return predictions[0]
                    
            return {"name": "Unknown", "confidence": 0, "person_id": None}
            
        except Exception as e:
            logger.error(f"Error calling model service: {e}")
            return {"name": "Error", "confidence": 0, "person_id": None}

# Initialize face recognition system
face_recognition = JetsonFaceRecognition()

# Flask routes
@app.route("/")
def root():
    """API root - provides basic info"""
    return jsonify({
        "name": "Jetson Face Recognition API (Flask)",
        "version": "1.0",
        "camera_method": face_recognition.camera_method,
        "endpoints": {
            "GET /": "This info",
            "POST /capture": "Capture and process a single frame",
            "GET /ping": "Health check"
        }
    })

@app.route("/capture", methods=['POST'])
def capture():
    """Capture and process a single frame"""
    try:
        frame = face_recognition.capture_frame()
        
        if frame is None:
            return jsonify({
                "success": False,
                "error": f"Failed to capture frame using {face_recognition.camera_method} method"
            }), 500
            
        faces, processed_frame = face_recognition.process_frame(frame)
        
        output_path = os.path.join(face_recognition.output_dir, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(output_path, processed_frame)
        
        return jsonify({
            "success": True,
            "faces": faces,
            "frame_path": output_path,
            "timestamp": datetime.now().isoformat(),
            "method_used": face_recognition.last_method
        })
        
    except Exception as e:
        logger.error(f"Error processing capture: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/ping")
def ping():
    """Health check"""
    try:
        try:
            response = requests.get(f"{face_recognition.model_service_url}/ping", timeout=2)
            model_status = "OK" if response.status_code == 200 else "ERROR"
        except:
            model_status = "UNREACHABLE"
            
        return jsonify({
            "status": "OK",
            "model_service": model_status,
            "timestamp": datetime.now().isoformat(),
            "camera_method": face_recognition.camera_method
        })
    except Exception as e:
        logger.error(f"Error in ping: {e}")
        return jsonify({
            "status": "ERROR",
            "model_service": "ERROR", 
            "timestamp": datetime.now().isoformat(),
            "camera_method": "ERROR"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)