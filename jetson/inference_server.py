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
        
        # Frame dimensions - Make sure these are supported by your camera's modes!
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
        
        # --- CSI Camera Check (Jetson) ---
        logger.info("Attempting CSI camera detection...")
        if self.test_csi_camera():
            logger.info("CSI camera detected and tested successfully.")
            return "csi"
        else:
            logger.info("CSI camera test failed or not available.")
        
        # --- USB Camera Check with GStreamer ---
        logger.info("Attempting USB camera (v4l2src) detection...")
        try:
            # First, check if v4l2src plugin exists and /dev/video0 is present
            result = subprocess.run("gst-inspect-1.0 v4l2src", shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and os.path.exists("/dev/video0"):
                logger.info("v4l2src plugin found and /dev/video0 exists. Testing USB camera...")
                if self.test_usb_camera():
                    logger.info("USB camera detected and tested successfully.")
                    return "usb"
                else:
                    logger.info("USB camera test failed.")
            else:
                logger.info(f"v4l2src inspect return: {result.returncode}, stderr: {result.stderr.strip()}")
                logger.info("USB camera requirements (v4l2src or /dev/video0) not met.")
        except subprocess.TimeoutExpired:
            logger.warning("gst-inspect-1.0 v4l2src timed out.")
        except Exception as e:
            logger.error(f"Error during USB camera detection check: {e}")
        
        # --- Fall back to OpenCV ---
        logger.info("Attempting OpenCV camera detection...")
        opencv_id = self.find_opencv_camera()
        if opencv_id is not None:
            logger.info(f"OpenCV camera found at ID: {opencv_id}.")
            return "opencv"
        else:
            logger.info("No OpenCV camera found.")
            
        logger.warning("No camera method detected!")
        return "none"
        
    def _construct_csi_gst_command(self, num_buffers=1, location=None):
        """Helper to construct the CSI GStreamer command as a list for subprocess.run"""
        cmd = [
            "gst-launch-1.0",
            "-e",
            "nvarguscamerasrc",
            f"num-buffers={num_buffers}",
            "!",
            f"video/x-raw(memory:NVMM),width={self.frame_width},height={self.frame_height},framerate=30/1",
            "!",
            "nvvidconv",
            "!",
            "jpegenc",
            "!"
        ]
        if location:
            cmd.extend(["filesink", f"location={location}"])
        else:
            # If no location, use a null sink for testing pipeline validity
            cmd.append("fakesink") 
            # For live preview during testing:
            # cmd.extend(["nvegltransform", "!", "nveglglessink"])
        return cmd

    def test_csi_camera(self):
        """Test if CSI camera works by capturing a single frame"""
        test_path = os.path.join(self.temp_dir, "test_csi.jpg")
        
        # Clean up previous test file
        if os.path.exists(test_path):
            os.remove(test_path)
            
        # Use the helper to construct a robust command list
        gst_cmd_list = self._construct_csi_gst_command(num_buffers=1, location=test_path)
        
        logger.info(f"CSI camera test command: {' '.join(gst_cmd_list)}")
        
        try:
            # Run the command
            result = subprocess.run(
                gst_cmd_list,
                capture_output=True,
                text=True,
                timeout=15 # Increased timeout slightly for first camera capture
            )
            
            logger.debug(f"CSI test stdout: {result.stdout.strip()}")
            logger.debug(f"CSI test stderr: {result.stderr.strip()}")
            
            # Check return code and whether the file was created and is not empty
            success = (
                result.returncode == 0 and
                "ERROR:" not in result.stderr and # Look for GStreamer errors in stderr
                "failed" not in result.stderr and # General failure keyword
                os.path.exists(test_path) and
                os.path.getsize(test_path) > 0
            )
            
            if not success:
                logger.warning(f"CSI camera test failed. Return code: {result.returncode}")
                if result.stderr:
                    logger.warning(f"CSI test stderr output: {result.stderr.strip()}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("CSI camera test timed out.")
            return False
        except Exception as e:
            logger.error(f"Error during CSI camera test: {e}")
            return False
            
    def test_usb_camera(self):
        """Test if USB camera works"""
        test_path = os.path.join(self.temp_dir, "test_usb.jpg")
        
        if os.path.exists(test_path):
            os.remove(test_path)
            
        gst_cmd_list = [
            "gst-launch-1.0",
            "-e",
            "v4l2src",
            "device=/dev/video0",
            "num-buffers=1",
            "!",
            "videoconvert",
            "!",
            "videoscale",
            "!",
            f"video/x-raw,width={self.frame_width},height={self.frame_height}",
            "!",
            "jpegenc",
            "!",
            "filesink",
            f"location={test_path}"
        ]
        
        logger.info(f"USB camera test command: {' '.join(gst_cmd_list)}")
        
        try:
            result = subprocess.run(
                gst_cmd_list,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            logger.debug(f"USB test stdout: {result.stdout.strip()}")
            logger.debug(f"USB test stderr: {result.stderr.strip()}")
            
            success = (
                result.returncode == 0 and
                "ERROR:" not in result.stderr and
                "failed" not in result.stderr and
                os.path.exists(test_path) and
                os.path.getsize(test_path) > 0
            )
            
            if not success:
                logger.warning(f"USB camera test failed. Return code: {result.returncode}")
                if result.stderr:
                    logger.warning(f"USB test stderr output: {result.stderr.strip()}")
                    
            return success
        except subprocess.TimeoutExpired:
            logger.error("USB camera test timed out.")
            return False
        except Exception as e:
            logger.error(f"Error during USB camera test: {e}")
            return False
            
    def find_opencv_camera(self):
        """Find working OpenCV camera"""
        for camera_id in [0, 1, 2]: # Iterate common camera IDs
            logger.debug(f"Testing OpenCV camera ID: {camera_id}")
            try:
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    logger.debug(f"OpenCV camera ID {camera_id} not opened.")
                    cap.release()
                    continue
                
                # Try setting properties (might fail if not supported, but doesn't hurt)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                # For USB cams, MJPG is often preferred for higher resolutions/framerate
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')) 
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Attempt to read a frame
                ret, frame = cap.read()
                cap.release() # Release immediately
                
                if ret and frame is not None and frame.size > 0:
                    logger.info(f"OpenCV camera found and working at ID: {camera_id} (captured frame shape: {frame.shape})")
                    return camera_id
                else:
                    logger.debug(f"OpenCV camera ID {camera_id} opened but failed to read frame or frame is empty.")
            except Exception as e:
                logger.debug(f"Error testing OpenCV camera ID {camera_id}: {e}")
        logger.info("No functional OpenCV camera found.")
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
        # Removed shell=True, using list of arguments now
        gst_cmd_list = self._construct_csi_gst_command(num_buffers=1, location=self.temp_frame_path)
        
        try:
            if os.path.exists(self.temp_frame_path):
                os.remove(self.temp_frame_path)
            
            # Using subprocess.run for simplicity and better error handling
            # It waits for the command to complete.
            result = subprocess.run(gst_cmd_list, capture_output=True, text=True, timeout=10)
            
            logger.debug(f"Capture CSI stdout: {result.stdout.strip()}")
            logger.debug(f"Capture CSI stderr: {result.stderr.strip()}")
            
            if result.returncode != 0:
                logger.warning(f"GStreamer CSI capture command failed with code {result.returncode}. Stderr: {result.stderr.strip()}")
                return None

            if os.path.exists(self.temp_frame_path) and os.path.getsize(self.temp_frame_path) > 0:
                frame = cv2.imread(self.temp_frame_path)
                if frame is not None and frame.size > 0:
                    self.last_method = "csi"
                    return frame
                else:
                    logger.warning(f"CSI camera capture produced empty or invalid image file at {self.temp_frame_path}.")
                    return None
            else:
                logger.warning(f"CSI camera capture failed: No file or empty file at {self.temp_frame_path}.")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("GStreamer CSI process timed out during capture.")
            return None
        except Exception as e:
            logger.error(f"Error capturing CSI frame: {e}")
            return None
            
    def capture_frame_gstreamer_usb(self):
        """Capture frame using USB camera with GStreamer"""
        gst_cmd_list = [
            "gst-launch-1.0",
            "-e",
            "v4l2src",
            "device=/dev/video0",
            "num-buffers=1",
            "!",
            "videoconvert",
            "!",
            "videoscale",
            "!",
            f"video/x-raw,width={self.frame_width},height={self.frame_height}",
            "!",
            "jpegenc",
            "!",
            "filesink",
            f"location={self.temp_frame_path}"
        ]
        
        try:
            if os.path.exists(self.temp_frame_path):
                os.remove(self.temp_frame_path)
            
            result = subprocess.run(gst_cmd_list, capture_output=True, text=True, timeout=10)
            
            logger.debug(f"Capture USB stdout: {result.stdout.strip()}")
            logger.debug(f"Capture USB stderr: {result.stderr.strip()}")
            
            if result.returncode != 0:
                logger.warning(f"GStreamer USB capture command failed with code {result.returncode}. Stderr: {result.stderr.strip()}")
                return None

            if os.path.exists(self.temp_frame_path) and os.path.getsize(self.temp_frame_path) > 0:
                frame = cv2.imread(self.temp_frame_path)
                if frame is not None and frame.size > 0:
                    self.last_method = "usb"
                    return frame
                else:
                    logger.warning(f"USB camera capture produced empty or invalid image file at {self.temp_frame_path}.")
                    return None
            else:
                logger.warning(f"USB camera capture failed: No file or empty file at {self.temp_frame_path}.")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("GStreamer USB process timed out during capture.")
            return None
        except Exception as e:
            logger.error(f"Error capturing USB frame: {e}")
            return None
            
    def capture_frame_opencv(self):
        """Capture frame using OpenCV"""
        if self.opencv_camera_id is None:
            logger.error("No OpenCV camera available")
            return None
        
        cap = None # Initialize cap to None
        try:
            cap = cv2.VideoCapture(self.opencv_camera_id)
            if not cap.isOpened():
                logger.error("Failed to open OpenCV camera")
                return None
        
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # KEY FIX
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
        
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"OpenCV camera actual settings: {actual_width}x{actual_height}")
        
            # Skip a few frames to let camera stabilize
            for _ in range(3):
                ret, frame = cap.read()
                if not ret:
                    logger.warning("OpenCV: Failed to read frame during warmup.")
                    break # Exit loop if read fails
                time.sleep(0.05) # Shorter sleep
        
            ret, frame = cap.read()
            
            if ret and frame is not None and frame.size > 0:
                self.last_method = "opencv"
                logger.debug(f"Captured frame: {frame.shape}, dtype: {frame.dtype}")
                return frame
            else:
                logger.error("Failed to capture OpenCV frame or frame is empty.")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing OpenCV frame: {e}")
            return None
        finally:
            if cap is not None and cap.isOpened():
                cap.release() # Ensure camera is released
            

    # ... (rest of your class, Flask routes, and main execution) ...
    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade"""
        faces = []
        
        try:
            if not hasattr(self, 'detector'):
                # Ensure the path to haarcascades is correct in a Docker container
                # It should be available if opencv-python-headless is installed.
                # You might need to add `opencv-data` package in your Dockerfile.
                haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not os.path.exists(haar_cascade_path):
                    logger.error(f"Haar cascade file not found: {haar_cascade_path}")
                    # You might need to download it or ensure opencv-data is installed
                    # Example: `RUN apt-get update && apt-get install -y opencv-data` in Dockerfile
                    return []
                self.detector = cv2.CascadeClassifier(haar_cascade_path)
            
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