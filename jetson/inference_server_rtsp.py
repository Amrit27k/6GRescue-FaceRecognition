#!/usr/bin/env python3
"""Flask-based Face Recognition Inference Service for Jetson with RTSP Streaming"""
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
import signal
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class JetsonFaceRecognitionRTSP:
    def __init__(self, model_service_url=None, temp_dir="temp_frames", rtsp_port="8554"):
        # Model service URL
        self.model_service_url = model_service_url or os.environ.get('MODEL_SERVICE_URL', 'http://localhost:5000')
        logger.info(f"Using model service: {self.model_service_url}")
        
        # RTSP configuration
        self.rtsp_port = rtsp_port
        self.rtsp_process = None
        self.rtsp_running = False
        self.rtsp_feed_running = False
        self.rtsp_feed_thread = None
        
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
        self.frame_width = 1280
        self.frame_height = 720
        
        # Most recent processed frame and results for streaming
        self.last_frame = None
        self.last_results = []
        self.last_method = "none"
        
        # For camera streaming and processing
        self.camera_running = False
        self.processing_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.processed_frame_queue = queue.Queue(maxsize=5)
        self.camera_thread = None
        self.processing_thread = None
        
        # Detect best camera method
        self.camera_method = self.detect_camera_method()
        self.opencv_camera_id = None
        if self.camera_method == "opencv":
            self.opencv_camera_id = self.find_opencv_camera()
        
        logger.info(f"Using camera method: {self.camera_method}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_rtsp_server()
        self.stop_streaming()
        sys.exit(0)
        
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
            cmd.append("fakesink") 
        return cmd

    def test_csi_camera(self):
        """Test if CSI camera works by capturing a single frame"""
        test_path = os.path.join(self.temp_dir, "test_csi.jpg")
        
        if os.path.exists(test_path):
            os.remove(test_path)
            
        gst_cmd_list = self._construct_csi_gst_command(num_buffers=1, location=test_path)
        
        logger.info(f"CSI camera test command: {' '.join(gst_cmd_list)}")
        
        try:
            result = subprocess.run(
                gst_cmd_list,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            logger.debug(f"CSI test stdout: {result.stdout.strip()}")
            logger.debug(f"CSI test stderr: {result.stderr.strip()}")
            
            success = (
                result.returncode == 0 and
                "ERROR:" not in result.stderr and
                "failed" not in result.stderr and
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
        for camera_id in [0, 1, 2]:
            logger.debug(f"Testing OpenCV camera ID: {camera_id}")
            try:
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    logger.debug(f"OpenCV camera ID {camera_id} not opened.")
                    cap.release()
                    continue
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                ret, frame = cap.read()
                cap.release()
                
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
        gst_cmd_list = self._construct_csi_gst_command(num_buffers=1, location=self.temp_frame_path)
        
        try:
            if os.path.exists(self.temp_frame_path):
                os.remove(self.temp_frame_path)
            
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
        
        cap = None
        try:
            cap = cv2.VideoCapture(self.opencv_camera_id)
            if not cap.isOpened():
                logger.error("Failed to open OpenCV camera")
                return None
        
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"OpenCV camera actual settings: {actual_width}x{actual_height}")
        
            # Skip a few frames to let camera stabilize
            for _ in range(3):
                ret, frame = cap.read()
                if not ret:
                    logger.warning("OpenCV: Failed to read frame during warmup.")
                    break
                time.sleep(0.05)
        
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
                cap.release()

    def detect_faces(self, frame):
        """Detect faces in frame using Haar Cascade"""
        faces = []
        
        try:
            if not hasattr(self, 'detector'):
                haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not os.path.exists(haar_cascade_path):
                    logger.error(f"Haar cascade file not found: {haar_cascade_path}")
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

    def start_rtsp_server(self):
        """Start RTSP server that will stream processed frames"""
        if self.rtsp_running:
            logger.warning("RTSP server already running")
            return True
            
        try:
            # For now, let's use the simpler approach - stream raw camera with overlays added via a separate process
            # This avoids the complexity of the named pipe approach
            
            # Choose appropriate GStreamer pipeline based on camera method
            if self.camera_method == "csi":
                gst_pipeline = (
                    f"nvarguscamerasrc ! "
                    f"video/x-raw(memory:NVMM),width={self.frame_width},height={self.frame_height},framerate=30/1 ! "
                    f"nvvidconv ! "
                    f"nvv4l2h264enc bitrate=2000000 ! "
                    f"h264parse ! "
                    f"rtph264pay name=pay0 pt=96"
                )
            else:
                # Fallback pipeline for USB/OpenCV cameras
                gst_pipeline = (
                    f"v4l2src device=/dev/video0 ! "
                    f"videoconvert ! "
                    f"videoscale ! "
                    f"video/x-raw,width={self.frame_width},height={self.frame_height},framerate=30/1 ! "
                    f"x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
                    f"h264parse ! "
                    f"rtph264pay name=pay0 pt=96"
                )
            
            # Use the compiled test-launch binary
            rtsp_cmd = [
                "/app/test-launch",
                f"( {gst_pipeline} )"
            ]
            
            logger.info(f"Starting RTSP server with camera pipeline: {gst_pipeline}")
            
            # Start RTSP server process (no stdin needed for direct camera access)
            self.rtsp_process = subprocess.Popen(
                rtsp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if process is still running
            if self.rtsp_process.poll() is None:
                self.rtsp_running = True
                logger.info(f"RTSP server started on port {self.rtsp_port}")
                logger.info(f"Raw camera stream available at: rtsp://localhost:{self.rtsp_port}/test")
                logger.info("Note: This streams raw camera feed. For processed frames with overlays, use /capture endpoint")
                return True
            else:
                stdout, stderr = self.rtsp_process.communicate()
                logger.error(f"RTSP server failed to start. stdout: {stdout}, stderr: {stderr}")
                return False
            
        except Exception as e:
            logger.error(f"Error starting RTSP server: {e}")
            return False
    
    # Remove the problematic _rtsp_feed_loop method - not needed with direct camera access
    
    def stop_rtsp_server(self):
        """Stop RTSP server"""
        self.rtsp_running = False
        self.rtsp_feed_running = False
        
        # Wait for feed thread to stop
        if hasattr(self, 'rtsp_feed_thread') and self.rtsp_feed_thread and self.rtsp_feed_thread.is_alive():
            self.rtsp_feed_thread.join(timeout=3)
        
        # Stop RTSP process
        if self.rtsp_process:
            try:
                if self.rtsp_process.stdin:
                    self.rtsp_process.stdin.close()
                self.rtsp_process.terminate()
                self.rtsp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rtsp_process.kill()
                self.rtsp_process.wait()
            except Exception as e:
                logger.debug(f"Error stopping RTSP process: {e}")
            finally:
                self.rtsp_process = None
                logger.info("RTSP server stopped")
        
        # Clean up named pipe
        if hasattr(self, 'rtsp_pipe_path') and os.path.exists(self.rtsp_pipe_path):
            try:
                os.unlink(self.rtsp_pipe_path)
            except Exception as e:
                logger.debug(f"Error removing pipe: {e}")

    def start_streaming(self):
        """Start continuous camera streaming and processing"""
        if self.camera_running:
            logger.warning("Streaming already running")
            return
            
        self.camera_running = True
        self.processing_running = True
        
        # Start camera capture thread
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Streaming started")
        
    def stop_streaming(self):
        """Stop continuous streaming"""
        self.camera_running = False
        self.processing_running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            
        logger.info("Streaming stopped")
        
    def _camera_loop(self):
        """Continuous camera capture loop"""
        while self.camera_running:
            try:
                frame = self.capture_frame()
                if frame is not None:
                    # Add frame to queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Remove oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                            
                time.sleep(1/30)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                time.sleep(1)
                
    def _processing_loop(self):
        """Continuous frame processing loop"""
        while self.processing_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1)
                
                # Process frame
                faces, processed_frame = self.process_frame(frame)
                
                # Store results
                self.last_frame = processed_frame.copy()
                self.last_results = faces
                
                # Add to processed frame queue
                try:
                    self.processed_frame_queue.put_nowait(processed_frame)
                except queue.Full:
                    try:
                        self.processed_frame_queue.get_nowait()
                        self.processed_frame_queue.put_nowait(processed_frame)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)

# Initialize face recognition system
face_recognition = JetsonFaceRecognitionRTSP()

# Flask routes
@app.route("/")
def root():
    """API root - provides basic info"""
    return jsonify({
        "name": "Jetson Face Recognition API with RTSP (Flask)",
        "version": "2.0",
        "camera_method": face_recognition.camera_method,
        "rtsp_running": face_recognition.rtsp_running,
        "rtsp_url": f"rtsp://localhost:{face_recognition.rtsp_port}/test" if face_recognition.rtsp_running else None,
        "endpoints": {
            "GET /": "This info",
            "GET /video_player": "Web interface to view processed video stream",
            "GET /video_feed": "MJPEG stream of processed frames",
            "POST /capture": "Capture and process a single frame",
            "POST /start_rtsp": "Start RTSP streaming server (raw camera)",
            "POST /stop_rtsp": "Stop RTSP streaming server",
            "POST /start_streaming": "Start continuous camera streaming",
            "POST /stop_streaming": "Stop continuous camera streaming",
            "GET /status": "Get current status",
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

@app.route("/start_rtsp", methods=['POST'])
def start_rtsp():
    """Start RTSP streaming server"""
    try:
        if face_recognition.start_rtsp_server():
            return jsonify({
                "success": True,
                "message": "RTSP server started",
                "rtsp_url": f"rtsp://localhost:{face_recognition.rtsp_port}/test",
                "port": face_recognition.rtsp_port
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to start RTSP server"
            }), 500
    except Exception as e:
        logger.error(f"Error starting RTSP server: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/stop_rtsp", methods=['POST'])
def stop_rtsp():
    """Stop RTSP streaming server"""
    try:
        face_recognition.stop_rtsp_server()
        return jsonify({
            "success": True,
            "message": "RTSP server stopped"
        })
    except Exception as e:
        logger.error(f"Error stopping RTSP server: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/start_streaming", methods=['POST'])
def start_streaming():
    """Start continuous camera streaming and processing"""
    try:
        face_recognition.start_streaming()
        return jsonify({
            "success": True,
            "message": "Streaming started",
            "camera_method": face_recognition.camera_method
        })
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/stop_streaming", methods=['POST'])
def stop_streaming():
    """Stop continuous camera streaming and processing"""
    try:
        face_recognition.stop_streaming()
        return jsonify({
            "success": True,
            "message": "Streaming stopped"
        })
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/status")
def status():
    """Get current status"""
    try:
        return jsonify({
            "camera_method": face_recognition.camera_method,
            "camera_running": face_recognition.camera_running,
            "processing_running": face_recognition.processing_running,
            "rtsp_running": face_recognition.rtsp_running,
            "rtsp_url": f"rtsp://localhost:{face_recognition.rtsp_port}/test" if face_recognition.rtsp_running else None,
            "fps": face_recognition.fps,
            "avg_inference_time": face_recognition.avg_inference_time,
            "last_results": face_recognition.last_results,
            "frame_queue_size": face_recognition.frame_queue.qsize(),
            "processed_queue_size": face_recognition.processed_frame_queue.qsize(),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/video_feed")
def video_feed():
    """Video streaming route for processed frames (MJPEG)"""
    def generate():
        while True:
            try:
                # Get the latest processed frame
                if face_recognition.last_frame is not None:
                    frame = face_recognition.last_frame.copy()
                    
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    if ret:
                        # Yield frame in MJPEG format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                    else:
                        # If encoding fails, wait and continue
                        time.sleep(0.1)
                else:
                    # No frame available, wait
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in video feed: {e}")
                break
                
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_player")
def video_player():
    """Simple HTML page to view the processed video stream"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition Live Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; }
            .video-container { text-align: center; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; border: 2px solid #333; border-radius: 5px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .status { margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Face Recognition Live Stream</h1>
            
            <div class="video-container">
                <img src="/video_feed" alt="Live Video Stream" id="videoStream">
                <div class="status" id="status">Loading...</div>
            </div>
            
            <div class="controls">
                <button onclick="refreshStream()">🔄 Refresh Stream</button>
                <button onclick="getStatus()">📊 Get Status</button>
                <button onclick="captureFrame()">📷 Capture Frame</button>
            </div>
            
            <div id="info"></div>
        </div>
        
        <script>
            function refreshStream() {
                const img = document.getElementById('videoStream');
                img.src = img.src.split('?')[0] + '?' + new Date().getTime();
            }
            
            function getStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerHTML = 
                            `FPS: ${data.fps.toFixed(1)} | ` +
                            `Inference: ${data.avg_inference_time.toFixed(1)}ms | ` +
                            `Faces: ${data.last_results.length} | ` +
                            `Camera: ${data.camera_running ? '✅' : '❌'} | ` +
                            `RTSP: ${data.rtsp_running ? '✅' : '❌'}`;
                    })
                    .catch(error => {
                        document.getElementById('status').innerHTML = '❌ Error getting status';
                    });
            }
            
            function captureFrame() {
                fetch('/capture', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert(`Frame captured! Faces detected: ${data.faces.length}`);
                        } else {
                            alert('Capture failed: ' + data.error);
                        }
                    })
                    .catch(error => {
                        alert('Error: ' + error);
                    });
            }
            
            // Auto-refresh status every 2 seconds
            setInterval(getStatus, 2000);
            getStatus(); // Initial call
        </script>
    </body>
    </html>
    """
    return html

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
            "camera_method": face_recognition.camera_method,
            "rtsp_running": face_recognition.rtsp_running
        })
    except Exception as e:
        logger.error(f"Error in ping: {e}")
        return jsonify({
            "status": "ERROR",
            "model_service": "ERROR", 
            "timestamp": datetime.now().isoformat(),
            "camera_method": "ERROR",
            "rtsp_running": False
        }), 500

if __name__ == "__main__":
    try:
        # Start streaming automatically on startup
        face_recognition.start_streaming()
        
        # Optionally start RTSP server automatically
        rtsp_auto_start = os.environ.get('RTSP_AUTO_START', 'true').lower() == 'true'
        if rtsp_auto_start:
            face_recognition.start_rtsp_server()
        
        app.run(host="0.0.0.0", port=5001, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        face_recognition.stop_rtsp_server()
        face_recognition.stop_streaming()
    except Exception as e:
        logger.error(f"Error running application: {e}")
        face_recognition.stop_rtsp_server()
        face_recognition.stop_streaming()