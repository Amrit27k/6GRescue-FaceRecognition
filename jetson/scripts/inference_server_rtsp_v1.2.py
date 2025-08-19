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
import time
import requests
import signal
import sys
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np # Added for numpy operations
import cv2 # Added for image processing, make sure it's installed (pip install opencv-python)
from PIL import Image # Needed for torchvision.transforms.ToPILImage()

# New import for YOLOv8
from ultralytics import YOLO
# New import for MQTT
import paho.mqtt.client as mqtt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class JetsonFaceRecognitionRTSP:
    def __init__(self, model_service_url=None, rtsp_port="8554", mqtt_broker_host="10.70.0.64", mqtt_port=1883):
        # Model service URL
        self.model_service_url = model_service_url or os.environ.get('MODEL_SERVICE_URL', 'http://localhost:5000')
        logger.info(f"Using model service: {self.model_service_url}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.rtsp_port = rtsp_port
        self.rtsp_process = None
        self.rtsp_running = False

        # Threads for logging subprocess output
        self.rtsp_stdout_thread = None
        self.rtsp_stderr_thread = None

        # YOLOv8n model for face detection
        self.yolo_model = None

        # Feature extractor (foundation model)
        self.feature_extractor = self._load_feature_extractor() # Renamed to _load_feature_extractor for consistency

        # RTSP Capture object
        self.rtsp_capture = None

        # MQTT Configuration
        self.mqtt_broker_host = mqtt_broker_host
        self.mqtt_port = mqtt_port
        self.mqtt_topic = "jetson/face_recognition/results" # Topic to publish results
        self.mqtt_client = None

        # Output directory for saved results (for JSON file)
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Output JSON file path for results
        self.results_file_path = os.path.join(self.output_dir, "face_recognition_results.json")

        # Performance metrics
        self.fps = 0 # This will be the FPS of the Flask app's processing
        self.avg_inference_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.inference_times = []

        # Frame dimensions (can be adjusted if the RTSP stream provides different dimensions)
        self.frame_width = 1280
        self.frame_height = 720

        # Most recent processed frame for streaming (e.g., via /video_feed)
        # This will now store the raw frame (no overlays)
        self.last_frame = None
        self.last_results = [] # This will store the raw detection/recognition results for status

        # For continuous camera streaming and processing
        self.camera_running = False
        self.processing_running = False
        self.frame_queue = queue.Queue(maxsize=5) # Queue for raw frames from camera loop
        self.camera_thread = None
        self.processing_thread = None

        # Camera method detection is now primarily for the C RTSP server's pipeline
        self.camera_method = self.detect_camera_method()
        logger.info(f"Detected camera method for RTSP server: {self.camera_method}")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self._init_mqtt()

    def _init_mqtt(self):
        """Initializes and connects the MQTT client."""
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

        try:
            self.mqtt_client.connect(self.mqtt_broker_host, self.mqtt_port, 60)
            self.mqtt_client.loop_start() # Start MQTT client loop in a background thread
            logger.info(f"Attempting to connect to MQTT broker at {self.mqtt_broker_host}:{self.mqtt_port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected successfully to MQTT Broker at {self.mqtt_broker_host}!")
        else:
            logger.error(f"Failed to connect to MQTT Broker at {self.mqtt_broker_host}, return code {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from MQTT Broker at {self.mqtt_broker_host} with code {rc}. Attempting to reconnect...")
        # Paho's loop_start() automatically handles reconnects, so explicit reconnect logic is often not needed here.

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_rtsp_server()
        self.stop_streaming() # Stop continuous streaming
        if self.mqtt_client:
            self.mqtt_client.loop_stop() # Stop MQTT background loop
            self.mqtt_client.disconnect()
            logger.info("MQTT client disconnected.")
        sys.exit(0)

    def detect_camera_method(self):
        """Detect the best available camera method for the C RTSP server."""
        logger.info("Attempting CSI camera detection for RTSP server pipeline...")
        if self.test_csi_camera():
            logger.info("CSI camera detected and tested successfully for RTSP server.")
            return "csi"
        else:
            logger.info("CSI camera test failed or not available for RTSP server.")

        logger.info("Attempting USB camera (v4l2src) detection for RTSP server pipeline...")
        try:
            result = subprocess.run("gst-inspect-1.0 v4l2src", shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and os.path.exists("/dev/video0"):
                logger.info("v4l2src plugin found and /dev/video0 exists. Testing USB camera for RTSP server...")
                if self.test_usb_camera():
                    logger.info("USB camera detected and tested successfully for RTSP server.")
                    return "usb"
                else:
                    logger.info("USB camera test failed for RTSP server.")
            else:
                logger.info(f"v4l2src inspect return: {result.returncode}, stderr: {result.stderr.strip()}")
                logger.info("USB camera requirements (v4l2src or /dev/video0) not met for RTSP server.")
        except subprocess.TimeoutExpired:
            logger.warning("gst-inspect-1.0 v4l2src timed out.")
        except Exception as e:
            logger.error(f"Error during USB camera detection check for RTSP server: {e}")

        logger.warning("No suitable camera method detected for RTSP server. Defaulting to CSI (might fail).")
        return "csi" # Fallback, though it might not work if no CSI/USB cam

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
        """Test if CSI camera works by capturing a single frame (for RTSP server pipeline detection)"""
        temp_dir = "/tmp" # Use /tmp for temporary test files
        test_path = os.path.join(temp_dir, "test_csi.jpg")

        if os.path.exists(test_path):
            os.remove(test_path)

        gst_cmd_list = self._construct_csi_gst_command(num_buffers=1, location=test_path)

        logger.debug(f"CSI camera test command: {' '.join(gst_cmd_list)}")

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

            if os.path.exists(test_path):
                os.remove(test_path) # Clean up test file
            return success
        except subprocess.TimeoutExpired:
            logger.error("CSI camera test timed out.")
            return False
        except Exception as e:
            logger.error(f"Error during CSI camera test: {e}")
            return False

    def test_usb_camera(self):
        """Test if USB camera works (for RTSP server pipeline detection)"""
        temp_dir = "/tmp" # Use /tmp for temporary test files
        test_path = os.path.join(temp_dir, "test_usb.jpg")

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
            f"video/x-raw,width={self.frame_width},height={self.frame_height},framerate=30/1 ! ", # Adjusted for GStreamer
            "jpegenc",
            "!",
            "filesink",
            f"location={test_path}"
        ]

        logger.debug(f"USB camera test command: {' '.join(gst_cmd_list)}")

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

            if os.path.exists(test_path):
                os.remove(test_path) # Clean up test file
            return success
        except subprocess.TimeoutExpired:
            logger.error("USB camera test timed out.")
            return False
        except Exception as e:
            logger.error(f"Error during USB camera test: {e}")
            return False

    def get_frame_from_rtsp_stream(self):
        """Captures a single frame from the RTSP stream using OpenCV."""
        rtsp_url = f"rtsp://localhost:{self.rtsp_port}/test"

        if self.rtsp_capture is None or not self.rtsp_capture.isOpened():
            logger.info(f"Attempting to open RTSP stream: {rtsp_url}")
            # Ensure OpenCV is properly linked to GStreamer on Jetson for RTSP capture
            self.rtsp_capture = cv2.VideoCapture(rtsp_url) # Use CAP_GSTREAMER backend

            if not self.rtsp_capture.isOpened():
                logger.error(f"Failed to open RTSP stream at {rtsp_url}. Ensure 'rtsp_server_fps' is running and streaming.")
                return None

            # Set buffer size to 1 to get the latest frame and avoid latency
            self.rtsp_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Attempt to set frame dimensions, though the stream might dictate them
            # These are often set by the GStreamer pipeline on the server side.
            self.rtsp_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.rtsp_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            logger.info(f"RTSP stream opened successfully: {rtsp_url}")

        ret, frame = self.rtsp_capture.read()
        if not ret:
            logger.warning("Failed to read frame from RTSP stream. Stream might be disconnected or empty. Attempting to re-open.")
            self.rtsp_capture.release()
            self.rtsp_capture = None # Force re-initialization on next call
            return None

        return frame

    def _load_feature_extractor(self): # Renamed for consistency
        """Load foundation model for feature extraction"""
        logger.info("Loading feature extraction model...")
        try:
            # Using MobileNetV3 for edge deployment, matching the training script
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            # Remove the last layer (classifier)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_extractor.to(self.device)
            feature_extractor.eval() # Set to evaluation mode
            logger.info("Feature extraction model loaded successfully!")
            return feature_extractor
        except Exception as e:
            logger.error(f"Error loading feature extractor: {e}")
            return None

    def extract_features(self, face_img):
        """Extract features from a face image using foundation model"""
        if self.feature_extractor is None:
            logger.error("Feature extractor not loaded. Cannot extract features. Returning dummy data.")
            # Fallback to dummy data if feature extractor failed to load
            return np.zeros(576) # MobileNetV3_small typically outputs 576 features. Adjust if your model is different.

        # Preprocess image for the feature extractor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = transform(face_img).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = features.squeeze().cpu().numpy()

        return features

    def detect_and_recognize_faces_yolo(self, frame):
        """Detects faces using YOLOv8n and then recognizes them using the model service.
           Does NOT overlay results on the frame; returns the original frame copy."""
        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO('yolov8n.pt') # Ensure this file is accessible
                logger.info("YOLOv8n model loaded successfully during processing.")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8n model: {e}. Please ensure 'ultralytics' is installed and the model file is accessible.")
                return [], frame.copy() # Return empty faces if YOLO fails to load

        faces = []
        processed_frame = frame.copy()

        try:
            # Remove the hardcoded test_frame. Use the 'frame' passed to the function.
            # img_path = "amrit_test.jpg"
            # test_frame = cv2.imread(img_path)
            # if test_frame is None:
            #     logger.info("Test Image file not read properly.")
            # results = self.yolo_model(test_frame, verbose=False, conf=0.25)

            results = self.yolo_model(frame, verbose=False, conf=0.25) # Use the actual live frame
            logger.debug(f"Result from yolo_model: {results}") # Changed to debug level for less verbosity

            for r in results:
                for box in r.boxes:
                    # if int(box.cls[0]) == 0: # Check if the detected object is a 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    logger.debug(f"Detected Box:{int(box.cls[0])} with confidence:{confidence:.2f}") # Debug level

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    w = x2 - x1
                    h = y2 - y1

                    if w > 0 and h > 0:
                        face_roi = frame[y1:y2, x1:x2]

                        if face_roi.size > 0:
                            logger.debug("Extracting features from face_roi.") # Debug level
                            query_features = self.extract_features(face_roi) # Removed .flatten().reshape(1, -1) as .tolist() will handle it

                            # Check if features were extracted successfully (not dummy data)
                            if query_features.size == 0 or np.all(query_features == 0):
                                logger.warning("Feature extraction returned empty or dummy data. Skipping recognition.")
                                recognition_result = {"name": "Feature Error", "confidence": 0, "person_id": None}
                            else:
                                # Pass the extracted features as a Python list to the recognition service
                                recognition_result = self.recognize_face(query_features.tolist())

                            faces.append({
                                "box": [x1, y1, w, h],
                                "name": recognition_result.get("name", "Unknown"),
                                "confidence": recognition_result.get("confidence", 0),
                                "person_id": recognition_result.get("person_id", None)
                            })
                        else:
                            logger.warning(f"Skipping empty face_roi for box: {[x1, y1, w, h]}")
        except Exception as e:
            logger.error(f"Error during YOLOv8n face detection/recognition: {e}", exc_info=True) # Added exc_info
            return [], frame.copy()

        # Update FPS and inference time based on this processing cycle
        inference_time = time.time() - self.start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1

        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.avg_inference_time = np.mean(self.inference_times[-30:]) * 1000
            self.frame_count = 0
            self.start_time = time.time()

        return faces, processed_frame # processed_frame is now the raw frame copy

    def recognize_face(self, query_features_list: List[float]): # Changed parameter name for clarity and type hint
        """Recognize face using model service"""
        try:
            data = {
                "instances": [
                    {"image_feature_vector": query_features_list} # CORRECTED: Send as image_feature_vector
                ]
            }

            response = requests.post(
                f"{self.model_service_url}/invocations",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10 # Increased timeout for robustness
            )
            logger.debug(f"Model API response status: {response.status_code}") # Debug level

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])
                if predictions:
                    return predictions[0]
                else:
                    logger.warning(f"Model API returned empty predictions: {response.text}")
                    return {"name": "No Prediction", "confidence": 0, "person_id": None}
            else:
                logger.error(f"Model API returned non-200 status {response.status_code}: {response.text}")
                return {"name": "API Error", "confidence": 0, "person_id": None}

        except requests.exceptions.Timeout:
            logger.error(f"Model service API call timed out after 10 seconds.")
            return {"name": "Timeout", "confidence": 0, "person_id": None}
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Could not connect to model service at {self.model_service_url}. Is it running? Error: {ce}")
            return {"name": "Connection Error", "confidence": 0, "person_id": None}
        except Exception as e:
            logger.error(f"Error calling model service: {e}", exc_info=True) # Added exc_info for full traceback
            return {"name": "Recognition Error", "confidence": 0, "person_id": None} # Changed 'Error' to 'Recognition Error'

    def save_results_to_json(self, results_data: Dict[str, Any]):
        """Appends the processing results to a JSON file."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            existing_data = []
            if os.path.exists(self.results_file_path) and os.path.getsize(self.results_file_path) > 0:
                with open(self.results_file_path, 'r') as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [] # Ensure it's a list
                    except json.JSONDecodeError:
                        logger.warning(f"Existing JSON file {self.results_file_path} is corrupt or empty. Starting fresh.")
                        existing_data = []

            existing_data.append(results_data)

            with open(self.results_file_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            logger.info(f"Results saved to {self.results_file_path}")
        except Exception as e:
            logger.error(f"Error saving results to JSON file: {e}")

    def _log_subprocess_output(self, pipe, log_level_func, process_running_flag):
        """Reads output from a subprocess pipe and logs it."""
        while process_running_flag():
            line = pipe.readline()
            if line:
                log_level_func(f"RTSP Server: {line.strip()}")
                if log_level_func == logger.info:
                    sys.stdout.flush()
                elif log_level_func == logger.error:
                    sys.stderr.flush()
            else:
                time.sleep(0.01) # Small delay to prevent busy-waiting
                if not process_running_flag():
                    break

    def start_rtsp_server(self):
        """Start RTSP server that will stream raw camera frames."""
        if self.rtsp_running:
            logger.warning("RTSP server already running")
            return True

        try:
            logger.info("inside rtsp_server method")
            if self.camera_method == "csi":
                gst_pipeline = (
                    f"nvarguscamerasrc ! "
                    f"video/x-raw(memory:NVMM),width={self.frame_width},height={self.frame_height},framerate=30/1,format=NV12 ! "
                    f"nvv4l2h264enc bitrate=2000000 ! "
                    f"h264parse ! "
                    f"rtph264pay name=pay0 pt=96"
                )
            elif self.camera_method == "usb":
                gst_pipeline = (
                    f"v4l2src device=/dev/video0 ! "
                    f"videoconvert ! "
                    f"videoscale ! "
                    f"video/x-raw,width={self.frame_width},height={self.frame_height},framerate=30/1 ! "
                    f"x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
                    f"h264parse ! "
                    f"rtph264pay name=pay0 pt=96"
                )
            else:
                logger.error("No valid camera method detected for RTSP server, cannot start.")
                return False

            rtsp_cmd = [
                "./rtsp_server_fps",
                f"( {gst_pipeline} )"
            ]

            logger.info(f"Starting RTSP server with camera pipeline: {gst_pipeline}")

            self.rtsp_process = subprocess.Popen(
                rtsp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            self.rtsp_stdout_thread = threading.Thread(
                target=self._log_subprocess_output,
                args=(self.rtsp_process.stdout, logger.info, lambda: self.rtsp_process.poll() is None or self.rtsp_running),
                daemon=True
            )
            self.rtsp_stderr_thread = threading.Thread(
                target=self._log_subprocess_output,
                args=(self.rtsp_process.stderr, logger.error, lambda: self.rtsp_process.poll() is None or self.rtsp_running),
                daemon=True
            )
            logger.info("starting rtsp threads")
            self.rtsp_stdout_thread.start()
            self.rtsp_stderr_thread.start()

            time.sleep(2)

            if self.rtsp_process.poll() is None:
                self.rtsp_running = True
                logger.info(f"RTSP server started on port {self.rtsp_port}")
                logger.info(f"Raw camera stream available at: rtsp://localhost:{self.rtsp_port}/test")
                logger.info("Note: This Flask app processes frames and sends results via MQTT, it does NOT overlay the stream.")
                return True
            else:
                logger.error("RTSP server failed to start immediately. Checking logs...")
                if self.rtsp_stdout_thread and self.rtsp_stdout_thread.is_alive():
                    self.rtsp_stdout_thread.join(timeout=1)
                if self.rtsp_stderr_thread and self.rtsp_stderr_thread.is_alive():
                    self.rtsp_stderr_thread.join(timeout=1)
                return False

        except Exception as e:
            logger.error(f"Error starting RTSP server: {e}", exc_info=True) # Added exc_info
            return False

    def stop_rtsp_server(self):
        """Stop RTSP server"""
        self.rtsp_running = False

        if self.rtsp_capture and self.rtsp_capture.isOpened():
            self.rtsp_capture.release()
            self.rtsp_capture = None
            logger.info("RTSP stream capture released.")

        if self.rtsp_process:
            try:
                if self.rtsp_process.stdout:
                    self.rtsp_process.stdout.close()
                if self.rtsp_process.stderr:
                    self.rtsp_process.stderr.close()

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

        if self.rtsp_stdout_thread and self.rtsp_stdout_thread.is_alive():
            self.rtsp_stdout_thread.join(timeout=1)
        if self.rtsp_stderr_thread and self.rtsp_stderr_thread.is_alive():
            self.rtsp_stderr_thread.join(timeout=1)

    def _camera_loop(self):
        """Continuous camera capture loop from RTSP stream."""
        while self.camera_running:
            try:
                frame = self.get_frame_from_rtsp_stream()
                if frame is not None:
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait() # Discard oldest frame
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty: # Should not happen right after get, but good for robustness
                            pass
                else:
                    time.sleep(0.1) # Wait if no frame

                time.sleep(0.01) # Small delay to yield CPU, adjust as needed for performance

            except Exception as e:
                logger.error(f"Error in camera loop: {e}", exc_info=True) # Added exc_info
                time.sleep(1) # Longer sleep on error

    def _processing_loop(self):
        """Continuous frame processing loop."""
        while self.processing_running:
            try:
                frame = self.frame_queue.get(timeout=1) # Get frame from queue, with timeout

                # Perform detection and recognition
                #faces, raw_frame_copy = self.detect_and_recognize_faces_yolo(frame)
                image_file = "amrit_test.jpg"
                image_path = os.path.join(image_file)
                logger.info(f"\nProcessing image: {image_file}")

                frame = cv2.imread(image_path)
                if frame is None:
                    logger.error(f"Could not read image")

                # Assuming ground truth can be extracted from filename (e.g., "amrit_1.jpg" -> "amrit")
                ground_truth_name = image_file.split('_')[0].lower() # Adjust parsing based on your filename convention
                logger.info(f"Ground Truth for {image_file}: {ground_truth_name}")
                faces, raw_frame_copy = self.detect_and_recognize_faces_yolo(frame)
                # Prepare data for JSON and MQTT
                results_data = {
                    "timestamp": datetime.now().isoformat(),
                    "frame_dimensions": {"width": frame.shape[1], "height": frame.shape[0]},
                    "detected_faces": faces, # This contains the raw detection and recognition results
                    "flask_processing_fps": self.fps, # FPS of this Flask app's processing loop
                    "avg_inference_time_ms": self.avg_inference_time # From the Flask side processing
                }

                # Save results to local JSON file
                self.save_results_to_json(results_data)

                # Publish results via MQTT
                try:
                    if self.mqtt_client and self.mqtt_client.is_connected():
                        payload = json.dumps(results_data)
                        self.mqtt_client.publish(self.mqtt_topic, payload, qos=0)
                        logger.debug(f"Published results to MQTT topic {self.mqtt_topic}")
                    else:
                        logger.warning(f"MQTT client not connected to {self.mqtt_broker_host}, cannot publish results.")
                except Exception as e:
                    logger.error(f"Error publishing MQTT message: {e}", exc_info=True) # Added exc_info

                # Update last_frame for /video_feed endpoint with the raw frame (no overlays)
                self.last_frame = raw_frame_copy.copy()
                self.last_results = faces # Keep last_results updated for status endpoint

            except queue.Empty:
                continue # No frame in queue, try again
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True) # Added exc_info
                time.sleep(0.1) # Small sleep on error to prevent tight looping

    def start_streaming(self):
        """Start continuous camera streaming and processing."""
        if self.camera_running:
            logger.warning("Streaming already running")
            return

        self.camera_running = True
        self.processing_running = True

        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        logger.info("Continuous streaming and processing started (publishing via MQTT).")

    def stop_streaming(self):
        """Stop continuous streaming and processing."""
        self.camera_running = False
        self.processing_running = False

        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)

        logger.info("Continuous streaming and processing stopped.")


# Initialize face recognition system
# IMPORTANT: Replace "<EDGE_SERVER_IP>" with the actual IP address of your edge server
# where the MQTT broker (Mosquitto) will be running.
# Example: face_recognition = JetsonFaceRecognitionRTSP(mqtt_broker_host="192.168.2.101")
face_recognition = JetsonFaceRecognitionRTSP(mqtt_broker_host=os.environ.get('MQTT_BROKER_HOST', '10.70.0.64'))


# Flask routes
@app.route("/")
def root():
    """API root - provides basic info"""
    return jsonify({
        "name": "Jetson Face Recognition API with RTSP (Flask)",
        "version": "5.2", # Updated version
        "camera_method_for_rtsp_server": face_recognition.camera_method,
        "rtsp_server_running": face_recognition.rtsp_running,
        "rtsp_url": f"rtsp://localhost:{face_recognition.rtsp_port}/test" if face_recognition.rtsp_running else None,
        "continuous_processing_running": face_recognition.camera_running, # New status
        "mqtt_broker": f"{face_recognition.mqtt_broker_host}:{face_recognition.mqtt_port}",
        "mqtt_topic": face_recognition.mqtt_topic,
        "endpoints": {
            "GET /": "This info",
            "GET /video_player": "Web interface to view RAW RTSP stream (no overlays from this app)",
            "GET /video_feed": "MJPEG stream of RAW frames (no overlays from this app)",
            "POST /capture": "Capture and process a single frame, save results to JSON and publish via MQTT",
            "POST /start_rtsp": "Start RTSP streaming server (raw camera)",
            "POST /stop_rtsp": "Stop RTSP streaming server",
            "POST /start_streaming": "Start continuous camera streaming and processing (results via MQTT)",
            "POST /stop_streaming": "Stop continuous camera streaming and processing",
            "GET /status": "Get current status",
            "GET /ping": "Health check"
        }
    })

@app.route("/capture", methods=['POST'])
def capture():
    """Capture and process a single frame from RTSP, save results to JSON and publish via MQTT."""
    try:
        frame = face_recognition.get_frame_from_rtsp_stream()

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Failed to capture frame from RTSP stream. Ensure RTSP server is running and streaming."
            }), 500

        faces, raw_frame_copy = face_recognition.detect_and_recognize_faces_yolo(frame)

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "source_method": "manual_rtsp_capture",
            "frame_dimensions": {"width": frame.shape[1], "height": frame.shape[0]},
            "detected_faces": faces,
            "flask_processing_fps": face_recognition.fps,
            "avg_inference_time_ms": face_recognition.avg_inference_time
        }
        face_recognition.save_results_to_json(results_data)

        # Publish results via MQTT for manual capture as well
        try:
            if face_recognition.mqtt_client and face_recognition.mqtt_client.is_connected():
                payload = json.dumps(results_data)
                face_recognition.mqtt_client.publish(face_recognition.mqtt_topic, payload, qos=0)
                logger.debug(f"Published manual capture results to MQTT topic {face_recognition.mqtt_topic}")
            else:
                logger.warning(f"MQTT client not connected to {face_recognition.mqtt_broker_host}, cannot publish manual capture results.")
        except Exception as e:
            logger.error(f"Error publishing manual MQTT message: {e}", exc_info=True) # Added exc_info

        # Update last_frame for /video_feed with the raw frame (no overlays)
        face_recognition.last_frame = raw_frame_copy.copy()
        face_recognition.last_results = faces

        return jsonify({
            "success": True,
            "message": "Frame captured, processed, results saved to JSON, and published via MQTT.",
            "faces_count": len(faces),
            "results_file": face_recognition.results_file_path,
            "timestamp": datetime.now().isoformat(),
            "method_used": "rtsp_yolo_mqtt"
        })

    except Exception as e:
        logger.error(f"Error processing capture: {e}", exc_info=True) # Added exc_info
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/start_rtsp", methods=['POST'])
def start_rtsp():
    """Start RTSP streaming server (C program)"""
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
                "error": "Failed to start RTSP server. Check logs for details."
            }), 500
    except Exception as e:
        logger.error(f"Error starting RTSP server: {e}", exc_info=True) # Added exc_info
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/stop_rtsp", methods=['POST'])
def stop_rtsp():
    """Stop RTSP streaming server (C program)"""
    try:
        face_recognition.stop_rtsp_server()
        return jsonify({
            "success": True,
            "message": "RTSP server stopped"
        })
    except Exception as e:
        logger.error(f"Error stopping RTSP server: {e}", exc_info=True) # Added exc_info
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/start_streaming", methods=['POST'])
def start_streaming():
    """Start continuous camera streaming and processing (results via MQTT)"""
    try:
        face_recognition.start_streaming()
        return jsonify({
            "success": True,
            "message": "Continuous streaming and processing started. Results will be published via MQTT."
        })
    except Exception as e:
        logger.error(f"Error starting continuous streaming: {e}", exc_info=True) # Added exc_info
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/stop_streaming", methods=['POST'])
def stop_streaming():
    """Stop continuous camera streaming and processing"""
    try:
        face_recognition.stop_streaming()
        return jsonify({
            "success": True,
            "message": "Continuous streaming and processing stopped."
        })
    except Exception as e:
        logger.error(f"Error stopping continuous streaming: {e}", exc_info=True) # Added exc_info
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/status")
def status():
    """Get current status"""
    try:
        return jsonify({
            "camera_method_for_rtsp_server": face_recognition.camera_method,
            "rtsp_server_running": face_recognition.rtsp_running,
            "rtsp_url": f"rtsp://localhost:{face_recognition.rtsp_port}/test" if face_recognition.rtsp_running else None,
            "continuous_processing_running": face_recognition.camera_running,
            "flask_app_processing_fps": face_recognition.fps,
            "avg_inference_time_ms": face_recognition.avg_inference_time,
            "last_processed_faces_count": len(face_recognition.last_results),
            "mqtt_broker": f"{face_recognition.mqtt_broker_host}:{face_recognition.mqtt_port}",
            "mqtt_topic": face_recognition.mqtt_topic,
            "mqtt_connected": face_recognition.mqtt_client.is_connected() if face_recognition.mqtt_client else False,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True) # Added exc_info
        return jsonify({"error": str(e)}), 500

@app.route("/video_feed")
def video_feed():
    """Video streaming route for the RAW RTSP stream (no overlays from this app)"""
    def generate():
        while True:
            try:
                # Get the latest raw frame
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
                logger.error(f"Error in video feed: {e}", exc_info=True) # Added exc_info
                break

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_player")
def video_player():
    """Simple HTML page to view the RAW RTSP stream (no overlays from this app)"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson Face Recognition - Raw Stream</title>
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
            <h1>🎥 Jetson Face Recognition - Raw Stream (MQTT Results)</h1>
            <p>This view shows the raw video stream from the Jetson. Face detection and recognition results are published via MQTT for external overlay.</p>

            <div class="video-container">
                <img src="/video_feed" alt="Live Video Stream" id="videoStream">
                <div class="status" id="status">Loading...</div>
            </div>

            <div class="controls">
                <button onclick="refreshStream()">🔄 Refresh Stream</button>
                <button onclick="getStatus()">📊 Get Status</button>
                <button onclick="captureFrame()">📷 Capture Frame (Manual Process & MQTT)</button>
                <button onclick="startStreaming()">▶️ Start Continuous Processing (MQTT)</button>
                <button onclick="stopStreaming()">⏹️ Stop Continuous Processing</button>
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
                            `RTSP Server: ${data.rtsp_server_running ? '✅' : '❌'} | ` +
                            `Processing: ${data.continuous_processing_running ? '✅' : '❌'} | ` +
                            `MQTT: ${data.mqtt_connected ? '✅' : '❌'} (${data.mqtt_broker}) | ` +
                            `Flask FPS: ${data.flask_app_processing_fps.toFixed(1)} | ` +
                            `Inference: ${data.avg_inference_time_ms.toFixed(1)}ms | ` +
                            `Faces: ${data.last_processed_faces_count}`;
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
                            alert(`Manual capture successful! Faces detected: ${data.faces_count}. Results saved to ${data.results_file} and published via MQTT.`);
                        } else {
                            alert('Manual capture failed: ' + data.error);
                        }
                    })
                    .catch(error => {
                        alert('Error during manual capture: ' + error);
                    });
            }

            function startStreaming() {
                fetch('/start_streaming', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        getStatus();
                    })
                    .catch(error => {
                        alert('Error starting streaming: ' + error);
                    });
            }

            function stopStreaming() {
                fetch('/stop_streaming', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        getStatus();
                    })
                    .catch(error => {
                        alert('Error stopping streaming: ' + error);
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
            "camera_method_for_rtsp_server": face_recognition.camera_method,
            "rtsp_server_running": face_recognition.rtsp_running,
            "continuous_processing_running": face_recognition.camera_running,
            "mqtt_connected": face_recognition.mqtt_client.is_connected() if face_recognition.mqtt_client else False
        })
    except Exception as e:
        logger.error(f"Error in ping: {e}", exc_info=True) # Added exc_info
        return jsonify({
            "status": "ERROR",
            "model_service": "ERROR",
            "timestamp": datetime.now().isoformat(),
            "camera_method_for_rtsp_server": "ERROR",
            "rtsp_server_running": False,
            "continuous_processing_running": False,
            "mqtt_connected": False
        }), 500

if __name__ == "__main__":
    try:
        # Start RTSP server (C program) automatically on startup
        rtsp_auto_start = os.environ.get('RTSP_AUTO_START', 'true').lower() == 'true'
        if rtsp_auto_start:
            face_recognition.start_rtsp_server()

        # Start continuous streaming and processing of frames from RTSP
        face_recognition.start_streaming()

        app.run(host="0.0.0.0", port=5001, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        face_recognition.stop_rtsp_server()
        face_recognition.stop_streaming()
    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True) # Added exc_info
        face_recognition.stop_rtsp_server()
        face_recognition.stop_streaming()