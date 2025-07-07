#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import threading
import queue
import logging
import subprocess
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JetsonFaceRecognition:
    def __init__(self, model_dir="models", temp_dir="temp_frames"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Model directory
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Temporary directory for frame capture
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Temp frame path
        self.temp_frame_path = os.path.join(self.temp_dir, "current_frame.jpg")
        
        # Load models and configurations
        self.load_models()
        
        # Performance metrics
        self.fps = 0
        self.avg_inference_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.inference_times = []
        
        # Frame dimensions (adjust based on your camera setup)
        self.frame_width = 1280
        self.frame_height = 720
        
    def load_models(self):
        """Load trained models from edge server"""
        logger.info("Loading models...")
        
        # Load face recognition model package
        model_path = os.path.join(self.model_dir, "face_model.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Using default configuration.")
            # Create default empty database
            self.face_database = {}
            self.face_features = {}
            self.similarity_threshold = 0.75
        else:
            with open(model_path, 'rb') as f:
                model_artifacts = pickle.load(f)
            
            self.face_database = model_artifacts["face_database"]
            self.face_features = model_artifacts["face_features"]
            self.similarity_threshold = model_artifacts.get("similarity_threshold", 0.75)
            
            logger.info(f"Loaded model with {len(self.face_database)} registered identities")
        
        # Load feature extractor
        self.load_feature_extractor()
        
        # Load face detection model
        self.load_detection_model()
        
    def load_feature_extractor(self):
        """Load feature extraction model"""
        try:
            # Load MobileNetV3 for Jetson (lightweight)
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            
            feature_extractor.to(self.device)
            feature_extractor.eval()
            
            # Optimize for inference on Jetson
            if self.device == 'cuda':
                # Use TorchScript for optimization
                example_input = torch.randn(1, 3, 224, 224).to(self.device)
                self.feature_extractor = torch.jit.trace(feature_extractor, example_input)
            else:
                self.feature_extractor = feature_extractor
                
            logger.info("Feature extractor loaded and optimized")
        except Exception as e:
            logger.error(f"Error loading feature extractor: {e}")
            logger.info("Using fallback histogram features")
            self.feature_extractor = None
            
    def load_detection_model(self):
        """Load face detection model"""
        # Use OpenCV's DNN module with a lightweight face detection model
        try:
            # Try to load a pre-trained face detection model
            prototxt_path = os.path.join(self.model_dir, "deploy.prototxt")
            model_path = os.path.join(self.model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.detector_type = "dnn"
                logger.info("Loaded DNN face detector")
            else:
                # Fallback to Haar Cascade
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.detector_type = "cascade"
                logger.info("Using Haar Cascade face detector")
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
            # Final fallback
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detector_type = "cascade"
    
    def capture_frame_gstreamer(self, width=1280, height=720):
        """Capture a single frame using GStreamer"""
        # GStreamer command to capture a single frame
        gst_cmd = (
            f"gst-launch-1.0 -e nvarguscamerasrc num-buffers=1 ! "
            f"'video/x-raw(memory:NVMM),width={width}, height={height}, framerate=30/1' ! "
            f"nvvidconv ! jpegenc ! filesink location={self.temp_frame_path}"
        )
        
        try:
            # Run with timeout to prevent hanging
            process = subprocess.Popen(gst_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for process with timeout
            timeout = 3  # seconds
            start_time = time.time()
            
            while process.poll() is None and time.time() - start_time < timeout:
                time.sleep(0.01)
            
            # If still running after timeout, terminate
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning("GStreamer process had to be killed")
            
            # Check if file was created
            if os.path.exists(self.temp_frame_path):
                # Check file size and creation time
                filesize = os.path.getsize(self.temp_frame_path)
                if filesize == 0:
                    logger.warning("Captured frame file is empty")
                    return None
                
                # Load frame with OpenCV
                frame = cv2.imread(self.temp_frame_path)
                if frame is None:
                    logger.warning("Failed to load captured frame")
                    return None
                
                return frame
            else:
                logger.warning("Frame capture failed - no file created")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
        
    def detect_faces(self, frame):
        """Detect faces in frame"""
        faces = []
        
        if self.detector_type == "dnn":
            # Prepare blob
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                       (300, 300), (104.0, 177.0, 123.0))
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            # Process detections
            h, w = frame.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    faces.append({
                        "box": (x1, y1, x2-x1, y2-y1),
                        "confidence": float(confidence)
                    })
        else:
            # Haar Cascade detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.detector.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            for (x, y, w, h) in detected:
                faces.append({
                    "box": (x, y, w, h),
                    "confidence": 0.9
                })
                
        return faces
    
    def extract_features(self, face_img):
        """Extract features from face image"""
        if self.feature_extractor is None:
            # Fallback: histogram features
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        
        # Neural network features
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = transform(face_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = features.squeeze().cpu().numpy()
            
        return features
    
    def recognize_face(self, face_roi):
        """Recognize face"""
        # Extract features
        query_features = self.extract_features(face_roi)
        
        best_match = "Unknown"
        best_similarity = -1
        
        # Compare with all known faces
        for person_id, features_list in self.face_features.items():
            similarities = []
            
            for features in features_list:
                if features.shape != query_features.shape:
                    continue
                    
                # Calculate cosine similarity
                sim = cosine_similarity([query_features], [features])[0][0]
                similarities.append(sim)
            
            if similarities:
                # Average of top-3 similarities
                similarities.sort(reverse=True)
                top_n = min(3, len(similarities))
                avg_similarity = sum(similarities[:top_n]) / top_n
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    if avg_similarity >= self.similarity_threshold:
                        best_match = self.face_database.get(person_id, "Unknown")
        
        return {
            "name": best_match,
            "confidence": best_similarity * 100 if best_match != "Unknown" else 0
        }
    
    def process_single_frame(self, save_output=False, display_frame=None):
        """Process a single frame for face recognition"""
        # Capture frame using GStreamer
        frame = self.capture_frame_gstreamer(self.frame_width, self.frame_height)
        
        if frame is None:
            logger.warning("Failed to capture frame")
            return None
        
        inference_start = time.time()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Recognize each face
        for face in faces:
            x, y, w, h = face["box"]
            
            # Ensure valid ROI
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w > 30 and h > 30:  # Minimum face size
                face_roi = frame[y:y+h, x:x+w]
                result = self.recognize_face(face_roi)
                
                # Draw results
                color = (0, 255, 0) if result["name"] != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{result['name']}"
                if result["confidence"] > 0:
                    label += f" ({result['confidence']:.0f}%)"
                
                cv2.putText(frame, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate performance
        inference_time = time.time() - inference_start
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Update FPS every second
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.avg_inference_time = np.mean(self.inference_times[-30:]) * 1000
            
            # Reset counters
            self.frame_count = 0
            self.start_time = time.time()
        
        # Display performance info
        info_text = f"FPS: {self.fps:.1f} | Inference: {self.avg_inference_time:.1f}ms | Faces: {len(faces)}"
        cv2.putText(frame, info_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save output if requested
        if save_output:
            output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(output_path, frame)
            
        # For headless mode, save periodic frames
        if self.frame_count % 30 == 0 or len(faces) > 0:
            headless_path = f"detected_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(headless_path, frame)
            if len(faces) > 0:
                logger.info(f"Detected {len(faces)} faces, saved to {headless_path}")
        
        return frame
    
    def run_inference(self, display=False, save_output=False, duration=None):
        """Run real-time face recognition"""
        logger.info("Starting face recognition inference...")
        
        # Check if we're in a headless environment (SSH)
        has_display = os.environ.get('DISPLAY') is not None and display
        
        if display and not has_display:
            logger.info("Detected headless environment (SSH session). Running in headless mode.")
            display = False
        
        # Setup video writer if saving video
        writer = None
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 10.0, (self.frame_width, self.frame_height))
            logger.info(f"Recording to {output_path}")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Check if duration specified and exceeded
                if duration is not None and (time.time() - start_time) > duration:
                    logger.info(f"Reached specified duration of {duration} seconds")
                    break
                
                # Process a single frame
                frame = self.process_single_frame(save_output=False)
                
                if frame is None:
                    logger.warning("Failed to process frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Save to video if requested
                if writer:
                    writer.write(frame)
                
                # Display frame if we have a display
                if has_display:
                    cv2.imshow("Face Recognition - Jetson Nano", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Limit frame rate in headless mode to reduce CPU usage
                if not has_display:
                    time.sleep(0.1)  # Approximately 10 FPS
                    
                    # Log status periodically
                    if frame_count % 10 == 0:
                        elapsed = time.time() - start_time
                        avg_fps = frame_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Running: {frame_count} frames, Avg FPS: {avg_fps:.1f}")
                        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
        finally:
            # Cleanup
            if writer:
                writer.release()
            if has_display:
                cv2.destroyAllWindows()
            
            # Final stats
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"Inference stopped. Processed {frame_count} frames in {elapsed:.1f}s (Avg FPS: {avg_fps:.1f})")
    
    def fetch_model_from_edge(self, edge_ip, edge_user="edgeuser", edge_path="~/face_recognition/jetson_deployment"):
        """Fetch latest model from edge server using SCP"""
        logger.info(f"Fetching model from {edge_user}@{edge_ip}...")
        
        try:
            # Create temporary directory for new model
            temp_dir = "models_temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # SCP command to fetch model files
            import subprocess
            
            # Fetch model file
            scp_cmd = f"scp {edge_user}@{edge_ip}:{edge_path}/face_model.pkl {temp_dir}/"
            result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to fetch model: {result.stderr}")
                return False
            
            # Fetch database file
            scp_cmd = f"scp {edge_user}@{edge_ip}:{edge_path}/face_database.json {temp_dir}/"
            subprocess.run(scp_cmd, shell=True)
            
            # Backup current model
            backup_dir = f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if os.path.exists(self.model_dir):
                os.rename(self.model_dir, backup_dir)
                logger.info(f"Current model backed up to {backup_dir}")
            
            # Move new model to production
            os.rename(temp_dir, self.model_dir)
            
            # Reload models
            self.load_models()
            
            logger.info("Model updated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching model: {e}")
            return False
    
    def benchmark(self, duration=30):
        """Run performance benchmark"""
        logger.info(f"Running {duration} second benchmark...")
        
        metrics = {
            "frames": 0,
            "faces_detected": 0,
            "faces_recognized": 0,
            "inference_times": []
        }
        
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                # Capture and process frame
                frame = self.capture_frame_gstreamer(self.frame_width, self.frame_height)
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                inference_start = time.time()
                
                # Process frame
                faces = self.detect_faces(frame)
                metrics["faces_detected"] += len(faces)
                
                for face in faces:
                    x, y, w, h = face["box"]
                    if w > 30 and h > 30:
                        face_roi = frame[y:y+h, x:x+w]
                        result = self.recognize_face(face_roi)
                        if result["name"] != "Unknown":
                            metrics["faces_recognized"] += 1
                
                inference_time = time.time() - inference_start
                metrics["inference_times"].append(inference_time)
                metrics["frames"] += 1
                
                # Add a small delay to avoid overwhelming the system
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            logger.info("Benchmark interrupted by user")
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_fps = metrics["frames"] / total_time
        avg_inference = np.mean(metrics["inference_times"]) * 1000
        
        logger.info("\nBenchmark Results:")
        logger.info(f"  Duration: {total_time:.1f}s")
        logger.info(f"  Frames processed: {metrics['frames']}")
        logger.info(f"  Average FPS: {avg_fps:.1f}")
        logger.info(f"  Average inference: {avg_inference:.1f}ms")
        logger.info(f"  Faces detected: {metrics['faces_detected']}")
        logger.info(f"  Faces recognized: {metrics['faces_recognized']}")
        
        return {
            "fps": avg_fps,
            "inference_ms": avg_inference,
            "total_frames": metrics["frames"]
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Face Recognition on Jetson Nano")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark/run duration in seconds")
    parser.add_argument("--display", action="store_true", help="Display output (if display available)")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--fetch-model", action="store_true", help="Fetch latest model from edge")
    parser.add_argument("--edge-ip", default="192.168.50.1", help="Edge server IP")
    parser.add_argument("--edge-user", default="edgeuser", help="Edge server username")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    
    args = parser.parse_args()
    
    # Initialize face recognition system
    face_rec = JetsonFaceRecognition()
    
    # Set frame dimensions
    face_rec.frame_width = args.width
    face_rec.frame_height = args.height
    
    # Fetch model if requested
    if args.fetch_model:
        face_rec.fetch_model_from_edge(args.edge_ip, args.edge_user)
    
    # Run benchmark or inference
    if args.benchmark:
        face_rec.benchmark(duration=args.duration)
    else:
        face_rec.run_inference(display=args.display, save_output=args.save, duration=args.duration)


if __name__ == "__main__":
    main()