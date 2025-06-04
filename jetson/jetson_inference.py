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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JetsonFaceRecognition:
    def __init__(self, model_dir="models"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Model directory
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load models and configurations
        self.load_models()
        
        # CSI Camera setup
        self.camera = None
        self.camera_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_camera = False
        
        # Performance metrics (simple tracking)
        self.fps = 0
        self.avg_inference_time = 0
        
    def load_models(self):
        """Load trained models from edge server"""
        logger.info("Loading models...")
        
        # Load face recognition model package
        model_path = os.path.join(self.model_dir, "face_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please deploy from edge server first.")
        
        with open(model_path, 'rb') as f:
            model_artifacts = pickle.load(f)
        
        self.face_database = model_artifacts["face_database"]
        self.face_features = model_artifacts["face_features"]
        self.similarity_threshold = model_artifacts.get("similarity_threshold", 0.75)
        
        # Load feature extractor
        self.load_feature_extractor(model_artifacts)
        
        # Load face detection model
        self.load_detection_model()
        
        logger.info(f"Loaded model with {len(self.face_database)} registered identities")
        
    def load_feature_extractor(self, model_artifacts):
        """Load feature extraction model"""
        try:
            # Load MobileNetV3 for Jetson (lightweight)
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=False)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            
            # Load saved weights if available
            if model_artifacts.get("feature_extractor_state"):
                feature_extractor.load_state_dict(model_artifacts["feature_extractor_state"])
            
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
            prototxt_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            
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
    
    def gstreamer_pipeline(self, capture_width=1280, capture_height=720, display_width=640, display_height=480, 
                          framerate=30, flip_method=0):
        """Create GStreamer pipeline for CSI camera"""
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){capture_width}, height=(int){capture_height}, "
            f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink"
        )
    
    def start_camera(self):
        """Start camera capture"""
        if self.camera is None:
            # Try CSI camera first
            logger.info("Attempting to open CSI camera...")
            self.camera = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
            
            if not self.camera.isOpened():
                logger.warning("CSI camera failed, trying USB camera...")
                self.camera = cv2.VideoCapture(0)
                
                if not self.camera.isOpened():
                    raise RuntimeError("No camera available!")
                    
        self.stop_camera = False
        self.camera_thread = threading.Thread(target=self.camera_capture_thread)
        self.camera_thread.start()
        logger.info("Camera started successfully")
        
    def camera_capture_thread(self):
        """Camera capture thread"""
        while not self.stop_camera:
            ret, frame = self.camera.read()
            if ret:
                # Drop old frames to maintain real-time performance
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)
                
    def stop_camera_capture(self):
        """Stop camera capture"""
        self.stop_camera = True
        if self.camera_thread:
            self.camera_thread.join()
        if self.camera:
            self.camera.release()
            self.camera = None
        logger.info("Camera stopped")
        
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
    
    def run_inference(self, display=True, save_output=False):
        """Run real-time face recognition"""
        logger.info("Starting face recognition inference...")
        
        # Start camera
        self.start_camera()
        
        # Video writer if saving
        writer = None
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
            logger.info(f"Recording to {output_path}")
        
        # Performance counters
        frame_count = 0
        start_time = time.time()
        inference_times = []
        
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                # Get frame
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
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
                inference_times.append(inference_time)
                frame_count += 1
                
                # Update FPS every second
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    self.fps = frame_count / elapsed
                    self.avg_inference_time = np.mean(inference_times[-30:]) * 1000
                    
                    # Reset counters
                    frame_count = 0
                    start_time = time.time()
                
                # Display performance info
                info_text = f"FPS: {self.fps:.1f} | Inference: {self.avg_inference_time:.1f}ms | Faces: {len(faces)}"
                cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save frame if recording
                if writer:
                    writer.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow("Face Recognition - Jetson Nano", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Cleanup
            self.stop_camera_capture()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            logger.info("Inference stopped")
    
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
        
        self.start_camera()
        
        metrics = {
            "frames": 0,
            "faces_detected": 0,
            "faces_recognized": 0,
            "inference_times": []
        }
        
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
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
                
        finally:
            self.stop_camera_capture()
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition on Jetson Nano")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration in seconds")
    parser.add_argument("--no-display", action="store_true", help="Run without display")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--fetch-model", action="store_true", help="Fetch latest model from edge")
    parser.add_argument("--edge-ip", default="192.168.50.1", help="Edge server IP")
    parser.add_argument("--edge-user", default="edgeuser", help="Edge server username")
    
    args = parser.parse_args()
    
    # Initialize system
    face_rec = JetsonFaceRecognition()
    
    # Fetch model if requested
    if args.fetch_model:
        face_rec.fetch_model_from_edge(args.edge_ip, args.edge_user)
    
    # Run benchmark or inference
    if args.benchmark:
        face_rec.benchmark(duration=args.duration)
    else:
        face_rec.run_inference(display=not args.no_display, save_output=args.save)


if __name__ == "__main__":
    main()