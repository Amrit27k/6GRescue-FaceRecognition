#!/usr/bin/env python3
"""Face Recognition Inference Service for Jetson with FastAPI"""
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
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Jetson Face Recognition API", 
              description="API for face recognition on Jetson devices",
              version="1.0")

class FaceDetection(BaseModel):
    box: List[int]
    name: str
    confidence: float
    person_id: Optional[str] = None
    
class CaptureResponse(BaseModel):
    success: bool
    faces: List[FaceDetection]
    frame_path: str
    timestamp: str
    
class StatusResponse(BaseModel):
    streaming: bool
    fps: float
    inference_time: float
    faces: List[FaceDetection]
    timestamp: str
    
class SystemStatus(BaseModel):
    status: str
    model_service: str
    timestamp: str

class JetsonFaceRecognition:
    def __init__(self, model_service_url=None, temp_dir="temp_frames"):
        # Model service URL
        self.model_service_url = model_service_url or os.environ.get('MODEL_SERVICE_URL', 'http://face-recognition-model-service')
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
        
        # Frame dimensions (adjust based on your camera setup)
        self.frame_width = 1280
        self.frame_height = 720
        
        # Most recent processed frame and results for streaming
        self.last_frame = None
        self.last_results = []
        
        # For camera streaming
        self.camera_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.camera_thread = None
        
    def start_camera(self):
        """Start camera capture thread"""
        if self.camera_running:
            return
            
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self.camera_capture_thread)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        logger.info("Camera thread started")
        
    def stop_camera(self):
        """Stop camera capture thread"""
        self.camera_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        logger.info("Camera thread stopped")
        
    def camera_capture_thread(self):
        """Thread for continuous camera capture"""
        while self.camera_running:
            frame = self.capture_frame_gstreamer()
            if frame is not None:
                # Process frame
                faces, processed_frame = self.process_frame(frame)
                
                # Update last frame and results
                self.last_frame = processed_frame
                self.last_results = faces
                
                # Drop old frames to maintain real-time performance
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                        
                self.frame_queue.put((processed_frame, faces))
            
            # Sleep to control frame rate
            time.sleep(0.1)  # ~10 FPS
            
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
        """Detect faces in frame using Haar Cascade"""
        faces = []
        
        try:
            # Load detector if not already loaded
            if not hasattr(self, 'detector'):
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
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
        # Start timing
        start_time = time.time()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Initialize processed frame
        processed_frame = frame.copy()
        
        # Process each face
        for face in faces:
            x, y, w, h = face["box"]
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Recognize face using model service
            try:
                result = self.recognize_face(face_roi)
                face["name"] = result.get("name", "Unknown")
                face["confidence"] = result.get("confidence", 0)
                face["person_id"] = result.get("person_id", None)
            except Exception as e:
                logger.error(f"Error recognizing face: {e}")
                face["name"] = "Error"
                face["confidence"] = 0
                
            # Draw results on processed frame
            color = (0, 255, 0) if face["name"] != "Unknown" else (0, 0, 255)
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{face['name']}"
            if face["confidence"] > 0:
                label += f" ({face['confidence']:.0f}%)"
            
            cv2.putText(processed_frame, label, (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate timing
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Update FPS calculation
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.avg_inference_time = np.mean(self.inference_times[-30:]) * 1000
            
            # Reset counters
            self.frame_count = 0
            self.start_time = time.time()
        
        # Add FPS and timing info
        info_text = f"FPS: {self.fps:.1f} | Inference: {self.avg_inference_time:.1f}ms | Faces: {len(faces)}"
        cv2.putText(processed_frame, info_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return faces, processed_frame
        
    def recognize_face(self, face_roi):
        """Recognize face using model service"""
        try:
            # Convert image to base64
            _, img_encoded = cv2.imencode('.jpg', face_roi)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            # Prepare request
            data = {
                "instances": [
                    {"face_image": img_base64}
                ]
            }
            
            # Send to model service
            import requests
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
face_recognition = JetsonFaceRecognition(model_service_url="http://192.168.2.100:5000")

# FastAPI routes
@app.get("/")
async def root():
    """API root - provides basic info"""
    return {
        "name": "Jetson Face Recognition API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This info",
            "POST /capture": "Capture and process a single frame",
            "POST /stream/start": "Start camera streaming",
            "POST /stream/stop": "Stop camera streaming",
            "GET /stream/frame": "Get the latest frame as JPEG",
            "GET /stream/status": "Get streaming status and latest results",
            "GET /ping": "Health check"
        }
    }

@app.post("/capture", response_model=CaptureResponse)
async def capture():
    """Capture and process a single frame"""
    try:
        # Capture frame
        frame = face_recognition.capture_frame_gstreamer()
        
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame")
            
        # Process frame
        faces, processed_frame = face_recognition.process_frame(frame)
        
        # Save processed frame
        output_path = os.path.join(face_recognition.output_dir, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(output_path, processed_frame)
        
        return {
            "success": True,
            "faces": faces,
            "frame_path": output_path,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing capture: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream/start")
async def start_stream():
    """Start camera streaming"""
    face_recognition.start_camera()
    return {"success": True, "message": "Camera streaming started"}

@app.post("/stream/stop")
async def stop_stream():
    """Stop camera streaming"""
    face_recognition.stop_camera()
    return {"success": True, "message": "Camera streaming stopped"}

@app.get("/stream/frame")
async def get_stream_frame():
    """Get the latest frame as JPEG image"""
    try:
        if face_recognition.last_frame is None:
            # Return a blank frame
            blank = np.zeros((face_recognition.frame_height, face_recognition.frame_width, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank)
            return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
            
        # Encode the last processed frame
        _, buffer = cv2.imencode('.jpg', face_recognition.last_frame)
        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error getting stream frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/status", response_model=StatusResponse)
async def get_stream_status():
    """Get streaming status and latest face detection results"""
    try:
        return {
            "streaming": face_recognition.camera_running,
            "fps": face_recognition.fps,
            "inference_time": face_recognition.avg_inference_time,
            "faces": face_recognition.last_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping", response_model=SystemStatus)
async def ping():
    """Health check"""
    try:
        # Try to ping model service
        import requests
        try:
            response = requests.get(f"{face_recognition.model_service_url}/ping", timeout=2)
            model_status = "OK" if response.status_code == 200 else "ERROR"
        except:
            model_status = "UNREACHABLE"
            
        return {
            "status": "OK",
            "model_service": model_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in ping: {e}")
        return {
            "status": "ERROR",
            "model_service": "ERROR", 
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)