#!/usr/bin/env python3
"""
Enhanced client for dockerized Face Recognition services
"""
import requests
import argparse
import json
import time
import cv2
import os
from datetime import datetime

class FaceRecognitionClient:
    def __init__(self, api_url, model_url=None):
        self.api_url = api_url
        self.model_url = model_url or api_url.replace('5001', '5000')
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        self.output_dir = "client_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def check_services(self):
        """Check both services status"""
        print("=== Service Status Check ===")
        
        # Check inference server
        try:
            response = requests.get(f"{self.api_url}/ping", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Inference Server: {data['status']}")
                print(f"  Camera Method: {data.get('camera_method', 'Unknown')}")
                inference_ok = True
            else:
                print(f"✗ Inference Server: HTTP {response.status_code}")
                inference_ok = False
        except Exception as e:
            print(f"✗ Inference Server: {e}")
            inference_ok = False
        
        # Check model server
        try:
            response = requests.get(f"{self.model_url}/ping", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Model Server: {data['status']}")
                print(f"  Model Type: {data.get('classifier', 'Unknown')}")
                print(f"  Classes: {data.get('classes', 0)}")
                model_ok = True
            else:
                print(f"✗ Model Server: HTTP {response.status_code}")
                model_ok = False
        except Exception as e:
            print(f"✗ Model Server: {e}")
            model_ok = False
        
        return inference_ok and model_ok
            
    def get_api_info(self):
        """Get API information"""
        try:
            response = requests.get(self.api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("=== Inference API Information ===")
                print(f"Name: {data['name']}")
                print(f"Version: {data['version']}")
                print(f"Camera Method: {data.get('camera_method', 'Unknown')}")
                print("\nAvailable Endpoints:")
                for endpoint, desc in data['endpoints'].items():
                    print(f"  {endpoint} - {desc}")
        except Exception as e:
            print(f"Error getting API info: {e}")
        
        # Get model info
        try:
            response = requests.get(f"{self.model_url}/info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("\n=== Model Information ===")
                print(f"Model Type: {data['model_type']}")
                print(f"Classes: {data['num_classes']}")
                print(f"Trained: {data['trained']}")
                print(f"Confidence Threshold: {data['confidence_threshold']}")
                if data['classes']:
                    print("Known People:")
                    for person in data['classes']:
                        print(f"  - {person}")
        except Exception as e:
            print(f"Error getting model info: {e}")
            
    def capture_frame(self):
        """Capture a single frame"""
        print("Capturing frame...")
        try:
            response = requests.post(f"{self.api_url}/capture", timeout=30)
            print(response)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Capture successful!")
                print(f"  Method: {data.get('method_used', 'Unknown')}")
                print(f"  Faces detected: {len(data['faces'])}")
                
                # Save response to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = os.path.join(self.output_dir, f"capture_result_{timestamp}.json")
                with open(result_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                print(f"  Results saved: {result_file}")
                
                # Display face information
                if data['faces']:
                    print("\nFace Detection Results:")
                    for i, face in enumerate(data['faces']):
                        name = face['name']
                        confidence = face['confidence']
                        box = face['box']
                        print(f"  Face {i+1}: {name} ({confidence:.1f}%) at {box}")
                else:
                    print("  No faces detected")
                
                return data
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                print(f"✗ Capture failed: {error_detail}")
                return None
        except Exception as e:
            print(f"✗ Capture error: {e}")
            return None
            
    def start_stream(self):
        """Start camera streaming"""
        print("Starting camera stream...")
        try:
            response = requests.post(f"{self.api_url}/stream/start", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Stream started ({data.get('method', 'Unknown')})")
                return True
            else:
                print(f"✗ Failed to start stream: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Stream start error: {e}")
            return False
            
    def stop_stream(self):
        """Stop camera streaming"""
        print("Stopping camera stream...")
        try:
            response = requests.post(f"{self.api_url}/stream/stop", timeout=5)
            if response.status_code == 200:
                print("✓ Stream stopped")
                return True
            else:
                print(f"✗ Failed to stop stream: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Stream stop error: {e}")
            return False
            
    def monitor_stream(self, duration=30, interval=1.0, save_frames=True):
        """Monitor streaming status for a duration"""
        print(f"Monitoring stream for {duration} seconds...")
        
        # Start stream
        if not self.start_stream():
            return False
            
        try:
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration:
                # Get stream status
                try:
                    response = requests.get(f"{self.api_url}/stream/status", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        elapsed = time.time() - start_time
                        
                        print(f"\r[{elapsed:6.1f}s] FPS: {data['fps']:4.1f} | "
                              f"Inference: {data['inference_time']:5.1f}ms | "
                              f"Faces: {len(data['faces'])} | "
                              f"Method: {data.get('method_used', 'N/A')}", end="", flush=True)
                        
                        # Save frame and data when faces are detected
                        if save_frames and len(data['faces']) > 0 and frame_count % 5 == 0:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Get frame
                            try:
                                frame_response = requests.get(f"{self.api_url}/stream/frame", timeout=5)
                                if frame_response.status_code == 200:
                                    # Save frame
                                    frame_file = os.path.join(self.output_dir, f"stream_frame_{timestamp}.jpg")
                                    with open(frame_file, 'wb') as f:
                                        f.write(frame_response.content)
                                    
                                    # Save face data
                                    data_file = os.path.join(self.output_dir, f"stream_data_{timestamp}.json")
                                    with open(data_file, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    
                                    print(f"\n  → Saved: {frame_file}")
                                    
                            except Exception as e:
                                print(f"\n  ✗ Frame save error: {e}")
                        
                        frame_count += 1
                    else:
                        print(f"\r✗ Status error: HTTP {response.status_code}", end="", flush=True)
                        
                except Exception as e:
                    print(f"\r✗ Status request error: {e}", end="", flush=True)
                
                time.sleep(interval)
                
            print(f"\n✓ Stream monitoring completed ({frame_count} status checks)")
            
        except KeyboardInterrupt:
            print("\n⚠ Monitoring interrupted by user")
        finally:
            # Stop stream
            self.stop_stream()
        
        return True

    def test_model_direct(self, image_path=None):
        """Test model server directly with an image"""
        if image_path and os.path.exists(image_path):
            print(f"Testing model with image: {image_path}")
            
            # Read and encode image
            import base64
            with open(image_path, 'rb') as f:
                img_data = f.read()
                img_b64 = base64.b64encode(img_data).decode('utf-8')
            
            # Send to model
            data = {
                "instances": [
                    {"face_image": img_b64}
                ]
            }
            
            try:
                response = requests.post(f"{self.model_url}/invocations", 
                                       json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get('predictions', [])
                    if predictions:
                        pred = predictions[0]
                        print(f"✓ Model prediction:")
                        print(f"  Name: {pred.get('name', 'Unknown')}")
                        print(f"  Confidence: {pred.get('confidence', 0):.1f}%")
                        print(f"  Person ID: {pred.get('person_id', 'None')}")
                    else:
                        print("✗ No predictions returned")
                else:
                    print(f"✗ Model error: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ Model test error: {e}")
        else:
            print("No valid image path provided for model test")

    def run_full_test(self):
        """Run comprehensive test suite"""
        print("=== Face Recognition Full Test Suite ===")
        print(f"Inference API: {self.api_url}")
        print(f"Model API: {self.model_url}")
        print()
        
        # Test 1: Service status
        print("1. Checking service status...")
        if not self.check_services():
            print("⚠ Some services are not responding. Continuing anyway...")
        print()
        
        # Test 2: API info
        print("2. Getting API information...")
        self.get_api_info()
        print()
        
        # Test 3: Single capture
        print("3. Testing single frame capture...")
        result = self.capture_frame()
        print(result)
        print()
        
        # Test 4: Short streaming test
        print("4. Testing streaming (10 seconds)...")
        self.monitor_stream(duration=10, save_frames=True)
        print()
        
        print("=== Test Complete ===")
        print(f"Results saved in: {self.output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition API Client for Docker")
    parser.add_argument("--api-url", default="http://localhost:5001", 
                       help="Inference API URL (default: http://localhost:5001)")
    parser.add_argument("--model-url", default="http://localhost:5000",
                       help="Model API URL (default: http://localhost:5000)")
    parser.add_argument("--action", 
                       choices=["status", "info", "capture", "monitor", "test", "model-test"], 
                       default="status", help="Action to perform")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Duration for monitoring (seconds)")
    parser.add_argument("--image", help="Image path for model testing")
    
    args = parser.parse_args()
    
    client = FaceRecognitionClient(args.api_url, args.model_url)
    
    if args.action == "status":
        client.check_services()
    elif args.action == "info":
        client.get_api_info()
    elif args.action == "capture":
        client.capture_frame()
    elif args.action == "monitor":
        client.monitor_stream(duration=args.duration)
    elif args.action == "test":
        client.run_full_test()
    elif args.action == "model-test":
        client.test_model_direct(args.image)

if __name__ == "__main__":
    main()