#!/usr/bin/env python3.10
"""
Jetson Face Recognition API Client

This client script demonstrates how to interact with the Face Recognition API
running on the Jetson device via k3s.
"""
import requests
import argparse
import json
import time
import cv2
import os
from datetime import datetime

class FaceRecognitionClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        self.output_dir = "api_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def check_status(self):
        """Check API status"""
        try:
            response = requests.get(f"{self.api_url}/ping")
            if response.status_code == 200:
                data = response.json()
                print(f"API Status: {data['status']}")
                print(f"Model Service: {data['model_service']}")
                print(f"Timestamp: {data['timestamp']}")
                return True
            else:
                print(f"Error: API returned status code {response.status_code}")
                return False
        except Exception as e:
            print(f"Error connecting to API: {e}")
            return False
            
    def get_api_info(self):
        """Get API information"""
        try:
            response = requests.get(self.api_url)
            if response.status_code == 200:
                data = response.json()
                print("API Information:")
                print(f"  Name: {data['name']}")
                print(f"  Version: {data['version']}")
                print("\nAvailable Endpoints:")
                for endpoint, desc in data['endpoints'].items():
                    print(f"  {endpoint} - {desc}")
                return data
            else:
                print(f"Error: API returned status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error connecting to API: {e}")
            return None
            
    def capture_frame(self):
        """Capture a single frame"""
        print("Capturing frame...")
        try:
            response = requests.post(f"{self.api_url}/capture")
            if response.status_code == 200:
                data = response.json()
                print(f"Capture successful - {len(data['faces'])} faces detected")
                
                # Save response to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = os.path.join(self.output_dir, f"capture_result_{timestamp}.json")
                with open(result_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                print(f"Results saved to {result_file}")
                
                # Display face information
                for i, face in enumerate(data['faces']):
                    name = face['name']
                    confidence = face['confidence']
                    box = face['box']
                    print(f"Face {i+1}: {name} (Confidence: {confidence:.1f}%) at {box}")
                
                return data
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
            
    def start_stream(self):
        """Start camera streaming"""
        print("Starting camera stream...")
        try:
            response = requests.post(f"{self.api_url}/stream/start")
            if response.status_code == 200:
                print("Camera streaming started")
                return True
            else:
                print(f"Error: API returned status code {response.status_code}")
                return False
        except Exception as e:
            print(f"Error starting stream: {e}")
            return False
            
    def stop_stream(self):
        """Stop camera streaming"""
        print("Stopping camera stream...")
        try:
            response = requests.post(f"{self.api_url}/stream/stop")
            if response.status_code == 200:
                print("Camera streaming stopped")
                return True
            else:
                print(f"Error: API returned status code {response.status_code}")
                return False
        except Exception as e:
            print(f"Error stopping stream: {e}")
            return False
            
    def monitor_stream(self, duration=30, interval=1.0):
        """Monitor streaming status for a duration"""
        print(f"Monitoring stream for {duration} seconds...")
        
        # Start stream
        if not self.start_stream():
            return False
            
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                # Get stream status
                response = requests.get(f"{self.api_url}/stream/status")
                if response.status_code == 200:
                    data = response.json()
                    print(f"\rFPS: {data['fps']:.1f} | Inference: {data['inference_time']:.1f}ms | Faces: {len(data['faces'])}", end="")
                    
                    # Save frame periodically
                    if len(data['faces']) > 0 and int((time.time() - start_time) % 5) == 0:
                        # Get frame
                        frame_response = requests.get(f"{self.api_url}/stream/frame")
                        if frame_response.status_code == 200:
                            # Save frame
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            frame_file = os.path.join(self.output_dir, f"stream_frame_{timestamp}.jpg")
                            with open(frame_file, 'wb') as f:
                                f.write(frame_response.content)
                            print(f"\nSaved frame to {frame_file}")
                            
                            # Save face data
                            data_file = os.path.join(self.output_dir, f"stream_data_{timestamp}.json")
                            with open(data_file, 'w') as f:
                                json.dump(data, f, indent=2)
                
                time.sleep(interval)
                
            print("\nStream monitoring completed")
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
        finally:
            # Stop stream
            self.stop_stream()
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Jetson Face Recognition API Client")
    parser.add_argument("--api-url", default="http://localhost:30080", help="API URL")
    parser.add_argument("--action", choices=["status", "info", "capture", "monitor"], default="status", 
                      help="Action to perform")
    parser.add_argument("--duration", type=int, default=30, help="Duration for monitoring (seconds)")
    
    args = parser.parse_args()
    
    client = FaceRecognitionClient(args.api_url)
    
    if args.action == "status":
        client.check_status()
    elif args.action == "info":
        client.get_api_info()
    elif args.action == "capture":
        client.capture_frame()
    elif args.action == "monitor":
        client.monitor_stream(duration=args.duration)

if __name__ == "__main__":
    main()