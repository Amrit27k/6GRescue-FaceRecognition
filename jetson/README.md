# Face Recognition Docker Setup with RTSP Streaming

A containerized face recognition inference service for NVIDIA Jetson devices with CSI/USB camera support and real-time RTSP streaming of processed frames.

## Features

- **Multi-camera support**: CSI, USB, and OpenCV camera detection
- **NVIDIA runtime**: Optimized for Jetson hardware with GPU acceleration
- **RTSP streaming**: Real-time H.264 stream with face detection overlays
- **REST API**: Flask-based inference service
- **Face detection**: Haar cascade-based face detection with ML model integration

## Prerequisites

- NVIDIA Jetson device (Nano/Orin)
- Docker with NVIDIA runtime support
- CSI camera connected (or USB camera)

## Quick Start

### 1. Setting the Service(Without and With Edge Setup)

#### a. Without Edge Setup

- **Clone the Repository**:
  ```bash
  git clone https://github.com/Amrit27k/6GRescue-FaceRecognition.git
  ```
- **Inside the repository, enter inside 'jetson' directory**
  ```bash
  cd 6GRescue-FaceRecognition
  cd jetson
  ```
- **Create a venv and activate (Optional)**
  ```bash
  python3 -m venv <env-name>
  source <env-name>/bin/activate
  ```
#### b. With Edge Setup

- **Use ansible playbook for jetson to setup the environment in jetson**
Follow the setup instructions provided in the repository - 6GRescueServices[https://github.com/Amrit27k/6GRescueServices/blob/main/README.md]
This will setup the edge environment with jupyterhub and clone the repository of the scripts.

- **Use MLFlow plugin to deploy the model files and scripts to jetson**
  ```bash
  cd mlflow_plugin_examples
  python simple_file_transfer.py --iot_ip 192.168.2.100 --model rf
  ```

### 2. Build the Docker Images
  ```bash
  # Build model server
  cd mlflow_deployments_v<version-number>/face_recognition_files

  docker build -f docker/Dockerfile.model-server \
      -t face-model-mlf-plugin:latest .

  # Build inference server with RTSP support
  docker build -f docker/Dockerfile.inference-server-rtsp \
      -t face-inference-mlf-plugin-rtsp:latest .
  ```

### 2. Create Network (Optional)
  ```bash
  docker network create face-net-mlf-plugin
  ```

### 3. Run the Containers

```bash
# Run model server
docker run -d \
    --name face-model-mlf-plugin \
    --network face-net-mlf-plugin \  #Optional
    -p 5000:5000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    --memory=512m \
    --restart unless-stopped \
    face-model-mlf-plugin:latest

# Run inference server with RTSP support
docker run -d \
    --name face-inference-mlf-plugin-rtsp \:
    --runtime=nvidia \
    -p 5001:5001 \
    -p 8554:8554 \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/temp_frames:/app/temp_frames \
    --volume /tmp/argus_socket:/tmp/argus_socket \
    --volume /usr/src/tensorrt/data:/usr/src/tensorrt/data:ro \
    --device /dev/nvhost-ctrl \
    --device /dev/nvhost-ctrl-gpu \
    --device /dev/nvhost-gpu \
    --device /dev/nvhost-as-gpu \
    --device /dev/nvhost-vic \
    --device /dev/nvhost-msenc \
    --device /dev/nvmap \
    -v /dev:/dev \
    --device /dev/video0 \
    --privileged \
    -e MODEL_SERVICE_URL=http://localhost:5000 \
    -e RTSP_AUTO_START=true \
    face-inference-mlf-plugin-rtsp:latest
```

### 3.a Run the inference script manually inside venv (optional)
```bash
source <venv-name>/bin/activate
pip install --no-cache-dir MarkupSafe paho-mqtt torch torchvision numpy requests Flask opencv-python-headless ultralytics
cd scripts/
gcc rtsp_server_fps.c -o rtsp_server_fps $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 glib-2.0)
python3 inference_server_v1.2.py
```

### 4. Start Face Recognition and Streaming

```bash
# Start camera streaming and processing
curl -X POST http://localhost:5001/start_streaming

# Start RTSP server for processed frames
curl -X POST http://localhost:5001/start_rtsp

# Check system status
curl http://localhost:5001/status
```

### 5. View the Streams

**RTSP Stream without Face Detection Overlays (Recommended):**
```bash
# Using ffplay
ffplay rtsp://localhost:8554/test

# From remote machine
ffplay rtsp://192.168.1.100:8554/test
```

**HTTP Stream in Web Browser:**
```
http://localhost:5001/video_player
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/ping` | GET | Health check for service and model connectivity |
| `/capture` | POST | Capture and process a single frame with face detection |
| `/start_streaming` | POST | Start continuous camera streaming and processing |
| `/stop_streaming` | POST | Stop camera streaming and processing |
| `/start_rtsp` | POST | Start RTSP server for processed frames |
| `/stop_rtsp` | POST | Stop RTSP server |
| `/status` | GET | Get current system status and performance metrics |
| `/video_feed` | GET | MJPEG stream of processed frames |
| `/video_player` | GET | Web interface for viewing processed stream |

## Client Usage

The enhanced Python client provides comprehensive control:

```bash
# Setup complete system (streaming + RTSP)
python3 client.py --host localhost setup

# Check service status
python3 client.py --host localhost info

# Monitor real-time performance
python3 client.py --host localhost monitor

# View RTSP stream
python3 client.py --host localhost view

# Control streaming
python3 client.py --host localhost start-streaming
python3 client.py --host localhost start-rtsp
python3 client.py --host localhost stop-rtsp

# Capture single frame
python3 client.py --host localhost capture

# For remote Jetson
python3 client.py --host 192.168.1.100 setup
```

## Streaming Options

### RTSP Stream (Recommended)
- **High performance**: Hardware-accelerated H.264 encoding
- **Low latency**: Optimized for real-time viewing
- **Universal compatibility**: Works with VLC, ffplay, etc.
- **Face detection overlays**: Green rectangles around detected faces
- **URL**: `rtsp://your-jetson-ip:8554/test`

### HTTP MJPEG Stream
- **Web browser compatible**: View directly in browser
- **Interactive interface**: Built-in controls and status
- **Lower performance**: Higher latency than RTSP
- **URL**: `http://your-jetson-ip:5001/video_player`

## Camera Support

The service automatically detects the best available camera method:

1. **CSI Camera** (preferred): Uses GStreamer with nvarguscamerasrc
2. **USB Camera**: Uses GStreamer with v4l2src  
3. **OpenCV**: Fallback method using OpenCV VideoCapture

## Configuration

### Environment Variables

- `MODEL_SERVICE_URL`: URL of the face recognition model service (default: http://localhost:5000)
- `RTSP_PORT`: RTSP server port (default: 8554)
- `RTSP_AUTO_START`: Auto-start RTSP server on container startup (default: true)

### Port Mappings

- `5001`: HTTP API and MJPEG stream
- `8554`: RTSP stream with face detection overlays

### Volume Mounts

- `./output`: Processed images with face detection results
- `./temp_frames`: Temporary frame storage during processing
- `/tmp/argus_socket`: Required for CSI camera communication
- `/usr/src/tensorrt/data`: TensorRT libraries for GPU acceleration

### Required Devices

The container needs access to NVIDIA GPU and camera devices:
- `/dev/nvhost-*`: NVIDIA GPU control devices
- `/dev/nvmap`: NVIDIA memory mapping
- `/dev/video0`: Camera device

## Performance Monitoring

Real-time metrics available via `/status` endpoint:
- **FPS**: Frames processed per second
- **Inference Time**: Face detection and recognition latency
- **Face Count**: Number of faces detected in current frame
- **Queue Status**: Processing pipeline health
- **Camera Method**: Active camera detection method
- **RTSP Status**: Streaming server status

## Troubleshooting

### Camera Resource Conflicts
If you get camera access errors:
```bash
# Ensure only one process accesses camera
curl -X POST http://localhost:5001/stop_rtsp
curl -X POST http://localhost:5001/start_streaming
```

### RTSP Stream Issues
Check RTSP server status:
```bash
# Verify RTSP is running
curl http://localhost:5001/status

# Restart RTSP server
curl -X POST http://localhost:5001/stop_rtsp
curl -X POST http://localhost:5001/start_rtsp

# Test with different players
ffplay -fflags nobuffer -flags low_delay rtsp://localhost:8554/test
vlc --network-caching=0 rtsp://localhost:8554/test
```

### Green Image Issue
If you see green frames instead of camera feed:
- Ensure NVIDIA runtime is properly configured
- Check that `/tmp/argus_socket` is mounted
- Verify all NVIDIA devices are accessible

### Container Health Issues
Check container logs:
```bash
docker logs face-inference-mlf-plugin-rtsp
docker inspect face-inference-mlf-plugin-rtsp | grep -A 10 Health
```

### Model Service Connection
Ensure your model service is running on port 5000:
```bash
curl http://localhost:5000/ping
```

## Advanced Usage

### Remote Access
Access from another machine on the network:
```bash
# Setup from remote machine
python3 client.py --host 192.168.1.100 setup

# View stream from remote machine
ffplay rtsp://192.168.1.100:8554/test
```

### SSH Tunneling
For secure remote access:
```bash
# On local machine
ssh -L 5001:localhost:5001 -L 8554:localhost:8554 user@jetson-ip

# Then access locally
http://localhost:5001/video_player
ffplay rtsp://localhost:8554/test
```

## File Structure

```
face_recognition_files/
├── docker/
│   ├── Dockerfile.model-server
│   └── Dockerfile.inference-server-rtsp
└── scripts/
    ├── inference_server_rtsp.py
    ├── client.py
    └── test-launch.c
```

## Example Output

**Status Response:**
```json
{
  "camera_method": "csi",
  "camera_running": true,
  "processing_running": true,
  "rtsp_running": true,
  "rtsp_url": "rtsp://localhost:8554/test",
  "fps": 28.5,
  "avg_inference_time": 45.2,
  "last_results": [
    {
      "box": [120, 80, 150, 150],
      "name": "Alice",
      "confidence": 87,
      "person_id": 123
    }
  ]
}
```