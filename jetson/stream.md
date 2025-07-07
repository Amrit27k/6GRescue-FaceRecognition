# Video Streaming from Jetson to Edge Server

Simple guide to stream face recognition video from Jetson Nano to your edge server.

## Quick Setup

### 1. Go inside the mlflow_deployment package deployed by mlflow to jetson.
```bash
cd mlflow_deployment
```

### 2. Build & Run Container
```bash
# Build the image
docker build -f face_recognition_files/docker/Dockerfile.inference-server-stream \
    -t face-inference-mlf-plugin-stream:latest .

# Run with streaming ports
docker run -d \
    --name face-inference-mlf-plugin-stream \
    --runtime=nvidia \
    --network host \
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
    face-inference-mlf-plugin-stream:latest
```

### 3. Start Streaming
```bash
# Start the camera stream
curl -X POST http://localhost:5001/stream/start

# Check if it's running
curl http://localhost:5001/stream/status
```
### FFmpeg (Create RTSP from HTTP) in Edge:(optional)
```bash
ffmpeg -re -f mjpeg -i http://192.168.2.100:5001/stream/mjpeg \
    -c:v libx264 -preset ultrafast \
    -f rtsp rtsp://localhost:8554/stream

## Access Video Stream

### From Edge Server (Replace `192.168.2.100` with your Jetson IP):

#### Option 1: VLC Player (Easiest)
```bash
vlc http://192.168.2.100:5001/stream/mjpeg
```

#### Option 2: Web Browser
Open in browser: `http://192.168.2.100:5001/stream/mjpeg`

#### Option 3: Single Frame (for testing)
```bash
curl http://192.168.2.100:5001/stream/frame > test_frame.jpg
```

#### Option 4: Python Script
```python
import requests
import cv2
import numpy as np

# Simple viewer
response = requests.get("http://192.168.2.100:5001/stream/mjpeg", stream=True)
for chunk in response.iter_content(chunk_size=1024):
    # Process video stream
    pass
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /stream/start` | Start camera and streaming |
| `POST /stream/stop` | Stop streaming |
| `GET /stream/status` | Get FPS, face count, etc. |
| `GET /stream/frame` | Get single JPEG frame |
| `GET /stream/mjpeg` | **Live video stream** |

## Testing Commands

```bash
# Health check
curl http://localhost:5001/ping

# Start streaming
curl -X POST http://localhost:5001/stream/start

# Get stream info
curl http://localhost:5001/stream/status

# Test single frame
curl http://localhost:5001/stream/frame > frame.jpg

# Test video stream (should show continuous data)
curl http://localhost:5001/stream/mjpeg

# Stop streaming
curl -X POST http://localhost:5001/stream/stop
```

## Troubleshooting

### No Video Stream?
1. Check if streaming started: `curl http://localhost:5001/stream/status`
2. Verify camera working: `curl -X POST http://localhost:5001/capture`
3. Check container logs: `docker logs face-inference-mlf-plugin-host`

### Can't Connect from Edge Server?
1. Replace `192.168.2.100` with actual Jetson IP
2. Test connectivity: `ping 192.168.2.100`
3. Check if port is open: `telnet 192.168.2.100 5001`

### VLC Connection Failed?
1. Try browser first: `http://JETSON_IP:5001/stream/mjpeg`
2. Check firewall settings
3. Use single frame test: `curl http://JETSON_IP:5001/stream/frame`

## What You Get

- **Live video stream** with face detection boxes
- **Real-time metrics** (FPS, inference time)
- **Face recognition results** overlaid on video
- **HTTP-based streaming** (no complex RTSP setup needed)
- **Works with VLC, browsers, Python, FFmpeg**

## Network Setup

Make sure your edge server can reach the Jetson:
- Jetson IP: `192.168.2.100` (example)
- Streaming Port: `5001`
- URL: `http://192.168.2.100:5001/stream/mjpeg`

The stream includes live face detection and recognition with performance metrics displayed on each frame!