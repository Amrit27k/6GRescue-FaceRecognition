# Object Recognition Docker Setup with RTSP Streaming

A containerized object recognition inference service for NVIDIA Jetson devices with CSI/USB camera support, a live RTSP camera stream and an annotated MJPEG stream of the recognition results.

> **Naming note:** the repository and the container/image names (`face-model-mlf-plugin`, `face-inference-mlf-plugin-rtsp`) are kept from the original face recognition version so existing deployment tooling keeps working. The pipeline recognizes objects, not faces.

## How It Works

```
USB/CSI camera
   |  rtsp_server_fps (GStreamer, H.264)
   v
rtsp://<JETSON_IP>:8554/test              raw camera stream, no overlays
   |  read with OpenCV
   v
Inference service (Jetson, port 5001)
   YOLOv8n boxes + centre crops -> MobileNetV3-small embeddings (576 values)
   |  POST /invocations, one batched request per frame
   v
Model server (Docker, port 5000)
   RandomForest trained offline, loaded from artifacts, never trained at startup
   |
   v
Annotated stream (/video_feed), MQTT topic, results file
```

## Features

- **Multi-camera support**: CSI and USB cameras, probed at startup (CSI first, then USB). Override with `CAMERA_METHOD=csi|usb`
- **NVIDIA runtime**: YOLO and MobileNetV3 run on the GPU when PyTorch sees CUDA
- **RTSP streaming**: real-time H.264 stream of the raw camera (no overlays)
- **Annotated MJPEG stream**: boxes with class name, confidence and crop source, at `/video_feed` and `/video_player`
- **Two-stage recognition**: candidate crops (YOLOv8n boxes plus centre crops) are embedded on the Jetson and classified by a separate model server
- **REST API**: Flask-based inference service
- **MQTT**: results are published for every processed frame

## Prerequisites

- NVIDIA Jetson device (Nano/Orin)
- Docker with NVIDIA runtime support
- CSI camera connected (or USB camera)
- For the venv option (4.a): PyTorch and torchvision built for your JetPack version, GStreamer development packages, internet access on the first run (downloads `yolov8n.pt` and the MobileNetV3 weights)
- An MQTT broker (optional). Without one the service keeps retrying in the background and recognition still works

## Quick Start

### 1. Setting the Service(Without and With Edge Setup)

#### a. Without Edge Setup

- **Clone the Repository**:
  ```bash
  git clone https://github.com/Amrit27k/6GRescue-FaceRecognition.git
  ```
- **Inside the repository**
  ```bash
  cd 6GRescue-FaceRecognition
  ```
- **Create a venv and activate (Optional)**
  ```bash
  python3 -m venv <env-name>
  source <env-name>/bin/activate
  ```
#### b. With Edge Setup

- **Use ansible playbook for jetson to setup the environment in jetson**
Follow the setup instructions provided in the repository - [6GRescueServices](https://github.com/Amrit27k/6GRescueServices/blob/main/README.md)
This will setup the edge environment with jupyterhub and clone the repository of the scripts.

- **Use MLFlow plugin to deploy the model files and scripts to jetson**
  ```bash
  cd 6GRescue-FaceRecognition/mlflow_plugin_examples
  python simple_file_transfer.py --iot_ip 192.168.2.100 --model rf
  ```

### 2. Build the Docker Images in IOT(Jetson)
  ```bash
  # Build model server
  cd mlflow_deployments_v<version-number>/face_recognition_files

  docker build -f docker/Dockerfile.model-server \
      -t face-model-mlf-plugin:latest .

  # Build inference server with RTSP support
  docker build -f docker/Dockerfile.inference-server-rtsp \
      -t face-inference-mlf-plugin-rtsp:latest .
  ```

### 3. Create Network (Optional)
  ```bash
  docker network create face-net-mlf-plugin
  ```

### 4. Run the Containers

The model server is needed for both ways of running the inference service. Start it first.

```bash
# Run model server
# Optional: to use the network from step 3, add "--network face-net-mlf-plugin \" after the --name line
docker run -d \
    --name face-model-mlf-plugin \
    -p 5000:5000 \
    -e ARTIFACT_DIR=/app/models \
    -e CONFIDENCE_THRESHOLD=0.4 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    --memory=512m \
    --restart unless-stopped \
    face-model-mlf-plugin:latest
```

`models/` must contain the trained artifacts (see [Model Server](#model-server)). If one is missing the container exits at startup instead of serving an empty model. Check with:

```bash
docker logs face-model-mlf-plugin | head -20
curl http://localhost:5000/info
```

Then run the inference service **either** in Docker (below) **or** as a script in a venv (4.a). Do not run both: they use the same camera and the same ports (5001, 8554).

```bash
# Run inference server with RTSP support
docker run -d \
    --name face-inference-mlf-plugin-rtsp \
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

### 4.a Run the inference script manually inside venv (optional)

The script runs on the host, so `http://localhost:5000` reaches the model server through its published port.

```bash
# System packages (once)
sudo apt update
sudo apt install -y build-essential pkg-config \
    gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev libgstrtspserver-1.0-dev

# Python environment
source <venv-name>/bin/activate

# Install PyTorch and torchvision first, using NVIDIA's wheels that match your JetPack version
# (PyPI wheels are generally CPU-only on Jetson), then check CUDA is visible:
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

pip install --no-cache-dir MarkupSafe paho-mqtt numpy requests Flask opencv-python-headless ultralytics

# Build the RTSP publisher (once). The script starts ./rtsp_server_fps from the current directory
cd scripts/
gcc rtsp_server_fps.c -o rtsp_server_fps $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 glib-2.0)

# Run (the model server from step 4 must already be up)
python3 jetson_object_inference.py
```

Set the broker and anything else through environment variables (see [Configuration](#configuration)):

```bash
MQTT_BROKER_HOST=<broker-ip> python3 jetson_object_inference.py
```

The script starts the RTSP server and the recognition loop on launch. Healthy startup logs include `Models warmed up`, `Camera method for RTSP server: ...`, `RTSP server up at rtsp://localhost:8554/test` and `Connected to MQTT broker ...`. Stop it with Ctrl+C.

Quick checks from another terminal:

```bash
curl http://localhost:5001/ping      # model_service must be "OK"
curl http://localhost:5001/ready     # 200 only when frames are flowing and results are fresh
curl http://localhost:5001/status    # stream_connected, metrics, last_objects
```

### 5. Start Recognition and Streaming

The venv script starts streaming and the RTSP server by itself (`RTSP_AUTO_START=true`). These commands are for controlling or restarting them:

```bash
# Start camera streaming and processing
curl -X POST http://localhost:5001/start_streaming

# Start the RTSP server (raw camera stream)
curl -X POST http://localhost:5001/start_rtsp

# Check system status
curl http://localhost:5001/status
```

### 6. View the Streams

**RTSP stream, raw camera with no overlays (low latency):**
```bash
# Using ffplay
ffplay -fflags nobuffer -flags low_delay rtsp://localhost:8554/test

# From remote machine
ffplay -fflags nobuffer -flags low_delay rtsp://<JETSON_IP>:8554/test
```

**HTTP stream in web browser, annotated with recognition results:**
```
http://localhost:5001/video_player
```

Boxes are labelled `name confidence% [source]`, for example `dove-cream 52% [center50]`. The source tells you which crop produced the match (`yolo:<coco class>`, `center80` or `center50`).

Find the Jetson IP on the WiFi network with `hostname -I` on the Jetson (or try `<jetson-hostname>.local`), then replace `<JETSON_IP>` in the ffplay and browser URLs.

## Model Server

The model server only **serves** a model that was trained offline. It never trains at startup.

`models/` (mounted at `/app/models`, set with `ARTIFACT_DIR`) must contain:

```
models/
  random_forest_model.pkl     joblib file from the training script
  label_encoder.pkl           pickle with label_encoder and reverse_label_encoder
  face_database.json          id -> class name
  model_params.json           optional, {"confidence_threshold": 0.4}
```

- Build requirements: `scikit-learn==1.7.0` (the version the pickles were created with, a different version can load them incorrectly), `joblib`, `numpy`, `Flask` and `gunicorn`.
- A prediction is `Unknown` when the best class is not `Unknown` but its probability is below the confidence threshold. Threshold order of precedence: `CONFIDENCE_THRESHOLD` env var, then `model_params.json`, then 0.2. Start around 0.4 and tune it from the logs.
- Every prediction is logged with its top 3 classes, for example `pred=dove-cream conf=0.519 top3=[...]`. Use `docker logs -f face-model-mlf-plugin` while testing.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Liveness and number of classes |
| `/health` | GET | Model loaded, feature size and threshold |
| `/info` | GET | Class list, number of trees, feature size, threshold, artifact directory |
| `/invocations` | POST | Classify feature vectors |

Request and response (confidence is a percentage):

```json
{"instances": [{"image_feature_vector": [0.01, 0.52, "... 576 numbers"]}]}
```
```json
{"predictions": [{"name": "dove-cream", "confidence": 51.9, "person_id": "6"}]}
```

## Updating the Model

Crops from the live camera look different from phone photos, so a model trained only on photos tends to return `Unknown` or flat scores on the stream. Capture frames from the same camera, then retrain:

1. **Capture frames per class** with the venv script. It saves a raw frame every `CAPTURE_EVERY_S` seconds, up to `CAPTURE_MAX`, to `captured/<label>/`. Move the object around, closer and farther, at different angles:
   ```bash
   CAPTURE_LABEL=dove-cream python3 jetson_object_inference.py
   ```
2. **Capture background frames** the same way with `CAPTURE_LABEL=Unknown`: empty desk, wall, floor, people walking past.
3. **Retrain** with the training script. Copy the captured folders into `images/<class>/` and include `Unknown` in `class_names` so the background frames are used as the `Unknown` class.
4. **Deploy**: copy `random_forest_model.pkl`, `label_encoder.pkl` and `face_database.json` into `models/` and restart:
   ```bash
   docker restart face-model-mlf-plugin
   curl http://localhost:5000/info      # check the class list
   ```

## API Endpoints

Inference service (port 5001):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/ping` | GET | Health check: model service reachability, stream state, frame and result age, MQTT state |
| `/ready` | GET | 200 only when the stream is connected and results are fresh, otherwise 503 (venv script) |
| `/capture` | POST | Process the most recent frame once, save and publish the result (503 until the first frame arrives) |
| `/start_streaming` | POST | Start continuous camera streaming and processing |
| `/stop_streaming` | POST | Stop camera streaming and processing |
| `/start_rtsp` | POST | Start the RTSP server (raw camera stream) |
| `/stop_rtsp` | POST | Stop RTSP server |
| `/status` | GET | Get current system status and performance metrics |
| `/video_feed` | GET | MJPEG stream of annotated frames |
| `/video_player` | GET | Web interface for viewing the annotated stream |

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
python3 client.py --host <JETSON_IP> setup
```

## Streaming Options

### RTSP Stream (raw camera)
- **High performance**: Hardware-accelerated H.264 encoding on CSI cameras (USB cameras use software x264)
- **Low latency**: Optimized for real-time viewing
- **Universal compatibility**: Works with VLC, ffplay, etc.
- **No overlays**: this is the unprocessed camera feed
- **URL**: `rtsp://<JETSON_IP>:8554/test`

### HTTP MJPEG Stream (annotated)
- **Recognition overlays**: green boxes with class, confidence and crop source
- **Web browser compatible**: View directly in browser
- **Interactive interface**: Built-in controls and a status line with timings
- **Lower performance**: Higher latency than RTSP
- **URL**: `http://<JETSON_IP>:5001/video_player`

## Camera Support

The service detects the camera method for the RTSP server at startup:

1. **CSI Camera** (preferred): GStreamer with `nvarguscamerasrc` and the hardware encoder `nvv4l2h264enc`
2. **USB Camera**: GStreamer with `v4l2src` on `/dev/video0` and software `x264enc`

If neither test passes it defaults to CSI, which will fail without a CSI camera. Skip the probe with `CAMERA_METHOD=csi` or `CAMERA_METHOD=usb`.

For USB webcams, `/dev/video0` is the capture node and `/dev/video1` is usually a metadata node, so only `/dev/video0` is used. `nvarguscamerasrc` works with CSI cameras only.

## Configuration

### Environment Variables

Inference service:

- `MODEL_SERVICE_URL`: URL of the model service (default: http://localhost:5000)
- `RTSP_PORT`: RTSP server port (default: 8554)
- `RTSP_AUTO_START`: Auto-start RTSP server on startup (default: true)
- `CAMERA_METHOD`: `csi` or `usb`, skips the startup camera probe (default: auto-detect)
- `YOLO_CONF`: YOLO confidence for candidate boxes (default: 0.25)
- `MQTT_BROKER_HOST`: MQTT broker address (default: 10.70.0.64, set this for your network)
- `MQTT_PORT`: MQTT broker port (default: 1883)
- `MQTT_TOPIC`: topic for results (default: jetson/face_recognition/results)
- `CAPTURE_LABEL`: if set, saves raw frames to `captured/<label>/` for retraining (default: off)
- `CAPTURE_EVERY_S`: seconds between captured frames (default: 1.5)
- `CAPTURE_MAX`: maximum captured frames per run (default: 40)

Model server:

- `ARTIFACT_DIR`: directory with the trained artifacts (default: /app/artifacts, use /app/models with the mount above)
- `CONFIDENCE_THRESHOLD`: minimum probability for a known class (default: from `model_params.json`, else 0.2)

### Port Mappings

- `5000`: model server API
- `5001`: HTTP API and annotated MJPEG stream
- `8554`: RTSP stream of the raw camera

### Volume Mounts

- `./models`: trained model artifacts for the model server
- `./logs`: model server logs
- `./output`: recognition results (the venv script writes `recognition_results.jsonl`)
- `./captured`: frames saved in capture mode (venv script)
- `./temp_frames`: Temporary frame storage during processing
- `/tmp/argus_socket`: Required for CSI camera communication
- `/usr/src/tensorrt/data`: TensorRT libraries for GPU acceleration

### Required Devices

The container needs access to NVIDIA GPU and camera devices:
- `/dev/nvhost-*`: NVIDIA GPU control devices
- `/dev/nvmap`: NVIDIA memory mapping
- `/dev/video0`: Camera device

## Outputs

Each processed frame is published to the MQTT topic (an empty `detected_objects` list when nothing was recognized). The same payload is appended as one JSON line to `output/recognition_results.jsonl` when something was recognized, and for every manual `/capture`.

Only recognized classes are reported, with one box per class (the most confident candidate).

```json
{
  "timestamp": "2026-10-12T12:00:00.000000",
  "source_method": "continuous",
  "frame_dimensions": {"width": 1280, "height": 720},
  "detected_objects": [
    {
      "box": [320, 180, 640, 360],
      "source": "center50",
      "name": "dove-cream",
      "confidence": 51.9,
      "object_id": "6"
    }
  ],
  "metrics": {"processing_fps": "<fps>", "detect_ms": "<ms>", "embed_ms": "<ms>", "serve_ms": "<ms>", "total_ms": "<ms>"}
}
```

The field values above are illustrative. Compared with the older face recognition payload, `detected_faces` is now `detected_objects` and `person_id` is now `object_id`. Update any MQTT subscriber that reads the old keys.

## Performance Monitoring

Real-time metrics available via `/status` and the `/video_player` status line:
- **processing_fps**: Frames processed per second
- **detect_ms, embed_ms, serve_ms, total_ms**: time per frame for YOLO plus cropping, MobileNet embedding, the model server call, and the total
- **last_objects**: Objects recognized in the current frame
- **stream_connected, last_frame_age_s, last_result_age_s**: Processing pipeline health
- **camera_method_for_rtsp_server**: Active camera detection method
- **rtsp_server_running, mqtt_connected**: Streaming server and broker status

## Troubleshooting

### Camera Resource Conflicts
If you get camera access errors:
```bash
# Ensure only one process accesses camera
curl -X POST http://localhost:5001/stop_rtsp
curl -X POST http://localhost:5001/start_streaming
```

Also make sure the inference container and the venv script are not both running, and that nothing else holds the camera:
```bash
docker ps | grep face-inference
fuser -v /dev/video0
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

`method DESCRIBE failed: 503` in the logs means an RTSP server is listening but nothing is publishing to `/test`. Check that `rtsp_server_fps` is running (look for `RTSP server:` lines in the log), that the camera works (`ls /dev/video*`), and that `stream_connected` in `/status` becomes `true`.

### Everything Is Unknown or Scores Are Low
Watch the model server while the object is in front of the camera:
```bash
docker logs -f face-model-mlf-plugin
```
- Flat scores (for example 0.18, 0.18, 0.17 across classes) mean the crops look unlike the training data. Capture frames from the live camera and retrain (see [Updating the Model](#updating-the-model)).
- Scores around 0.3 to 0.5 with a clear gap to the second class are real matches. Tune `CONFIDENCE_THRESHOLD` to fit.
- Keep the object near the middle of the frame and filling a good part of it, since the centre crops are what match it.

### Slow Processing
Check the timings in `/status`. A high `detect_ms` usually means YOLO is running on the CPU:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
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

If the model container exits right after starting, `docker logs face-model-mlf-plugin` names the missing artifact. Check that the three files are in `models/` and that `ARTIFACT_DIR` matches the mount.

If `docker run` fails with `The container name "/face-model-mlf-plugin" is already in use`, remove the old container first:
```bash
docker rm -f face-model-mlf-plugin
```

A scikit-learn `InconsistentVersionWarning` at startup means the image uses a different version than the training environment. Pin `scikit-learn==1.7.0`.

### Cannot Open the Stream From Another Machine
- Use the address from `hostname -I` on the Jetson, and make sure both machines are on the same network
- Home routers can reassign addresses after a reboot, so reserve the address in the router or use `<jetson-hostname>.local`
- Guest and campus WiFi often isolate clients from each other, which blocks the connection
- Check the service listens on all interfaces: `ss -ltnp | grep 5001`

## Advanced Usage

### Remote Access
Access from another machine on the network:
```bash
# Setup from remote machine
python3 client.py --host <JETSON_IP> setup

# View stream from remote machine
ffplay rtsp://<JETSON_IP>:8554/test
```

### SSH Tunneling
For secure remote access:
```bash
# On local machine
ssh -L 5001:localhost:5001 -L 8554:localhost:8554 user@<JETSON_IP>

# Then access locally
http://localhost:5001/video_player
ffplay rtsp://localhost:8554/test
```

## Example Output

**Status Response (shape, values depend on your hardware):**
```json
{
  "camera_method_for_rtsp_server": "usb",
  "rtsp_server_running": true,
  "stream_connected": true,
  "last_frame_age_s": "<seconds>",
  "last_result_age_s": "<seconds>",
  "continuous_processing_running": true,
  "metrics": {
    "processing_fps": "<fps>",
    "detect_ms": "<ms>",
    "embed_ms": "<ms>",
    "serve_ms": "<ms>",
    "total_ms": "<ms>"
  },
  "last_objects": [
    {
      "box": [320, 180, 640, 360],
      "source": "center50",
      "name": "dove-cream",
      "confidence": 51.9,
      "object_id": "6"
    }
  ],
  "mqtt_connected": true
}
```

## Known Limitations

- The classifier is a RandomForest on ImageNet features, trained on a handful of images per class from one scene. Expect sensitivity to background, lighting, distance and camera. Retrain on frames from the deployment camera.
- It is a closed-set classifier. The `Unknown` class is partly synthetic (noise added to known features), so it does not reliably reject unfamiliar objects. Add real background frames as `Unknown`.
- The COCO-trained YOLO does not know most of these objects, so recognition mostly relies on the centre crops. Place the object near the middle of the frame.
- Forest probabilities are not calibrated. Tune the threshold from the logged `top3` scores.
- A person in view is treated like any other candidate crop.
