#!/usr/bin/env python3
"""Jetson object recognition client.

Pipeline per frame:
  RTSP camera -> YOLOv8n candidate boxes (all COCO classes, NMS) + centre crops
  -> MobileNetV3-small embeddings (one batched forward pass)
  -> model server /invocations (RandomForest trained by the training script)
  -> annotated MJPEG stream, JSONL results file, MQTT.

Crops are built the same way as in training: YOLO boxes (conf 0.25, NMS 0.45,
min side 30 px) or the 80% centre region. Frames stay BGR all the way into
ToPILImage, exactly like the training script, so do NOT add a BGR->RGB step.

Environment variables (all optional):
  MODEL_SERVICE_URL   default http://localhost:5000
  MQTT_BROKER_HOST    default 10.70.0.64
  MQTT_PORT           default 1883
  MQTT_TOPIC          default jetson/face_recognition/results
  RTSP_AUTO_START     default true
  CAMERA_METHOD       csi | usb (skips the startup camera probe)
  YOLO_CONF           default 0.25
  CAPTURE_LABEL       if set, saves raw frames to captured/<label>/ for retraining
  CAPTURE_EVERY_S     default 1.5
  CAPTURE_MAX         default 40
"""
import os

# Must be set before cv2 is imported. Lower RTSP latency on the FFMPEG backend.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import json
import logging
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify
from torchvision.ops import nms
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:5000")
MQTT_HOST = os.environ.get("MQTT_BROKER_HOST", "10.70.0.64")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "jetson/face_recognition/results")
RTSP_PORT = os.environ.get("RTSP_PORT", "8554")
FRAME_W, FRAME_H = 1280, 720

YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
NMS_IOU = 0.45
MIN_BOX = 30                 # same minimum side as the training script
CENTER_FRACS = (0.8, 0.5)    # 0.8 is the training fallback, 0.5 is a tighter crop

CAPTURE_LABEL = os.environ.get("CAPTURE_LABEL", "").strip()
CAPTURE_EVERY_S = float(os.environ.get("CAPTURE_EVERY_S", "1.5"))
CAPTURE_MAX = int(os.environ.get("CAPTURE_MAX", "40"))

NOT_RECOGNISED = {
    "Unknown", "Error", "Connection Error", "API Error", "Timeout",
    "Recognition Error", "No Prediction", "Feature Error",
}

app = Flask(__name__)


class JetsonObjectRecognition:
    def __init__(self):
        self.model_service_url = MODEL_SERVICE_URL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device == "cuda"
        self.yolo_device = 0 if self.use_half else "cpu"
        logger.info("Device: %s, model service: %s", self.device, self.model_service_url)

        self.rtsp_port = RTSP_PORT
        self.rtsp_process = None
        self.rtsp_running = False
        self.rtsp_threads: List[threading.Thread] = []

        # Models (fail fast: no silent dummy features)
        self.yolo = YOLO("yolov8n.pt")
        self.feature_extractor = self._load_feature_extractor()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._warmup()

        # Output
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_file_path = os.path.join(self.output_dir, "recognition_results.jsonl")
        self._file_lock = threading.Lock()

        # Metrics (seconds, last 30 frames)
        self.t_detect = deque(maxlen=30)
        self.t_embed = deque(maxlen=30)
        self.t_serve = deque(maxlen=30)
        self.t_total = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)

        # Frames and state
        self.latest_raw = None
        self.last_raw_ts = 0.0
        self.last_frame = None          # annotated frame for /video_feed
        self.frame_id = 0
        self.last_results: List[Dict[str, Any]] = []
        self.last_result_ts = 0.0
        self.stream_connected = False
        self.rtsp_capture = None

        self.camera_running = False
        self.processing_running = False
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=1)  # drop-oldest, no stale frames
        self.camera_thread = None
        self.processing_thread = None

        self._last_srv_err = 0.0
        self._last_capture = 0.0
        self._capture_count = 0

        self.camera_method = os.environ.get("CAMERA_METHOD") or self.detect_camera_method()
        logger.info("Camera method for RTSP server: %s", self.camera_method)

        # MQTT
        self.mqtt_client = None
        self._init_mqtt()

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def _load_feature_extractor(self):
        logger.info("Loading MobileNetV3-small feature extractor...")
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        extractor = torch.nn.Sequential(*list(model.children())[:-1])
        extractor.to(self.device)
        extractor.eval()
        return extractor

    def _warmup(self):
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.yolo(dummy, verbose=False, device=self.yolo_device, half=self.use_half)
        self.embed([dummy[:224, :224]])
        logger.info("Models warmed up")

    # ------------------------------------------------------------------
    # MQTT
    # ------------------------------------------------------------------
    def _init_mqtt(self):
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)  # paho 2.x
        except AttributeError:
            client = mqtt.Client()                                   # paho 1.x
        client.on_connect = self._on_mqtt_connect
        client.on_disconnect = self._on_mqtt_disconnect
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.mqtt_client = client
        try:
            # connect_async so an unreachable broker at startup is retried forever
            client.connect_async(MQTT_HOST, MQTT_PORT, 60)
            client.loop_start()
            logger.info("MQTT connecting to %s:%s", MQTT_HOST, MQTT_PORT)
        except Exception as e:
            logger.error("MQTT setup failed: %s", e)

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker %s", MQTT_HOST)
        else:
            logger.error("MQTT connect failed, rc=%s", rc)

    def _on_mqtt_disconnect(self, client, userdata, rc):
        logger.warning("Disconnected from MQTT broker (rc=%s), paho will retry", rc)

    def mqtt_connected(self) -> bool:
        return bool(self.mqtt_client and self.mqtt_client.is_connected())

    def signal_handler(self, signum, frame):
        logger.info("Signal %s received, shutting down", signum)
        self.stop_streaming()
        self.stop_rtsp_server()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Camera detection for the RTSP server pipeline
    # ------------------------------------------------------------------
    def _run_gst_test(self, cmd, test_path, label):
        if os.path.exists(test_path):
            os.remove(test_path)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            logger.error("%s camera test timed out", label)
            return False
        except Exception as e:
            logger.error("%s camera test error: %s", label, e)
            return False
        ok = (
            r.returncode == 0
            and "ERROR:" not in r.stderr
            and "failed" not in r.stderr
            and os.path.exists(test_path)
            and os.path.getsize(test_path) > 0
        )
        if not ok:
            logger.warning("%s camera test failed (rc=%s): %s", label, r.returncode, r.stderr.strip()[:300])
        if os.path.exists(test_path):
            os.remove(test_path)
        return ok

    def test_csi_camera(self):
        path = "/tmp/test_csi.jpg"
        cmd = [
            "gst-launch-1.0", "-e", "nvarguscamerasrc", "num-buffers=1", "!",
            f"video/x-raw(memory:NVMM),width={FRAME_W},height={FRAME_H},framerate=30/1", "!",
            "nvvidconv", "!", "jpegenc", "!", "filesink", f"location={path}",
        ]
        return self._run_gst_test(cmd, path, "CSI")

    def test_usb_camera(self):
        path = "/tmp/test_usb.jpg"
        cmd = [
            "gst-launch-1.0", "-e", "v4l2src", "device=/dev/video0", "num-buffers=1", "!",
            "videoconvert", "!", "videoscale", "!",
            f"video/x-raw,width={FRAME_W},height={FRAME_H},framerate=30/1", "!",
            "jpegenc", "!", "filesink", f"location={path}",
        ]
        return self._run_gst_test(cmd, path, "USB")

    def detect_camera_method(self):
        logger.info("Probing CSI camera...")
        if self.test_csi_camera():
            return "csi"
        logger.info("Probing USB camera...")
        try:
            r = subprocess.run("gst-inspect-1.0 v4l2src", shell=True, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and os.path.exists("/dev/video0") and self.test_usb_camera():
                return "usb"
        except Exception as e:
            logger.error("USB probe error: %s", e)
        logger.warning("No camera detected, defaulting to csi (will probably fail)")
        return "csi"

    # ------------------------------------------------------------------
    # RTSP server (external ./rtsp_server_fps binary) and capture
    # ------------------------------------------------------------------
    def _pipe_logger(self, pipe, level_fn):
        for line in iter(pipe.readline, ""):
            line = line.strip()
            if line:
                level_fn("RTSP server: %s", line)

    def start_rtsp_server(self):
        if self.rtsp_running:
            logger.warning("RTSP server already running")
            return True
        if self.camera_method == "csi":
            pipeline = (
                f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={FRAME_W},height={FRAME_H},"
                f"framerate=30/1,format=NV12 ! nvv4l2h264enc bitrate=2000000 ! h264parse ! "
                f"rtph264pay name=pay0 pt=96"
            )
        elif self.camera_method == "usb":
            pipeline = (
                f"v4l2src device=/dev/video0 ! videoconvert ! videoscale ! "
                f"video/x-raw,width={FRAME_W},height={FRAME_H},framerate=30/1 ! "
                f"x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! h264parse ! "
                f"rtph264pay name=pay0 pt=96"
            )
        else:
            logger.error("No valid camera method, cannot start RTSP server")
            return False

        logger.info("Starting RTSP server: %s", pipeline)
        try:
            self.rtsp_process = subprocess.Popen(
                ["./rtsp_server_fps", f"( {pipeline} )"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except Exception as e:
            logger.error("Could not start RTSP server: %s", e, exc_info=True)
            return False

        self.rtsp_threads = [
            threading.Thread(target=self._pipe_logger, args=(self.rtsp_process.stdout, logger.info), daemon=True),
            threading.Thread(target=self._pipe_logger, args=(self.rtsp_process.stderr, logger.warning), daemon=True),
        ]
        for t in self.rtsp_threads:
            t.start()

        time.sleep(2)
        if self.rtsp_process.poll() is None:
            self.rtsp_running = True
            logger.info("RTSP server up at rtsp://localhost:%s/test", self.rtsp_port)
            return True
        logger.error("RTSP server exited immediately, see the log lines above")
        return False

    def stop_rtsp_server(self):
        self.rtsp_running = False
        if self.rtsp_capture is not None:
            self.rtsp_capture.release()
            self.rtsp_capture = None
        if self.rtsp_process:
            try:
                self.rtsp_process.terminate()
                self.rtsp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rtsp_process.kill()
                self.rtsp_process.wait()
            except Exception as e:
                logger.debug("Error stopping RTSP process: %s", e)
            finally:
                self.rtsp_process = None
                logger.info("RTSP server stopped")

    def _open_capture(self):
        url = f"rtsp://localhost:{self.rtsp_port}/test"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        return None

    def _set_stream_state(self, connected: bool):
        if connected != self.stream_connected:
            self.stream_connected = connected
            logger.info("RTSP stream %s", "connected" if connected else "lost")

    def _camera_loop(self):
        """Reads RTSP continuously, keeps only the newest frame."""
        backoff = 0.5
        while self.camera_running:
            if self.rtsp_capture is None:
                self.rtsp_capture = self._open_capture()
                if self.rtsp_capture is None:
                    self._set_stream_state(False)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue
            ok, frame = self.rtsp_capture.read()
            if not ok:
                self._set_stream_state(False)
                self.rtsp_capture.release()
                self.rtsp_capture = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff = 0.5
            self._set_stream_state(True)
            self.latest_raw = frame
            self.last_raw_ts = time.time()
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

    # ------------------------------------------------------------------
    # Detection and recognition
    # ------------------------------------------------------------------
    def candidate_boxes(self, frame) -> List[Dict[str, Any]]:
        """YOLO boxes (all classes, NMS) plus centre crops, as in training."""
        h, w = frame.shape[:2]
        cands = []

        results = self.yolo(frame, verbose=False, conf=YOLO_CONF,
                            device=self.yolo_device, half=self.use_half)
        xyxy, scores, names = [], [], []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if (x2 - x1) > MIN_BOX and (y2 - y1) > MIN_BOX:
                    xyxy.append([x1, y1, x2, y2])
                    scores.append(float(b.conf[0]))
                    names.append(self.yolo.names[int(b.cls[0])])
        if xyxy:
            keep = nms(torch.tensor(xyxy, dtype=torch.float32),
                       torch.tensor(scores, dtype=torch.float32), NMS_IOU).tolist()
            for i in keep:
                x1, y1, x2, y2 = xyxy[i]
                x, y = max(0, int(x1)), max(0, int(y1))
                bw, bh = min(w, int(x2)) - x, min(h, int(y2)) - y
                if bw > MIN_BOX and bh > MIN_BOX:
                    cands.append({"box": (x, y, bw, bh), "src": ""})

        for frac in CENTER_FRACS:
            cw, ch = int(w * frac), int(h * frac)
            cands.append({"box": ((w - cw) // 2, (h - ch) // 2, cw, ch),
                          "src": f"center{int(frac * 100)}"})
        return cands

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        """One batched MobileNetV3 forward pass. Crops are BGR, like training."""
        batch = torch.stack([self.transform(np.ascontiguousarray(c)) for c in crops]).to(self.device)
        with torch.inference_mode():
            feats = self.feature_extractor(batch).flatten(1)
        return feats.cpu().numpy()

    def _log_server_error(self, msg):
        now = time.time()
        if now - self._last_srv_err > 10:
            logger.error(msg)
            self._last_srv_err = now

    def recognize_batch(self, vectors: List[List[float]]) -> List[Dict[str, Any]]:
        n = len(vectors)

        def fail(name):
            return [{"name": name, "confidence": 0, "person_id": None} for _ in range(n)]

        try:
            resp = requests.post(
                f"{self.model_service_url}/invocations",
                json={"instances": [{"image_feature_vector": v} for v in vectors]},
                timeout=10,
            )
            if resp.status_code != 200:
                self._log_server_error(f"Model server returned {resp.status_code}: {resp.text[:200]}")
                return fail("API Error")
            preds = resp.json().get("predictions", [])
            if len(preds) != n:
                self._log_server_error(f"Model server returned {len(preds)} predictions for {n} crops")
                return fail("No Prediction")
            return preds
        except requests.exceptions.Timeout:
            self._log_server_error("Model server timed out")
            return fail("Timeout")
        except requests.exceptions.ConnectionError as e:
            self._log_server_error(f"Cannot reach model server at {self.model_service_url}: {e}")
            return fail("Connection Error")
        except Exception as e:
            self._log_server_error(f"Model server call failed: {e}")
            return fail("Recognition Error")

    def process_frame(self, frame):
        """Returns (objects, annotated_frame). Objects are recognised classes only,
        one box per class (the most confident candidate)."""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        cands = self.candidate_boxes(frame)
        t1 = time.perf_counter()

        crops, valid = [], []
        for c in cands:
            x, y, bw, bh = c["box"]
            roi = frame[y:y + bh, x:x + bw]
            if roi.size:
                crops.append(roi)
                valid.append(c)

        best: Dict[str, Dict[str, Any]] = {}
        t2 = t3 = t1
        if crops:
            feats = self.embed(crops)
            t2 = time.perf_counter()
            preds = self.recognize_batch(feats.tolist())
            t3 = time.perf_counter()
            for c, p in zip(valid, preds):
                name = p.get("name", "Unknown")
                conf = float(p.get("confidence", 0))
                logger.debug("candidate %s %s -> %s %.1f%%", c["src"], c["box"], name, conf)
                if name in NOT_RECOGNISED:
                    continue
                if name not in best or conf > best[name]["confidence"]:
                    x, y, bw, bh = c["box"]
                    best[name] = {
                        "box": [x, y, bw, bh],
                        "source": c["src"],
                        "name": name,
                        "confidence": conf,
                        "object_id": p.get("person_id"),
                    }
        objects = sorted(best.values(), key=lambda o: -o["confidence"])
        annotated = self.draw_overlay(frame, objects)
        t4 = time.perf_counter()

        self.t_detect.append(t1 - t0)
        self.t_embed.append(t2 - t1)
        self.t_serve.append(t3 - t2)
        self.t_total.append(t4 - t0)
        self.frame_times.append(time.time())
        return objects, annotated

    def draw_overlay(self, frame, objects):
        out = frame.copy()
        for o in objects:
            x, y, bw, bh = o["box"]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
            label = f"{o['name']} {o['confidence']:.0f}% [{o['source']}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(th + 6, y)
            cv2.rectangle(out, (x, ty - th - 6), (x + tw + 4, ty), (0, 200, 0), -1)
            cv2.putText(out, label, (x + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Metrics and outputs
    # ------------------------------------------------------------------
    @staticmethod
    def _ms(d) -> float:
        return float(np.mean(d) * 1000) if d else 0.0

    def metrics(self) -> Dict[str, float]:
        fps = 0.0
        if len(self.frame_times) >= 2:
            span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / span if span > 0 else 0.0
        return {
            "processing_fps": round(fps, 2),
            "detect_ms": round(self._ms(self.t_detect), 1),
            "embed_ms": round(self._ms(self.t_embed), 1),
            "serve_ms": round(self._ms(self.t_serve), 1),
            "total_ms": round(self._ms(self.t_total), 1),
        }

    def save_result(self, results: Dict[str, Any]):
        try:
            with self._file_lock:
                with open(self.results_file_path, "a") as f:
                    f.write(json.dumps(results) + "\n")
        except Exception as e:
            logger.error("Could not write results file: %s", e)

    def publish(self, results: Dict[str, Any]):
        if self.mqtt_connected():
            try:
                self.mqtt_client.publish(MQTT_TOPIC, json.dumps(results), qos=0)
            except Exception as e:
                logger.error("MQTT publish failed: %s", e)
        else:
            logger.debug("MQTT not connected, result not published")

    def build_results(self, frame, objects, source):
        return {
            "timestamp": datetime.now().isoformat(),
            "source_method": source,
            "frame_dimensions": {"width": frame.shape[1], "height": frame.shape[0]},
            "detected_objects": objects,
            "metrics": self.metrics(),
        }

    def _maybe_capture_frame(self, frame):
        """Capture mode: save raw frames for retraining with the training script."""
        if not CAPTURE_LABEL or self._capture_count >= CAPTURE_MAX:
            return
        now = time.time()
        if now - self._last_capture < CAPTURE_EVERY_S:
            return
        d = os.path.join("captured", CAPTURE_LABEL)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{CAPTURE_LABEL}_{int(now * 1000)}.jpg")
        cv2.imwrite(path, frame)
        self._last_capture = now
        self._capture_count += 1
        logger.info("Captured training frame %d/%d: %s", self._capture_count, CAPTURE_MAX, path)

    def _processing_loop(self):
        while self.processing_running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._maybe_capture_frame(frame)
                objects, annotated = self.process_frame(frame)
                results = self.build_results(frame, objects, "continuous")
                if objects:
                    self.save_result(results)
                self.publish(results)
                self.last_frame = annotated
                self.frame_id += 1
                self.last_results = objects
                self.last_result_ts = time.time()
            except Exception as e:
                logger.error("Processing loop error: %s", e, exc_info=True)
                time.sleep(0.1)

    def start_streaming(self):
        if self.camera_running:
            logger.warning("Streaming already running")
            return
        self.camera_running = True
        self.processing_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.camera_thread.start()
        self.processing_thread.start()
        logger.info("Continuous recognition started")

    def stop_streaming(self):
        self.camera_running = False
        self.processing_running = False
        for t in (self.camera_thread, self.processing_thread):
            if t and t.is_alive():
                t.join(timeout=2)
        if self.rtsp_capture is not None:
            self.rtsp_capture.release()
            self.rtsp_capture = None
        logger.info("Continuous recognition stopped")


rec = JetsonObjectRecognition()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def root():
    return jsonify({
        "name": "Jetson Object Recognition (RTSP, YOLO + MobileNetV3 + model server)",
        "version": "6.0",
        "camera_method_for_rtsp_server": rec.camera_method,
        "rtsp_server_running": rec.rtsp_running,
        "continuous_processing_running": rec.camera_running,
        "mqtt_broker": f"{MQTT_HOST}:{MQTT_PORT}",
        "mqtt_topic": MQTT_TOPIC,
        "capture_label": CAPTURE_LABEL or None,
        "endpoints": {
            "GET /video_player": "Web page with the annotated stream",
            "GET /video_feed": "MJPEG stream with boxes and labels",
            "POST /capture": "Process the latest frame once",
            "POST /start_rtsp": "Start RTSP server",
            "POST /stop_rtsp": "Stop RTSP server",
            "POST /start_streaming": "Start continuous recognition",
            "POST /stop_streaming": "Stop continuous recognition",
            "GET /status": "Detailed status and timings",
            "GET /ping": "Liveness plus pipeline health",
            "GET /ready": "200 only when frames are flowing",
        },
    })


@app.route("/capture", methods=["POST"])
def capture():
    frame = rec.latest_raw
    if frame is None:
        return jsonify({"success": False, "error": "No frame yet. Is the RTSP stream up?"}), 503
    try:
        objects, annotated = rec.process_frame(frame.copy())
        results = rec.build_results(frame, objects, "manual_capture")
        rec.save_result(results)
        rec.publish(results)
        rec.last_frame = annotated
        rec.frame_id += 1
        rec.last_results = objects
        return jsonify({"success": True, "objects": objects, "metrics": rec.metrics()})
    except Exception as e:
        logger.error("Capture error: %s", e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/start_rtsp", methods=["POST"])
def start_rtsp():
    ok = rec.start_rtsp_server()
    code = 200 if ok else 500
    return jsonify({"success": ok, "rtsp_url": f"rtsp://localhost:{rec.rtsp_port}/test"}), code


@app.route("/stop_rtsp", methods=["POST"])
def stop_rtsp():
    rec.stop_rtsp_server()
    return jsonify({"success": True})


@app.route("/start_streaming", methods=["POST"])
def start_streaming_route():
    rec.start_streaming()
    return jsonify({"success": True})


@app.route("/stop_streaming", methods=["POST"])
def stop_streaming_route():
    rec.stop_streaming()
    return jsonify({"success": True})


def _age(ts):
    return round(time.time() - ts, 2) if ts else None


@app.route("/status")
def status():
    return jsonify({
        "camera_method_for_rtsp_server": rec.camera_method,
        "rtsp_server_running": rec.rtsp_running,
        "stream_connected": rec.stream_connected,
        "last_frame_age_s": _age(rec.last_raw_ts),
        "last_result_age_s": _age(rec.last_result_ts),
        "continuous_processing_running": rec.camera_running,
        "metrics": rec.metrics(),
        "last_objects": rec.last_results,
        "mqtt_connected": rec.mqtt_connected(),
        "capture_label": CAPTURE_LABEL or None,
        "captured_frames": rec._capture_count,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/video_feed")
def video_feed():
    def generate():
        last_id = -1
        while True:
            frame = rec.last_frame
            if frame is None or rec.frame_id == last_id:
                time.sleep(0.03)
                continue
            last_id = rec.frame_id
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_player")
def video_player():
    return """<!DOCTYPE html>
<html><head><title>Jetson Object Recognition</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
.box { max-width: 1200px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 8px; }
img { max-width: 100%; height: auto; border: 2px solid #333; }
button { padding: 8px 16px; margin: 4px; }
#status { margin: 10px 0; padding: 10px; background: #e9ecef; font-family: monospace; }
</style></head><body><div class="box">
<h2>Jetson Object Recognition</h2>
<img src="/video_feed" id="v" alt="stream">
<div id="status">loading...</div>
<button onclick="post('/start_streaming')">Start processing</button>
<button onclick="post('/stop_streaming')">Stop processing</button>
<button onclick="post('/capture')">Process one frame</button>
</div>
<script>
function post(u){ fetch(u,{method:'POST'}).then(r=>r.json()).then(()=>poll()); }
function poll(){
  fetch('/status').then(r=>r.json()).then(d=>{
    const m=d.metrics;
    document.getElementById('status').textContent =
      'stream:'+(d.stream_connected?'ok':'DOWN')+' | mqtt:'+(d.mqtt_connected?'ok':'DOWN')+
      ' | fps:'+m.processing_fps+' | detect:'+m.detect_ms+'ms embed:'+m.embed_ms+
      'ms serve:'+m.serve_ms+'ms total:'+m.total_ms+'ms | objects:'+
      d.last_objects.map(o=>o.name+' '+o.confidence.toFixed(0)+'%').join(', ');
  }).catch(()=>{document.getElementById('status').textContent='status error';});
}
setInterval(poll,2000); poll();
</script></body></html>"""


@app.route("/ping")
def ping():
    try:
        r = requests.get(f"{rec.model_service_url}/ping", timeout=2)
        model_status = "OK" if r.status_code == 200 else "ERROR"
    except Exception:
        model_status = "UNREACHABLE"
    return jsonify({
        "status": "OK",
        "model_service": model_status,
        "stream_connected": rec.stream_connected,
        "last_frame_age_s": _age(rec.last_raw_ts),
        "last_result_age_s": _age(rec.last_result_ts),
        "mqtt_connected": rec.mqtt_connected(),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/ready")
def ready():
    age = _age(rec.last_result_ts)
    ok = rec.stream_connected and age is not None and age < 5
    return jsonify({"ready": ok, "last_result_age_s": age}), (200 if ok else 503)


if __name__ == "__main__":
    try:
        if os.environ.get("RTSP_AUTO_START", "true").lower() == "true":
            rec.start_rtsp_server()
        rec.start_streaming()
        app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        rec.stop_streaming()
        rec.stop_rtsp_server()
