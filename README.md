# Face Recognition System with Edge Training and Jetson Inference

This project implements a face recognition system using few-shot learning on an edge server with MLflow tracking, and real-time inference on Jetson Nano with CSI camera support.

## System Architecture

- **Edge Server**: Handles model training, few-shot learning, and model management
- **Jetson Nano**: Performs real-time inference with CSI camera
- **MLflow**: Tracks experiments, models, and metrics on both devices
- **SQLite**: Backend store for MLflow (lightweight for edge devices)

## Directory Structure

```
face_recognition_system/
├── edge_server/                    # Edge server components
│   ├── edge_training.py           # Main training script
│   ├── mlflow_config.py           # MLflow configuration
│   ├── mlflow_edge.db             # SQLite database for Edge MLflow
│   ├── mlruns/                    # MLflow artifacts
│   ├── images/                    # Input images for training
│   ├── few_shot_examples/         # Few-shot learning examples
│   │   ├── unknown/               # Unknown face captures
│   │   └── <person_id>/           # Person-specific examples
│   ├── models/                    # Saved models
│   │   ├── face_model_v*.pkl     # Model versions
│   │   ├── face_database.json     # Identity database
│   │   └── face_features.pkl      # Feature cache
│   └── jetson_deployment/         # Deployment package
│       ├── face_model.pkl         # Current production model
│       ├── face_database.json     # Identity database
│       └── deploy.sh              # Deployment script
│
├── jetson_nano/                   # Jetson Nano components
│   ├── jetson_inference_simple.py # Main inference script (no MLflow)
│   ├── models/                    # Deployed models
│   │   ├── face_model.pkl         # Current model
│   │   └── face_database.json     # Identity database
│   └── output/                    # Inference outputs
│       ├── output_*.mp4           # Recorded videos
│       └── screenshot_*.jpg       # Screenshots
│
└── mlflow_ui.sh                   # Script to launch MLflow UI
```

## Prerequisites

### Edge Server Requirements
```bash
# Python 3.8+
pip install opencv-python numpy torch torchvision
pip install scikit-learn matplotlib pillow
pip install mlflow ultralytics
pip install ipython jupyter  # Optional for notebook interface
```

### Jetson Nano Requirements
```bash
# Python 3.6+ (Jetson typically has 3.6)
# Install PyTorch for Jetson (ARM64)
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# Other dependencies
pip3 install opencv-python numpy
pip3 install scikit-learn pillow

# For CSI camera support
sudo apt-get update
sudo apt-get install python3-opencv

# Note: MLflow is NOT required on Jetson (only on Edge server)
```

## Setup Instructions

### 1. Edge Server Setup

1. **Clone the repository on edge server:**
   ```bash
   mkdir face_recognition_system
   cd face_recognition_system
   mkdir edge_server
   cd edge_server
   ```

2. **Copy the scripts:**
   - Copy `edge_training.py` to `edge_server/`
   - Copy `mlflow_config.py` to `edge_server/`

3. **Initialize MLflow:**
   ```bash
   python mlflow_config.py
   ```

4. **Prepare training images:**
   ```bash
   mkdir images
   # Copy person images to images/ directory
   # Naming convention: personname_1.jpg, personname_2.jpg, etc.
   ```

### 2. Jetson Nano Setup

1. **SSH into Jetson Nano:**
   ```bash
   ssh newcastleuni@192.168.50.94
   ```

2. **Create project directory:**
   ```bash
   mkdir ~/face_recognition
   cd ~/face_recognition
   ```

3. **Copy the scripts:**
   - Copy `jetson_inference.py` to Jetson
   - Copy `mlflow_config.py` to Jetson

4. **Initialize MLflow:**
   ```bash
   python3 mlflow_config.py
   ```

5. **Test camera connection:**
   ```bash
   python3 -c "import jetson_inference; jetson_inference.test_camera_connection()"
   ```

### 3. Network Configuration

1. **Ensure edge server and Jetson are on same network**
2. **Configure SSH keys for passwordless deployment:**
   ```bash
   # On edge server
   ssh-keygen -t rsa
   ssh-copy-id newcastleuni@192.168.50.94
   ```

## Usage

### Training on Edge Server

1. **Register new identities:**
   ```python
   from edge_training import EdgeFaceRecognitionTrainer
   
   # Initialize trainer
   trainer = EdgeFaceRecognitionTrainer()
   
   # Register a person
   person_name = "John Doe"
   image_paths = ["images/john_1.jpg", "images/john_2.jpg", "images/john_3.jpg"]
   success, person_id = trainer.train_few_shot_model(person_name, image_paths)
   ```

2. **Batch registration from folders:**
   ```python
   # Register multiple people
   trainer.register_new_identity_from_folder("Alice Smith", "images/alice/")
   trainer.register_new_identity_from_folder("Bob Johnson", "images/bob/")
   ```

3. **Evaluate model:**
   ```python
   test_images = ["test/alice_test.jpg", "test/bob_test.jpg"]
   accuracy = trainer.evaluate_model(test_images)
   ```

4. **Deploy to Jetson:**
   ```python
   trainer.deploy_to_jetson(jetson_ip="192.168.50.94")
   ```

   Then run the deployment script:
   ```bash
   cd jetson_deployment
   ./deploy.sh
   ```

### Inference on Jetson Nano

1. **Run real-time inference:**
   ```bash
   python3 jetson_inference_simple.py
   
   # Additional options:
   python3 jetson_inference_simple.py --save              # Save output video
   python3 jetson_inference_simple.py --no-display        # Run headless
   python3 jetson_inference_simple.py --benchmark         # Run performance test
   ```

2. **Fetch latest model from edge server:**
   ```bash
   python3 jetson_inference_simple.py --fetch-model --edge-ip 192.168.50.1 --edge-user edgeuser
   ```

3. **Run benchmark:**
   ```bash
   python3 jetson_inference_simple.py --benchmark --duration 60
   ```

### Monitoring

1. **View Edge MLflow UI:**
   ```bash
   # On edge server
   ./mlflow_ui.sh edge
   # Access at http://edge-server-ip:5000
   ```

2. **Monitor Jetson Performance:**
   ```bash
   # On Jetson (in separate terminal)
   tegrastats
   
   # Or use the built-in performance display in the inference window
   ```

## Model Management

### Model Versioning
- Models are automatically versioned (v1, v2, v3, etc.)
- Each training session creates a new version
- MLflow tracks all model versions and metrics

### Model Deployment Flow
1. Train model on edge server
2. Evaluate performance
3. Deploy to Jetson using deployment script or SCP
4. Jetson fetches model using `--fetch-model` flag

### Manual Model Deployment
```bash
# On edge server
cd jetson_deployment
scp face_model.pkl face_database.json newcastleuni@192.168.50.94:~/face_recognition/models/

# On Jetson
python3 jetson_inference_simple.py --fetch-model
```

## Performance Optimization

### Edge Server
- Use GPU if available for faster training
- Batch process images for efficiency
- Use MobileNetV3 for faster feature extraction

### Jetson Nano
- Use TensorRT for model optimization (future enhancement)
- Adjust camera resolution for performance
- Use threading for camera capture
- Monitor GPU usage with `tegrastats`

## Troubleshooting

### Camera Issues on Jetson
```bash
# Test CSI camera
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink

# Check video devices
ls /dev/video*

# Use USB camera as fallback
# The code automatically falls back to USB if CSI fails
```

### MLflow Connection Issues
```bash
# Check SQLite database
sqlite3 mlflow_edge.db ".tables"

# Reset MLflow
rm -rf mlruns/ mlflow_*.db
python mlflow_config.py
```

### Model Loading Errors
```bash
# Verify model files
ls -la models/
# Check permissions
chmod 644 models/*
```

## Security Considerations

1. **SSH Security**: Use key-based authentication
2. **Model Protection**: Encrypt model files in production
3. **Network Security**: Use VPN for edge-to-jetson communication
4. **Access Control**: Implement authentication for MLflow UI

## Future Enhancements

1. **TensorRT Integration**: Optimize models for Jetson inference
2. **REST API**: Add HTTP endpoints for model updates
3. **Distributed Training**: Support multi-edge training
4. **Active Learning**: Automatically retrain on unknown faces
5. **Privacy Features**: Add face anonymization options