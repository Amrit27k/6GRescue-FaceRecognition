# MLflow Jetson File Transfer Plugin

A simple MLflow plugin for transferring face recognition models and files to NVIDIA Jetson devices.

## What It Does

- Downloads models from MLflow registry
- Transfers face recognition files (face_features.pkl, face_database.json, model_params.json, etc.)
- Copies Python scripts (inference_server.py, client.py)
- Transfers everything to Jetson via SSH/SCP
- **Note**: Only transfers files - no automatic execution on Jetson

## Quick Setup

### 1. Install Plugin

```bash
cd mlflow_plugins
pip install -e .
```

### 2. Configure Jetson Access

```bash
# Generate SSH key
ssh-keygen -t rsa -f ~/.ssh/jetson_key

# Copy to Jetson (enter password when prompted)
ssh-copy-id -i ~/.ssh/jetson_key.pub newcastleuni@192.168.2.100

# Test connection
ssh -i ~/.ssh/jetson_key newcastleuni@192.168.2.100 "echo 'SSH works'"
```

### 3. Update Configuration

Edit `deployment_configs/simple_jetson.yaml`:
```yaml
device_ip: "192.168.2.100"
username: "newcastleuni"
ssh_key_path: "~/.ssh/jetson_key"
deployment_base_path: "/home/newcastleuni/mlflow_deployments"
```

### 4. Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_edge.db --port 5000 --host 0.0.0.0
```

## Usage

### Python API
```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("jetson://192.168.2.100")
deployment = client.create_deployment(
    name="face_recognition_files",
    model_uri="models:/face_recognition_model/latest"
)
```

### MLflow CLI
```bash
mlflow deployments create \
  -t jetson://192.168.2.100 \
  -m models:/face_recognition_model/latest \
  --name face_recognition_files
```

### Example Script
```bash
python examples/simple_file_transfer.py
```

## Files Transferred

The plugin automatically finds and transfers:
- MLflow model files
- `face_features.pkl`, `face_database.json`, `model_params.json`
- `inference_server.py`, `client.py`
- `Dockerfile.inference-server`, `Dockerfile.model-server`

Files are extracted to: `/home/newcastleuni/mlflow_deployments/{deployment_name}/`

## Development Mode

When making changes to the plugin:

### 1. Edit Plugin Files
```bash
# Edit any plugin file
nano mlflow_plugins/jetson_deployment/simple_plugin.py
# or
nano mlflow_plugins/jetson_deployment/ssh_manager.py
# or
nano mlflow_plugins/jetson_deployment/simple_package_builder.py
```

### 2. Reinstall Plugin
```bash
cd mlflow_plugins
pip install -e . --force-reinstall
cd ..
```

### 3. Test Changes
```bash
python examples/simple_file_transfer.py
```

### Quick Development Workflow
```bash
# Make changes, then:
cd mlflow_plugins && pip install -e . --force-reinstall && cd .. && python examples/simple_file_transfer.py
```

## Troubleshooting

### Plugin Registration Issues
```bash
# Test plugin loading
python -c "from mlflow.deployments import get_deploy_client; 
client = get_deploy_client('jetson://192.168.2.100'); 
print('Plugin works!')"
```

### SSH Connection Issues
```bash
# Test SSH manually
ssh -i ~/.ssh/jetson_key newcastleuni@192.168.2.100

# Check SSH key permissions
chmod 600 ~/.ssh/jetson_key
chmod 644 ~/.ssh/jetson_key.pub
```

### File Transfer Issues
```bash
# Check what files are found
find . -name "face_features.pkl" -o -name "inference_server.py"

# Test with dummy files if needed
mkdir -p jetson
echo '{}' > jetson/face_features.pkl
echo '{}' > jetson/face_database.json
```

### Clear Python Cache
```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

## Commands Reference

```bash
# Install plugin
pip install -e ./mlflow_plugins/

# Test plugin
python -c "from mlflow.deployments import get_deploy_client; print('Plugin OK')"

# Transfer files
python examples/simple_file_transfer.py

# List deployments
mlflow deployments list -t jetson://192.168.2.100

# Delete deployment
mlflow deployments delete -t jetson://192.168.2.100 --name face_recognition_files

# Plugin help
mlflow deployments help -t jetson
```