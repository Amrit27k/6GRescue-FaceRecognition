import mlflow
import os
import sqlite3
from datetime import datetime
import subprocess
import yaml

class MLflowConfig:
    """Configure MLflow for Edge server only (no Jetson MLflow)"""
    
    @staticmethod
    def setup_edge_mlflow(db_path="mlflow_edge.db"):
        """Setup MLflow for Edge server"""
        # Create MLflow directory structure
        mlflow_dir = "mlruns"
        os.makedirs(mlflow_dir, exist_ok=True)
        
        # Set tracking URI
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiments
        experiments = [
            "face_recognition_training",
            "model_evaluation",
            "few_shot_learning",
            "k8s_deployment"
        ]
        
        for exp_name in experiments:
            try:
                mlflow.create_experiment(exp_name)
                print(f"Created experiment: {exp_name}")
            except:
                print(f"Experiment {exp_name} already exists")
        
        # Initialize database schema if needed
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create custom metrics table for edge-specific tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS edge_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version INTEGER,
            num_identities INTEGER,
            training_accuracy REAL,
            deployment_status TEXT,
            k8s_deployment_status TEXT
        )
        """)
        
        # Create deployment tracking table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS k8s_deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_version INTEGER,
            jetson_ip TEXT,
            deployment_status TEXT,
            docker_image TEXT,
            manifest_files TEXT
        )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"Edge MLflow configured with SQLite backend: {tracking_uri}")
        return tracking_uri
    
    @staticmethod
    def log_k8s_deployment(model_version, jetson_ip, status, docker_image, db_path="mlflow_edge.db"):
        """Log K8s deployment information"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO k8s_deployments 
        (model_version, jetson_ip, deployment_status, docker_image, manifest_files)
        VALUES (?, ?, ?, ?, ?)
        """, (model_version, jetson_ip, status, docker_image, "model-service.yaml,inference-app.yaml"))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def create_mlflow_ui_script():
        """Create script to launch MLflow UI"""
        script_content = """#!/bin/bash
# Launch MLflow UI for Edge server

if [ "$1" == "edge" ] || [ -z "$1" ]; then
    echo "Starting MLflow UI for Edge server..."
    mlflow ui --backend-store-uri sqlite:///mlflow_edge.db --port 5000 --host 0.0.0.0
else
    echo "Usage: ./mlflow_ui.sh [edge]"
    echo "  edge   - Start MLflow UI for Edge server on port 5000"
fi
"""
        
        with open("mlflow_ui.sh", 'w') as f:
            f.write(script_content)
        
        os.chmod("mlflow_ui.sh", 0o755)
        print("Created mlflow_ui.sh script")

    @staticmethod
    def create_k8s_build_script():
        """Create script to build and deploy K8s models"""
        script_content = """#!/bin/bash
# Build Docker images for K8s deployment

set -e

MODEL_VERSION=${1:-"latest"}
JETSON_IP=${2:-"192.168.50.94"}
JETSON_USER=${3:-"newcastleuni"}

echo "Building Docker images for model version: $MODEL_VERSION"
echo "Target Jetson: $JETSON_USER@$JETSON_IP"

if [ "$MODEL_VERSION" == "latest" ]; then
    echo "ERROR: Please specify a specific model version"
    echo "Usage: ./build_k8s.sh <version> [jetson_ip] [jetson_user]"
    exit 1
fi

# Build model service using Python script (custom Docker build)
echo "Building MLflow model service..."
python3 -c "
from edge_training import EdgeFaceRecognitionTrainer
trainer = EdgeFaceRecognitionTrainer()
trainer.build_docker_model($MODEL_VERSION)
"

# Build inference application Docker image
echo "Building inference application..."
sudo docker build -t face-recognition-inference:v$MODEL_VERSION \\
    -f Dockerfile.inference \\
    --platform linux/arm64 .

# Save images as tar files for transfer
echo "Saving Docker images..."
if [ ! -f "k8s_deployment/face-recognition-model-v$MODEL_VERSION.tar" ]; then
    echo "ERROR: Model service tar file not found!"
    exit 1
fi

sudo docker save face-recognition-inference:v$MODEL_VERSION > k8s_deployment/face-recognition-inference-v$MODEL_VERSION.tar

# Fix permissions on tar files
sudo chown $USER:$USER k8s_deployment/face-recognition-inference-v$MODEL_VERSION.tar

# Create deployment package
echo "Creating deployment package..."
tar -czf k8s-deployment-v$MODEL_VERSION.tar.gz \\
    k8s_deployment/

echo "Deployment package created: k8s-deployment-v$MODEL_VERSION.tar.gz"
echo ""
echo "To deploy to Jetson:"
echo "1. Copy package: scp k8s-deployment-v$MODEL_VERSION.tar.gz $JETSON_USER@$JETSON_IP:~/"
echo "2. SSH and extract: ssh $JETSON_USER@$JETSON_IP 'tar -xzf k8s-deployment-v$MODEL_VERSION.tar.gz'"
echo "3. Deploy: ssh $JETSON_USER@$JETSON_IP 'cd k8s_deployment && ./deploy.sh'"
echo ""
echo "Or use automated deployment:"
echo "./deploy_to_jetson.sh $MODEL_VERSION $JETSON_IP $JETSON_USER"
"""
        
        with open("build_k8s.sh", 'w') as f:
            f.write(script_content)
        
        os.chmod("build_k8s.sh", 0o755)
        print("Created build_k8s.sh script")
        
        # Also create automated deployment script
        deploy_script = """#!/bin/bash
# Automated deployment to Jetson

set -e

MODEL_VERSION=${1:-"1"}
JETSON_IP=${2:-"192.168.50.94"}
JETSON_USER=${3:-"newcastleuni"}

echo "Automated deployment to $JETSON_USER@$JETSON_IP"
echo "Model version: $MODEL_VERSION"

# Create necessary directories on Jetson
echo "Creating directories on Jetson..."
ssh $JETSON_USER@$JETSON_IP "
    mkdir -p ~/face_recognition &&
    mkdir -p ~/k8s_deployment &&
    echo 'Directories created successfully'
" || echo "Warning: Directory creation failed, continuing anyway..."

# Build if package doesn't exist
if [ ! -f "k8s-deployment-v$MODEL_VERSION.tar.gz" ]; then
    echo "Building deployment package..."
    ./build_k8s.sh $MODEL_VERSION $JETSON_IP $JETSON_USER
fi

# Copy to Jetson
echo "Copying deployment package to Jetson..."
scp k8s-deployment-v$MODEL_VERSION.tar.gz $JETSON_USER@$JETSON_IP:~/

# Deploy on Jetson
echo "Deploying on Jetson..."
ssh $JETSON_USER@$JETSON_IP "
    cd ~ &&
    tar -xzf k8s-deployment-v$MODEL_VERSION.tar.gz &&
    cd k8s_deployment &&
    chmod +x deploy.sh &&
    echo 'Files ready for deployment. Run ./deploy.sh to complete.'
"

echo "✅ Deployment package ready on Jetson!"
echo ""
echo "To complete deployment, SSH to Jetson and run:"
echo "  ssh $JETSON_USER@$JETSON_IP"
echo "  cd k8s_deployment"
echo "  ./deploy.sh"
echo ""
echo "Check status with: ssh $JETSON_USER@$JETSON_IP 'kubectl get pods'"
"""
        
        with open("deploy_to_jetson.sh", 'w') as f:
            f.write(deploy_script)
        
        os.chmod("deploy_to_jetson.sh", 0o755)
        print("Created deploy_to_jetson.sh script")


# Model Registry utilities for K8s deployment
class ModelRegistry:
    """Manage model versions and K8s deployments"""
    
    def __init__(self, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, run_id, model_name="face_recognition_model"):
        """Register a model from a run"""
        model_uri = f"runs:/{run_id}/face_recognition_model"
        
        # Register model
        result = mlflow.register_model(model_uri, model_name)
        
        # Transition to staging
        self.client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging"
        )
        
        print(f"Registered model {model_name} version {result.version}")
        return result.version
    
    def promote_model_for_k8s(self, model_name, version):
        """Promote model from staging to production for K8s deployment"""
        # Archive current production model
        try:
            prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
            for v in prod_versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived"
                )
        except:
            pass
        
        # Promote new version to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        # Add description for K8s deployment
        self.client.update_model_version(
            name=model_name,
            version=version,
            description=f"Model version {version} - Ready for K8s deployment on Jetson"
        )
        
        print(f"Promoted model {model_name} version {version} to Production for K8s deployment")
    
    def get_production_model(self, model_name):
        """Get current production model"""
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0]
        return None
    
    def list_models_for_k8s(self):
        """List all registered models with K8s deployment info"""
        models = self.client.search_registered_models()
        for model in models:
            print(f"\nModel: {model.name}")
            for version in model.latest_versions:
                k8s_ready = "✓" if version.current_stage == "Production" else "✗"
                print(f"  Version {version.version}: {version.current_stage} | K8s Ready: {k8s_ready}")
                if version.description:
                    print(f"    Description: {version.description}")
    
    def create_k8s_model_config(self, model_name, version):
        """Create K8s-specific model configuration"""
        model_version = self.client.get_model_version(model_name, version)
        
        config = {
            "model_name": model_name,
            "model_version": version,
            "model_uri": f"models:/{model_name}/{version}",
            "run_id": model_version.run_id,
            "stage": model_version.current_stage,
            "docker_image": f"face-recognition-model:v{version}",
            "k8s_deployment": {
                "namespace": "default",
                "service_name": "face-recognition-model-service",
                "service_port": 80,
                "container_port": 5000,
                "replicas": 1,
                "resources": {
                    "requests": {"memory": "512Mi", "cpu": "500m"},
                    "limits": {"memory": "1Gi", "cpu": "1000m"}
                }
            }
        }
        
        # Save config
        config_path = f"k8s_deployment/model-config-v{version}.yaml"
        os.makedirs("k8s_deployment", exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created K8s model config: {config_path}")
        return config


class JetsonK8sDeployer:
    """Handle deployment to Jetson K8s cluster"""
    
    def __init__(self, jetson_ip, jetson_user="newcastleuni"):
        self.jetson_ip = jetson_ip
        self.jetson_user = jetson_user
    
    def check_k3s_status(self):
        """Check if K3s is running on Jetson"""
        try:
            cmd = f"ssh {self.jetson_user}@{self.jetson_ip} 'kubectl get nodes'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def deploy_model_service(self, model_version, docker_image_tar):
        """Deploy model service to K3s"""
        print(f"Deploying model service v{model_version} to Jetson K3s...")
        
        try:
            # Copy Docker image
            scp_cmd = f"scp {docker_image_tar} {self.jetson_user}@{self.jetson_ip}:~/"
            subprocess.run(scp_cmd, shell=True, check=True)
            
            # Import image to K3s
            import_cmd = f"""ssh {self.jetson_user}@{self.jetson_ip} '
                sudo k3s ctr images import {docker_image_tar} &&
                sudo k3s ctr images ls | grep face-recognition
            '"""
            
            result = subprocess.run(import_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to import Docker image: {result.stderr}")
                return False
            
            print("Model service deployed successfully!")
            return True
            
        except Exception as e:
            print(f"Error deploying model service: {e}")
            return False
    
    def deploy_inference_app(self, model_version, docker_image_tar):
        """Deploy inference application to K3s"""
        print(f"Deploying inference app v{model_version} to Jetson K3s...")
        
        try:
            # Copy Docker image
            scp_cmd = f"scp {docker_image_tar} {self.jetson_user}@{self.jetson_ip}:~/"
            subprocess.run(scp_cmd, shell=True, check=True)
            
            # Import image to K3s
            import_cmd = f"""ssh {self.jetson_user}@{self.jetson_ip} '
                sudo k3s ctr images import {docker_image_tar} &&
                sudo k3s ctr images ls | grep face-recognition-inference
            '"""
            
            result = subprocess.run(import_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to import Docker image: {result.stderr}")
                return False
            
            print("Inference application deployed successfully!")
            return True
            
        except Exception as e:
            print(f"Error deploying inference app: {e}")
            return False
    
    def apply_k8s_manifests(self, manifests_dir):
        """Apply K8s manifests to Jetson cluster"""
        try:
            # Copy manifests
            scp_cmd = f"scp -r {manifests_dir} {self.jetson_user}@{self.jetson_ip}:~/"
            subprocess.run(scp_cmd, shell=True, check=True)
            
            # Apply manifests
            apply_cmd = f"""ssh {self.jetson_user}@{self.jetson_ip} '
                cd k8s_deployment &&
                kubectl apply -f model-service.yaml &&
                kubectl apply -f inference-app.yaml &&
                kubectl get pods
            '"""
            
            result = subprocess.run(apply_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("K8s manifests applied successfully!")
                print(result.stdout)
                return True
            else:
                print(f"Failed to apply manifests: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error applying K8s manifests: {e}")
            return False
    
    def get_service_status(self):
        """Get status of deployed services"""
        try:
            status_cmd = f"""ssh {self.jetson_user}@{self.jetson_ip} '
                kubectl get pods,services -o wide &&
                echo "--- Service URLs ---" &&
                kubectl get service face-recognition-model-service -o jsonpath="{{.spec.clusterIP}}" &&
                echo " (Model Service)" &&
                kubectl get service face-recognition-inference-service -o jsonpath="{{.spec.ports[0].nodePort}}" &&
                echo " (Inference App NodePort)"
            '"""
            
            result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else result.stderr
            
        except Exception as e:
            return f"Error getting service status: {e}"


if __name__ == "__main__":
    # Setup configurations
    print("Setting up MLflow configurations for K8s deployment...")
    
    # Setup Edge MLflow
    edge_uri = MLflowConfig.setup_edge_mlflow()
    
    # Create scripts
    MLflowConfig.create_mlflow_ui_script()
    MLflowConfig.create_k8s_build_script()
    
    # Initialize model registry
    registry = ModelRegistry(edge_uri)
    
    print("\nMLflow setup complete!")
    print("Available commands:")
    print("  ./mlflow_ui.sh edge     - Start MLflow UI")
    print("  ./build_k8s.sh <version> <jetson_ip> - Build and package for K8s deployment")
    print("  python mlflow_config.py - Initialize MLflow")
    
    # Demo: List models
    print("\nRegistered models:")
    registry.list_models_for_k8s()