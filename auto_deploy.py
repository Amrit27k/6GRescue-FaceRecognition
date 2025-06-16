"""
Automatic deployment script for Jetson k3s
This script automatically deploys the face recognition system to k3s
"""
import subprocess
import time
import json
import os
import sys
import yaml

class JetsonAutoDeployer:
    def __init__(self, model_version="1"):
        self.model_version = model_version
        self.namespace = "face-recognition"
        self.model_image = f"face-recognition-lightweight:v{model_version}"
        self.inference_image = f"face-recognition-inference:v{model_version}"
        
    def run_command(self, cmd, capture_output=True):
        """Run shell command and return result"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def check_prerequisites(self):
        """Check if k3s and required tools are available"""
        print("🔍 Checking prerequisites...")
        
        # Check k3s
        success, _, _ = self.run_command("sudo kubectl version --client")
        print(self.run_command("sudo kubectl version --client"))
        if not success:
            print("❌ kubectl not found. Please install k3s first.")
            return False
        
        # Check if k3s cluster is running
        success, _, _ = self.run_command("sudo kubectl get nodes")
        if not success:
            print("❌ k3s cluster not accessible. Please check k3s status.")
            return False
        
        # Check Docker
        success, _, _ = self.run_command("sudo docker version")
        if not success:
            print("❌ Docker not accessible. Please check Docker installation.")
            return False
        
        print("✅ Prerequisites check passed")
        return True
    
    def create_namespace(self):
        """Create namespace for face recognition system"""
        print(f"📁 Creating namespace: {self.namespace}")
        
        namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.namespace}
  labels:
    name: {self.namespace}
"""
        
        with open("namespace.yaml", "w") as f:
            f.write(namespace_yaml)
        
        success, _, _ = self.run_command(f"sudo kubectl apply -f namespace.yaml")
        if success:
            print(f"✅ Namespace {self.namespace} created/updated")
        else:
            print(f"⚠️  Namespace creation failed, continuing anyway...")
        
        # Clean up
        os.remove("namespace.yaml")
        return True
    
    def load_docker_images(self):
        """Load Docker images into k3s"""
        print("📦 Loading Docker images into k3s...")
        
        # Check if model image exists
        model_tar = f"face-recognition-lightweight-v{self.model_version}.tar"
        if os.path.exists(model_tar):
            print(f"Loading model image from {model_tar}...")
            success, _, error = self.run_command(f"sudo k3s ctr images import {model_tar}")
            if success:
                print("✅ Model image loaded successfully")
            else:
                print(f"❌ Failed to load model image: {error}")
                return False
        else:
            print(f"⚠️  Model image tar not found: {model_tar}")
            # Check if image already exists in k3s
            success, output, _ = self.run_command(f"sudo k3s ctr images ls | grep face-recognition-lightweight")
            if not success:
                print(f"❌ Model image not found. Please build the image first.")
                return False
            else:
                print("✅ Model image already exists in k3s")
        
        # Check if inference image exists
        inference_tar = f"face-recognition-inference-v{self.model_version}.tar"
        if os.path.exists(inference_tar):
            print(f"Loading inference image from {inference_tar}...")
            success, _, error = self.run_command(f"sudo k3s ctr images import {inference_tar}")
            if success:
                print("✅ Inference image loaded successfully")
            else:
                print(f"❌ Failed to load inference image: {error}")
                return False
        else:
            print(f"ℹ️  Inference image tar not found: {inference_tar}")
            print("Will build inference image from Dockerfile...")
            
            if os.path.exists("Dockerfile.inference"):
                success, _, error = self.run_command(f"sudo docker build -t {self.inference_image} -f Dockerfile.inference .")
                if success:
                    print("✅ Inference image built successfully")
                else:
                    print(f"❌ Failed to build inference image: {error}")
                    return False
            else:
                print("❌ Dockerfile.inference not found")
                return False
        
        return True
    
    def deploy_model_service(self):
        """Deploy model service pod"""
        print("🚀 Deploying model service...")
        
        model_service_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition-model
  namespace: {self.namespace}
  labels:
    app: face-recognition-model
    version: v{self.model_version}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: face-recognition-model
  template:
    metadata:
      labels:
        app: face-recognition-model
        version: v{self.model_version}
    spec:
      containers:
      - name: model-server
        image: {self.model_image}
        imagePullPolicy: Never
        ports:
        - containerPort: 5000
          name: http
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /ping
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        livenessProbe:
          httpGet:
            path: /ping
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: face-recognition-model-service
  namespace: {self.namespace}
  labels:
    app: face-recognition-model
spec:
  selector:
    app: face-recognition-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
    name: http
  type: ClusterIP
"""
        
        with open("model-service.yaml", "w") as f:
            f.write(model_service_yaml)
        
        success, _, error = self.run_command(f"sudo kubectl apply -f model-service.yaml")
        if success:
            print("✅ Model service deployed successfully")
        else:
            print(f"❌ Failed to deploy model service: {error}")
            return False
        
        # Clean up
        os.remove("model-service.yaml")
        return True
    
    def deploy_inference_app(self):
        """Deploy inference application pod"""
        print("🎥 Deploying inference application...")
        
        # Get Jetson IP for NodePort service
        success, jetson_ip, _ = self.run_command("hostname -I | awk '{print $1}'")
        jetson_ip = jetson_ip.strip() if success else "localhost"
        
        inference_app_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition-inference
  namespace: {self.namespace}
  labels:
    app: face-recognition-inference
    version: v{self.model_version}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: face-recognition-inference
  template:
    metadata:
      labels:
        app: face-recognition-inference
        version: v{self.model_version}
    spec:
      containers:
      - name: inference-app
        image: {self.inference_image}
        imagePullPolicy: Never
        ports:
        - containerPort: 8080
          name: web
        env:
        - name: MODEL_SERVICE_URL
          value: "http://face-recognition-model-service.{self.namespace}.svc.cluster.local"
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: camera-device
          mountPath: /dev/video0
        - name: output-storage
          mountPath: /app/output
        securityContext:
          privileged: true
      volumes:
      - name: camera-device
        hostPath:
          path: /dev/video0
          type: CharDevice
      - name: output-storage
        hostPath:
          path: /tmp/face-recognition-output
          type: DirectoryOrCreate
      nodeSelector:
        kubernetes.io/arch: arm64
---
apiVersion: v1
kind: Service
metadata:
  name: face-recognition-inference-service
  namespace: {self.namespace}
  labels:
    app: face-recognition-inference
spec:
  selector:
    app: face-recognition-inference
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
    nodePort: 30080
    name: web
  type: NodePort
"""
        
        with open("inference-app.yaml", "w") as f:
            f.write(inference_app_yaml)
        
        success, _, error = self.run_command(f"sudo kubectl apply -f inference-app.yaml")
        if success:
            print("✅ Inference application deployed successfully")
            print(f"🌐 Web interface will be available at: http://{jetson_ip}:30080")
        else:
            print(f"❌ Failed to deploy inference application: {error}")
            return False
        
        # Clean up
        os.remove("inference-app.yaml")
        return True
    
    def wait_for_pods(self):
        """Wait for pods to be ready"""
        print("⏳ Waiting for pods to be ready...")
        
        # Wait for model service
        for i in range(60):  # Wait up to 5 minutes
            success, output, _ = self.run_command(
                f"sudo kubectl get pods -n {self.namespace} -l app=face-recognition-model -o jsonpath='{{.items[0].status.phase}}'"
            )
            if success and output.strip() == "Running":
                print("✅ Model service pod is running")
                break
            time.sleep(5)
            print(f"   Waiting for model service... ({i*5}s)")
        else:
            print("❌ Model service pod failed to start")
            return False
        
        # Wait for inference app
        for i in range(60):  # Wait up to 5 minutes
            success, output, _ = self.run_command(
                f"sudo kubectl get pods -n {self.namespace} -l app=face-recognition-inference -o jsonpath='{{.items[0].status.phase}}'"
            )
            if success and output.strip() == "Running":
                print("✅ Inference application pod is running")
                break
            time.sleep(5)
            print(f"   Waiting for inference app... ({i*5}s)")
        else:
            print("❌ Inference application pod failed to start")
            return False
        
        return True
    
    def verify_deployment(self):
        """Verify that the deployment is working"""
        print("🔍 Verifying deployment...")
        
        # Get pod status
        success, output, _ = self.run_command(f"sudo kubectl get pods -n {self.namespace}")
        if success:
            print("Pod Status:")
            print(output)
        
        # Test model service health
        print("\n🩺 Testing model service health...")
        success, _, _ = self.run_command(
            f"sudo kubectl port-forward -n {self.namespace} service/face-recognition-model-service 8081:80 &"
        )
        
        if success:
            time.sleep(5)  # Wait for port-forward to establish
            success, output, _ = self.run_command("curl -s http://localhost:8081/ping")
            if success and "pong" in output:
                print("✅ Model service health check passed")
            else:
                print("⚠️  Model service health check failed")
            
            # Kill port-forward
            self.run_command("sudo pkill -f 'kubectl port-forward'")
        
        # Get service URLs
        success, jetson_ip, _ = self.run_command("hostname -I | awk '{print $1}'")
        jetson_ip = jetson_ip.strip() if success else "localhost"
        
        print(f"\n🌐 Access URLs:")
        print(f"   Web Interface: http://{jetson_ip}:30080")
        print(f"   Model API: http://face-recognition-model-service.{self.namespace}.svc.cluster.local (internal)")
        
        return True
    
    def create_management_scripts(self):
        """Create management scripts for the deployment"""
        print("📝 Creating management scripts...")
        
        # Status script
        status_script = f"""#!/bin/bash
# Face Recognition System Status

echo "🔍 Face Recognition System Status"
echo "================================="

echo "Namespace: {self.namespace}"
echo ""

echo "Pods:"
sudo kubectl get pods -n {self.namespace} -o wide

echo ""
echo "Services:"
sudo kubectl get services -n {self.namespace}

echo ""
echo "Resource Usage:"
sudo kubectl top pods -n {self.namespace} 2>/dev/null || echo "Metrics server not available"

echo ""
echo "Recent Events:"
sudo kubectl get events -n {self.namespace} --sort-by='.lastTimestamp' | tail -5

# Get access URL
JETSON_IP=$(hostname -I | awk '{{print $1}}')
echo ""
echo "🌐 Access URL: http://$JETSON_IP:30080"
"""
        
        with open("status.sh", "w") as f:
            f.write(status_script)
        os.chmod("status.sh", 0o755)
        
        # Logs script
        logs_script = f"""#!/bin/bash
# View logs for Face Recognition System

COMPONENT=${{1:-"model"}}

case $COMPONENT in
    "model"|"m")
        echo "📋 Model Service Logs:"
        kubectl logs -n {self.namespace} -l app=face-recognition-model -f
        ;;
    "inference"|"i")
        echo "📋 Inference App Logs:"
        kubectl logs -n {self.namespace} -l app=face-recognition-inference -f
        ;;
    "all"|"a")
        echo "📋 All Logs:"
        kubectl logs -n {self.namespace} -l app=face-recognition-model --tail=20
        echo ""
        kubectl logs -n {self.namespace} -l app=face-recognition-inference --tail=20
        ;;
    *)
        echo "Usage: ./logs.sh [model|inference|all]"
        echo "  model     - Show model service logs"
        echo "  inference - Show inference app logs" 
        echo "  all       - Show recent logs from both"
        ;;
esac
"""
        
        with open("logs.sh", "w") as f:
            f.write(logs_script)
        os.chmod("logs.sh", 0o755)
        
        # Cleanup script
        cleanup_script = f"""#!/bin/bash
# Cleanup Face Recognition System

echo "🧹 Cleaning up Face Recognition System..."

# Delete deployments
kubectl delete deployment face-recognition-model face-recognition-inference -n {self.namespace}

# Delete services  
kubectl delete service face-recognition-model-service face-recognition-inference-service -n {self.namespace}

# Delete namespace
kubectl delete namespace {self.namespace}

# Clean up Docker images (optional)
read -p "Remove Docker images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo docker rmi {self.model_image} {self.inference_image} 2>/dev/null || true
    sudo k3s ctr images rm {self.model_image} {self.inference_image} 2>/dev/null || true
fi

echo "✅ Cleanup complete"
"""
        
        with open("cleanup.sh", "w") as f:
            f.write(cleanup_script)
        os.chmod("cleanup.sh", 0o755)
        
        print("✅ Management scripts created:")
        print("   ./status.sh      - Check system status")
        print("   ./logs.sh <component> - View logs")
        print("   ./cleanup.sh     - Remove deployment")
    
    def deploy(self):
        """Main deployment function"""
        print("🚀 Starting Face Recognition System Deployment")
        print("=" * 50)
        
        if not self.check_prerequisites():
            return False
        
        if not self.create_namespace():
            return False
        
        if not self.load_docker_images():
            return False
        
        if not self.deploy_model_service():
            return False
        
        if not self.deploy_inference_app():
            return False
        
        if not self.wait_for_pods():
            print("❌ Deployment failed - pods not ready")
            return False
        
        if not self.verify_deployment():
            return False
        
        self.create_management_scripts()
        
        print("\n" + "=" * 50)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 50)
        
        # Get access URL
        success, jetson_ip, _ = self.run_command("hostname -I | awk '{print $1}'")
        jetson_ip = jetson_ip.strip() if success else "localhost"
        
        print(f"🌐 Web Interface: http://{jetson_ip}:30080")
        print(f"📊 System Status: ./status.sh")
        print(f"📋 View Logs: ./logs.sh")
        print(f"🧹 Cleanup: ./cleanup.sh")
        
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-deploy Face Recognition System to Jetson k3s")
    parser.add_argument("--model-version", default="1", help="Model version to deploy")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup existing deployment")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    
    args = parser.parse_args()
    
    deployer = JetsonAutoDeployer(model_version=args.model_version)
    
    if args.cleanup:
        print("🧹 Cleaning up existing deployment...")
        deployer.run_command(f"sudo kubectl delete namespace {deployer.namespace}")
        print("✅ Cleanup complete")
        return
    
    if args.status:
        deployer.run_command(f"sudo kubectl get all -n {deployer.namespace}")
        return
    
    # Run deployment
    success = deployer.deploy()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
