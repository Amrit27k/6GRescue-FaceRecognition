# mlflow_plugins/jetson_deployment/simple_plugin.py

import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

from .config import JetsonConfig
from .ssh_manager import SSHManager
from .simple_package_builder import SimplePackageBuilder

logger = logging.getLogger(__name__)

class SimpleJetsonDeploymentTarget(BaseDeploymentClient):
    """
    Simple MLflow deployment plugin that just transfers files to Jetson.
    No Docker, no containers - just file transfer.
    """
    
    def __init__(self, target_uri: str):
        """Initialize simple file transfer plugin."""
        super().__init__(target_uri)
        self.config = JetsonConfig.from_uri(target_uri)
        self.config.validate()
        
        # Initialize components
        self.ssh_manager = SSHManager(self.config)
        self.package_builder = SimplePackageBuilder(self.config)
        
        logger.info(f"Simple Jetson deployment initialized for {self.config.device_ip}")
    
    def create_deployment(self, name: str, model_uri: str, flavor: str = None, 
                         config: Dict[str, Any] = None, endpoint: str = None) -> Dict[str, Any]:
        """
        Simple deployment: just transfer files to Jetson.
        
        Steps:
        1. Download model from MLflow
        2. Collect all face recognition files
        3. Create simple package
        4. Transfer to Jetson
        5. Extract files (but don't run anything)
        """
        deployment_config = config or {}
        
        try:
            logger.info(f"Starting simple file transfer deployment: '{name}'")
            
            # Step 1: Download model from MLflow
            logger.info("Step 1: Downloading model from MLflow...")
            model_info = self._download_model(model_uri)
            
            # Step 2: Create package with all files
            logger.info("Step 2: Creating file package...")
            package_info = self.package_builder.build_simple_package(
                model_path=model_info["local_path"],
                deployment_name=name,
                model_uri=model_uri,
                config=deployment_config
            )
            
            # Step 3: Transfer to Jetson
            logger.info("Step 3: Transferring files to Jetson...")
            transfer_info = self._transfer_files(package_info, name)
            
            # Step 4: Extract files on Jetson
            logger.info("Step 4: Extracting files on Jetson...")
            extraction_info = self._extract_files(transfer_info, name)
            
            deployment_result = {
                "name": name,
                "model_uri": model_uri,
                "flavor": flavor or "pytorch",
                "status": "files_transferred",
                "deployment_type": "file_transfer_only",
                "jetson_path": extraction_info["deployment_path"],
                "transferred_files": package_info["included_files"],
                "created_at": extraction_info["timestamp"],
                "config": deployment_config,
                "size_mb": package_info["size_mb"]
            }
            
            logger.info(f"✅ Files successfully transferred to Jetson at: {extraction_info['deployment_path']}")
            logger.info(f"📁 Transferred {len(package_info['included_files'])} files ({package_info['size_mb']:.2f} MB)")
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"File transfer failed: {str(e)}")
            # Cleanup on failure
            self._cleanup_failed_transfer(name)
            raise MlflowException(f"Failed to transfer files to Jetson: {str(e)}")
    
    def update_deployment(self, name: str, model_uri: str = None, 
                         flavor: str = None, config: Dict[str, Any] = None, 
                         endpoint: str = None) -> Dict[str, Any]:
        """Update files on Jetson with new model version."""
        logger.info(f"Updating files for deployment '{name}'")
        
        try:
            # Remove old files
            self._cleanup_deployment_files(name)
            
            # Transfer new files
            return self.create_deployment(name, model_uri, flavor, config, endpoint)
            
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            raise MlflowException(f"Failed to update deployment: {str(e)}")
    
    def delete_deployment(self, name: str, config: Dict[str, Any] = None, 
                         endpoint: str = None) -> None:
        """Remove files from Jetson device."""
        try:
            logger.info(f"Removing files for deployment '{name}'")
            self._cleanup_deployment_files(name)
            logger.info(f"✅ Files removed successfully")
            
        except Exception as e:
            logger.error(f"Failed to remove files: {str(e)}")
            raise MlflowException(f"Failed to delete deployment: {str(e)}")
    
    def list_deployments(self, endpoint: str = None) -> List[Dict[str, Any]]:
        """List all file deployments on Jetson."""
        try:
            return self._list_file_deployments()
        except Exception as e:
            logger.error(f"Failed to list deployments: {str(e)}")
            return []
    
    def get_deployment(self, name: str, endpoint: str = None) -> Dict[str, Any]:
        """Get information about a specific deployment."""
        try:
            deployments = self.list_deployments()
            
            for deployment in deployments:
                if deployment["name"] == name:
                    # Add current file status
                    file_status = self._check_deployment_files(name)
                    deployment.update({"file_status": file_status})
                    return deployment
            
            raise MlflowException(f"Deployment '{name}' not found")
            
        except Exception as e:
            logger.error(f"Failed to get deployment info: {str(e)}")
            raise MlflowException(f"Failed to get deployment: {str(e)}")
    
    def predict(self, deployment_name: str = None, inputs: Any = None, endpoint: str = None) -> Any:
        """
        REQUIRED abstract method for MLflow BaseDeploymentClient.
        
        Since this is a file transfer plugin, prediction is not supported.
        This method exists only to satisfy the abstract base class requirement.
        """
        raise MlflowException(
            "Prediction is not supported by the Jetson file transfer plugin. "
            "This plugin only transfers files to Jetson devices. "
            "For predictions, SSH to the Jetson device and run the inference server manually."
        )
    
    def _download_model(self, model_uri: str) -> Dict[str, Any]:
        """Download model from MLflow registry."""
        try:
            temp_dir = tempfile.mkdtemp(prefix="mlflow_simple_")
            local_path = _download_artifact_from_uri(model_uri, temp_dir)
            
            return {
                "model_uri": model_uri,
                "local_path": local_path,
                "temp_dir": temp_dir
            }
            
        except Exception as e:
            logger.error(f"Model download failed: {str(e)}")
            raise Exception(f"Failed to download model from {model_uri}: {str(e)}")
    
    def _transfer_files(self, package_info: Dict[str, Any], deployment_name: str) -> Dict[str, Any]:
        """Transfer package to Jetson."""
        try:
            # Create remote deployment directory
            remote_deployment_path = f"{self.config.deployment_base_path}/{deployment_name}"
            self.ssh_manager.execute_command(f"mkdir -p {remote_deployment_path}")
            
            # Transfer package file
            remote_package_path = f"{remote_deployment_path}/files_package.tar.gz"
            self.ssh_manager.transfer_file(
                local_path=package_info["package_path"],
                remote_path=remote_package_path
            )
            
            return {
                "remote_package_path": remote_package_path,
                "remote_deployment_path": remote_deployment_path,
                "package_size_mb": package_info["size_mb"]
            }
            
        except Exception as e:
            logger.error(f"File transfer failed: {str(e)}")
            raise Exception(f"Failed to transfer files: {str(e)}")
    
    def _extract_files(self, transfer_info: Dict[str, Any], deployment_name: str) -> Dict[str, Any]:
        """Extract files on Jetson device."""
        try:
            # Extract package
            extract_cmd = f"cd {transfer_info['remote_deployment_path']} && tar -xzf files_package.tar.gz"
            result = self.ssh_manager.execute_command(extract_cmd)
            
            if not result["success"]:
                raise Exception(f"Extraction failed: {result['stderr']}")
            
            # Set permissions
            chmod_cmd = f"find {transfer_info['remote_deployment_path']} -type f -name '*.py' -exec chmod +x {{}} +"
            self.ssh_manager.execute_command(chmod_cmd)
            
            # Remove the tar file
            cleanup_cmd = f"rm -f {transfer_info['remote_package_path']}"
            self.ssh_manager.execute_command(cleanup_cmd)
            
            return {
                "deployment_path": transfer_info["remote_deployment_path"],
                "timestamp": datetime.now().isoformat(),
                "status": "extracted"
            }
            
        except Exception as e:
            logger.error(f"File extraction failed: {str(e)}")
            raise Exception(f"Failed to extract files: {str(e)}")
    
    def _list_file_deployments(self) -> List[Dict[str, Any]]:
        """List all file deployments."""
        try:
            # Find all deployment directories
            find_cmd = f"find {self.config.deployment_base_path} -maxdepth 1 -type d -not -path {self.config.deployment_base_path}"
            result = self.ssh_manager.execute_command(find_cmd)
            
            if not result["success"]:
                return []
            
            deployments = []
            deployment_dirs = result["stdout"].split('\n')
            
            for deployment_dir in deployment_dirs:
                if deployment_dir.strip():
                    deployment_name = Path(deployment_dir).name
                    
                    # Check if deployment has files
                    file_status = self._check_deployment_files(deployment_name)
                    
                    deployments.append({
                        "name": deployment_name,
                        "path": deployment_dir,
                        "file_count": file_status["file_count"],
                        "has_model": file_status["has_model"],
                        "has_face_files": file_status["has_face_files"],
                        "status": "files_present" if file_status["file_count"] > 0 else "empty"
                    })
            
            return deployments
            
        except Exception as e:
            logger.error(f"Failed to list deployments: {str(e)}")
            return []
    
    def _check_deployment_files(self, deployment_name: str) -> Dict[str, Any]:
        """Check what files are present in a deployment."""
        try:
            deployment_path = f"{self.config.deployment_base_path}/{deployment_name}"
            
            # Count total files
            count_cmd = f"find {deployment_path} -type f | wc -l"
            count_result = self.ssh_manager.execute_command(count_cmd)
            file_count = int(count_result["stdout"]) if count_result["success"] else 0
            
            # Check for specific file types
            checks = {
                "has_model": f"find {deployment_path} -name '*.pkl' -o -name '*.pth' -o -name '*.pt' | head -1",
                "has_face_features": f"find {deployment_path} -name 'face_features.pkl' | head -1",
                "has_face_database": f"find {deployment_path} -name 'face_database.json' | head -1",
                "has_model_params": f"find {deployment_path} -name 'model_params.json' | head -1",
                "has_inference_script": f"find {deployment_path} -name 'inference_server.py' | head -1"
            }
            
            file_status = {"file_count": file_count}
            
            for check_name, check_cmd in checks.items():
                result = self.ssh_manager.execute_command(check_cmd)
                file_status[check_name] = bool(result["success"] and result["stdout"].strip())
            
            # Summary check
            file_status["has_face_files"] = all([
                file_status.get("has_face_features", False),
                file_status.get("has_face_database", False),
                file_status.get("has_model_params", False)
            ])
            
            return file_status
            
        except Exception as e:
            logger.warning(f"Failed to check deployment files: {str(e)}")
            return {"file_count": 0, "error": str(e)}
    
    def _cleanup_deployment_files(self, deployment_name: str) -> None:
        """Remove deployment files from Jetson."""
        try:
            deployment_path = f"{self.config.deployment_base_path}/{deployment_name}"
            cleanup_cmd = f"rm -rf {deployment_path}"
            self.ssh_manager.execute_command(cleanup_cmd)
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
    
    def _cleanup_failed_transfer(self, deployment_name: str) -> None:
        """Cleanup after failed transfer."""
        try:
            logger.info(f"Cleaning up failed transfer for {deployment_name}")
            self._cleanup_deployment_files(deployment_name)
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")


# REQUIRED MLflow Plugin Functions
def run_local(target, name, model_uri, flavor=None, config=None, endpoint=None):
    """
    Entry point for MLflow CLI deployment.
    This function is REQUIRED by MLflow plugin interface.
    """
    plugin = SimpleJetsonDeploymentTarget(target)
    return plugin.create_deployment(name, model_uri, flavor, config, endpoint)


def target_help():
    """
    Help text for the plugin target.
    This function is REQUIRED by MLflow plugin interface.
    """
    return """
Simple Jetson File Transfer Plugin

This plugin transfers MLflow models and face recognition files to Jetson devices
without running them. It's a simple file transfer utility.

Target URI formats:
  jetson://192.168.2.100              # Direct IP address
  jetson://simple_jetson.yaml         # Configuration file

Usage examples:

1. Transfer files using MLflow CLI:
   mlflow deployments create \\
     -t jetson://192.168.2.100 \\
     -m models:/face_recognition_model/latest \\
     --name face_recognition_files

2. Transfer files using Python API:
   from mlflow.deployments import get_deploy_client
   
   client = get_deploy_client("jetson://192.168.2.100")
   deployment = client.create_deployment(
       name="face_recognition_files",
       model_uri="models:/face_recognition_model/latest"
   )

Files transferred:
- MLflow model files (from registry)
- face_features.pkl, face_database.json, model_params.json
- inference_server.py, client.py
- Dockerfile.inference-server, Dockerfile.model-server

Files are extracted to: /home/newcastleuni/mlflow_deployments/{deployment_name}/

Note: This plugin only transfers files. Manual execution required on Jetson.
Prediction is not supported - SSH to Jetson to run inference manually.
"""