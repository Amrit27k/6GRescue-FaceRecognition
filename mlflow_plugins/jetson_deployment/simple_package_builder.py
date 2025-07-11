# mlflow_plugins/jetson_deployment/simple_package_builder.py

import os
import tarfile
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .config import JetsonConfig

logger = logging.getLogger(__name__)

class SimplePackageBuilder:
    """
    Simple package builder that just collects files without Docker configs.
    Only creates a tar.gz with model and face recognition files.
    """
    
    def __init__(self, config: JetsonConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        
        # Your face recognition files to look for
        self.face_recognition_files = {
            "face_features.pkl": None,
            "face_database.json": None,
            "model_params.json": None,
            "inference_server_rtsp.py": None,
            "model_server.py": None,
            "client.py": None,
            "Dockerfile.inference-server": None,
            "Dockerfile.model-server": None,
            "model_server_requirements.txt": None,
            "face_model_v2.pkl": None, 
            "label_encoder.pkl": None, 
            "random_forest_model.pkl": None,
            "amrit_test.jpg": None
        }
        
        self._locate_files()
    
    def _locate_files(self):
        """Find all face recognition files in your project."""
        logger.info("Locating face recognition files...")
        
        # Search in common locations
        search_paths = [
            self.project_root / "jetson",
            self.project_root / "models",
            self.project_root / "jetson/docker"
        ]
        
        for file_name in self.face_recognition_files.keys():
            for search_path in search_paths:
                potential_file = search_path / file_name
                if potential_file.exists():
                    self.face_recognition_files[file_name] = str(potential_file)
                    logger.info(f"✓ Found {file_name} at {potential_file}")
                    break
            
            if self.face_recognition_files[file_name] is None:
                logger.info(f"  {file_name} not found (will skip)")
    
    def build_simple_package(self, model_path: str, deployment_name: str, 
                           model_uri: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build simple package with just files - no Docker, no configs.
        
        Package structure:
        ├── models/          # MLflow model files
        ├── data/           # Face recognition files
        ├── scripts/        # Python scripts
        └── README.txt      # Simple info file
        """
        logger.info(f"Building simple file package for {deployment_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "package"
            package_dir.mkdir()
            
            # 1. Copy model files
            self._copy_model_files(model_path, package_dir / "models")
            
            # 2. Copy face recognition data files
            self._copy_face_recognition_data(package_dir / "data")
            
            # 3. Copy Python scripts
            self._copy_scripts(package_dir / "scripts")
            
            # 4. Copy Docker files (if they exist)
            self._copy_docker_files(package_dir / "docker")
            
            # 5. Create simple README
            self._create_readme(package_dir, deployment_name, model_uri, config)
            
            # 6. Create tar package
            package_path = self._create_tar_package(package_dir, deployment_name, temp_dir)
            
            return {
                "package_path": package_path,
                "deployment_name": deployment_name,
                "created_at": datetime.now().isoformat(),
                "size_mb": os.path.getsize(package_path) / (1024 * 1024),
                "included_files": self._get_file_list(package_dir)
            }
    
    def _copy_model_files(self, model_path: str, dest_dir: Path) -> None:
        """Copy model files from MLflow."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        model_source = Path(model_path)
        
        if model_source.is_file():
            shutil.copy2(model_source, dest_dir / model_source.name)
            logger.info(f"✓ Copied model file: {model_source.name}")
        elif model_source.is_dir():
            shutil.copytree(model_source, dest_dir / "mlflow_model", dirs_exist_ok=True)
            logger.info(f"✓ Copied model directory")
        else:
            logger.warning(f"Model path not found: {model_path}")
            # Create placeholder
            placeholder = dest_dir / "model_placeholder.txt"
            placeholder.write_text(f"Model not found at: {model_path}")
    
    def _copy_face_recognition_data(self, dest_dir: Path) -> None:
        """Copy face recognition data files."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        data_files = ["face_features.pkl", "face_database.json", "model_params.json", "face_model_v2.pkl", "label_encoder.pkl", "random_forest_model.pkl"]
        copied_count = 0
        
        for file_name in data_files:
            source_path = self.face_recognition_files.get(file_name)
            if source_path and Path(source_path).exists():
                dest_path = dest_dir / file_name
                shutil.copy2(source_path, dest_path)
                logger.info(f"✓ Copied {file_name}")
                copied_count += 1
            else:
                logger.info(f"  Skipped {file_name} (not found)")
                # Create placeholder
                placeholder = dest_dir / f"{file_name}.placeholder"
                placeholder.write_text(f"Original file not found: {file_name}")
        
        logger.info(f"Copied {copied_count}/3 face recognition data files")
    
    def _copy_scripts(self, dest_dir: Path) -> None:
        """Copy Python scripts."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        script_files = ["inference_server_rtsp.py", "client.py", "model_server.py", "amrit_test.jpg", ]
        copied_count = 0
        
        for script_name in script_files:
            source_path = self.face_recognition_files.get(script_name)
            if source_path and Path(source_path).exists():
                dest_path = dest_dir / script_name
                shutil.copy2(source_path, dest_path)
                # Make executable
                dest_path.chmod(0o755)
                logger.info(f"✓ Copied {script_name}")
                copied_count += 1
            else:
                logger.info(f"  Skipped {script_name} (not found)")
        
        logger.info(f"Copied {copied_count}/2 Python scripts")
    
    def _copy_docker_files(self, dest_dir: Path) -> None:
        """Copy Docker files if they exist."""
        docker_files = ["Dockerfile.inference-server", "Dockerfile.model-server", "model_server_requirements.txt"]
        docker_files_found = []
        
        for docker_file in docker_files:
            source_path = self.face_recognition_files.get(docker_file)
            if source_path and Path(source_path).exists():
                if not dest_dir.exists():
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                dest_path = dest_dir / docker_file
                shutil.copy2(source_path, dest_path)
                docker_files_found.append(docker_file)
                logger.info(f"✓ Copied {docker_file}")
        
        if docker_files_found:
            logger.info(f"Copied {len(docker_files_found)} Docker files")
        else:
            logger.info("No Docker files found (skipping docker/ directory)")
    
    def _create_readme(self, package_dir: Path, deployment_name: str, 
                      model_uri: str, config: Dict[str, Any]) -> None:
        """Create simple README file."""
        readme_content = f"""Face Recognition Deployment Package
=====================================

Deployment Name: {deployment_name}
Model URI: {model_uri}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Plugin: MLflow Simple Jetson Transfer

Package Contents:
================

models/
  └── MLflow model files

data/
  ├── face_features.pkl     # Face feature embeddings
  ├── face_database.json    # Face database
  └── model_params.json     # Model parameters

scripts/
  ├── inference_server.py   # Inference server script
  └── client.py            # Client script
  └── model_server.py
  └── inference_server_stream.py # Inference server streaming with mjpeg and rtsp

docker/ (if present)
  ├── Dockerfile.inference-server
  └── Dockerfile.model-server
  └── Dockerfile.inference-server-stream

Usage on Jetson:
===============

1. Files are extracted to: {self.config.deployment_base_path}/{deployment_name}/

2. To run inference server manually:
   cd {self.config.deployment_base_path}/{deployment_name}/scripts/
   python3 inference_server.py

3. To use Docker (if Dockerfile present):
   cd {self.config.deployment_base_path}/{deployment_name}/docker/
   docker build -f Dockerfile.inference-server -t {deployment_name} .
   docker run -p 8080:8080 --device=/dev/video0 {deployment_name}

   From the mlflow_deployment directory:
   docker build -f face_recognition_files/docker/Dockerfile.inference-server-stream -t face-inference-mlf-plugin-stream:latest .
   docker run -d --name face-inference-mlf-plugin-stream --runtime=nvidia --network host -p 5001:5001 -p 8554:8554 -v $(pwd)/output:/app/output -v  $(pwd)/temp_frames:/app/temp_frames --volume /tmp/argus_socket:/tmp/argus_socket --volume /usr/src/tensorrt/data:/usr/src/tensorrt/data:ro --device /dev/nvhost-ctrl --device /dev/nvhost-ctrl-gpu --device /dev/nvhost-gpu --device /dev/nvhost-as-gpu --device /dev/nvhost-vic --device /dev/nvhost-msenc --device /dev/nvmap -v /dev:/dev --device /dev/video0 --privileged -e MODEL_SERVICE_URL=http://localhost:5000 face-inference-mlf-plugin-stream:latest
   
Configuration:
=============
{str(config) if config else 'No additional configuration'}

Note: This package only transfers files. No automatic startup or Docker deployment.
"""
        
        with open(package_dir / "README.txt", 'w') as f:
            f.write(readme_content)
        
        logger.info("✓ Created README.txt")
    
    def _get_file_list(self, package_dir: Path) -> List[str]:
        """Get list of all files in the package."""
        file_list = []
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(package_dir)
                file_list.append(str(relative_path))
        
        return sorted(file_list)
    
    def _create_tar_package(self, package_dir: Path, deployment_name: str, temp_dir: str) -> str:
        """Create tar.gz package in a persistent location."""
        # Create tar file in a location that won't be deleted immediately
        # Use tempfile.mkstemp to create a persistent temporary file
        import tempfile as tf
        
        try:
            # Create a persistent temporary file
            fd, tar_path = tf.mkstemp(suffix=f'_{deployment_name}_files.tar.gz', prefix='mlflow_jetson_')
            os.close(fd)  # Close the file descriptor, but keep the path
            
            tar_path = Path(tar_path)
            
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(package_dir, arcname='.')
            
            logger.info(f"✓ Created package: {tar_path}")
            logger.info(f"  Size: {os.path.getsize(tar_path) / (1024 * 1024):.2f} MB")
            
            # Verify file exists
            if not tar_path.exists():
                raise Exception(f"Package file was not created: {tar_path}")
                
            return str(tar_path)
            
        except Exception as e:
            logger.error(f"Failed to create tar package: {str(e)}")
            raise Exception(f"Package creation failed: {str(e)}")