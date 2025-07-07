# mlflow_plugins/jetson_deployment/ssh_manager.py

import paramiko
import scp
import logging
import time
import requests
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from .config import JetsonConfig

logger = logging.getLogger(__name__)

class SSHManager:
    """Manages SSH connections and operations on Jetson device."""
    
    def __init__(self, config: JetsonConfig):
        self.config = config
        self._ssh_client = None
    
    def connect(self) -> paramiko.SSHClient:
        """Establish SSH connection to Jetson device."""
        if self._ssh_client is not None:
            try:
                # Test if connection is still alive
                self._ssh_client.exec_command('echo "test"', timeout=5)
                return self._ssh_client
            except:
                # Connection is dead, close it
                self._ssh_client.close()
                self._ssh_client = None
        
        # Create new connection
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            if self.config.ssh_key_path:
                logger.info(f"Connecting to {self.config.device_ip} using SSH key")
                self._ssh_client.connect(
                    hostname=self.config.device_ip,
                    username=self.config.username,
                    key_filename=self.config.ssh_key_path,
                    timeout=self.config.timeout
                )
            else:
                logger.info(f"Connecting to {self.config.device_ip} using password")
                self._ssh_client.connect(
                    hostname=self.config.device_ip,
                    username=self.config.username,
                    password=self.config.password,
                    timeout=self.config.timeout
                )
                
            logger.info("SSH connection established successfully")
            return self._ssh_client
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.device_ip}: {str(e)}")
            raise ConnectionError(f"SSH connection failed: {str(e)}")
    
    def transfer_file(self, local_path: str, remote_path: str) -> None:
        """
        Transfer file to Jetson device using SCP.
        This replaces your manual scp command.
        """
        logger.info(f"Transferring {local_path} to {remote_path}")
        
        ssh_client = self.connect()
        
        try:
            with scp.SCPClient(ssh_client.get_transport()) as scp_client:
                # Create remote directory if it doesn't exist
                remote_dir = str(Path(remote_path).parent)
                self.execute_command(f"mkdir -p {remote_dir}")
                
                # Transfer file
                scp_client.put(local_path, remote_path)
                
            logger.info(f"File transferred successfully to {remote_path}")
            
        except Exception as e:
            logger.error(f"File transfer failed: {str(e)}")
            raise Exception(f"SCP transfer failed: {str(e)}")
    
    def execute_command(self, command: str, timeout: int = None) -> Dict[str, Any]:
        """Execute command on Jetson device."""
        timeout = timeout or self.config.timeout
        ssh_client = self.connect()
        
        logger.debug(f"Executing command: {command}")
        
        try:
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=timeout)
            
            # Wait for command completion
            exit_status = stdout.channel.recv_exit_status()
            
            # Read outputs
            stdout_text = stdout.read().decode('utf-8').strip()
            stderr_text = stderr.read().decode('utf-8').strip()
            
            result = {
                "exit_status": exit_status,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "command": command,
                "success": exit_status == 0
            }
            
            if exit_status != 0:
                logger.warning(f"Command failed with exit code {exit_status}: {stderr_text}")
            else:
                logger.debug(f"Command completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            raise Exception(f"Failed to execute command '{command}': {str(e)}")
    
    def create_deployment_directory(self, deployment_name: str) -> str:
        """Create deployment directory on Jetson device."""
        deployment_path = f"{self.config.deployment_base_path}/{deployment_name}"
        
        logger.info(f"Creating deployment directory: {deployment_path}")
        
        # Create directory structure
        commands = [
            f"mkdir -p {deployment_path}",
            f"mkdir -p {deployment_path}/models",
            f"mkdir -p {deployment_path}/logs", 
            f"mkdir -p {deployment_path}/scripts"
        ]
        
        for cmd in commands:
            result = self.execute_command(cmd)
            if not result["success"]:
                raise Exception(f"Failed to create directory: {result['stderr']}")
        
        return deployment_path
    
    def extract_deployment_package(self, package_path: str, extraction_path: str) -> None:
        """Extract deployment package on Jetson device."""
        logger.info(f"Extracting package {package_path} to {extraction_path}")
        
        extract_cmd = f"cd {extraction_path} && tar -xzf {package_path}"
        result = self.execute_command(extract_cmd)
        
        if not result["success"]:
            raise Exception(f"Package extraction failed: {result['stderr']}")
        
        # Set executable permissions
        chmod_cmd = f"find {extraction_path} -name '*.sh' -exec chmod +x {{}} +"
        self.execute_command(chmod_cmd)
        
        logger.info("Package extracted successfully")
    
    def check_service_health(self, port: int) -> Dict[str, Any]:
        """Check if service is running and responsive."""
        try:
            # Check if port is listening
            netstat_result = self.execute_command(f"netstat -tulpn | grep :{port}")
            port_listening = netstat_result["success"] and str(port) in netstat_result["stdout"]
            
            # Try HTTP health check
            health_url = f"http://{self.config.device_ip}:{port}/health"
            
            try:
                response = requests.get(health_url, timeout=10)
                http_responsive = response.status_code == 200
                response_time = response.elapsed.total_seconds()
            except:
                http_responsive = False
                response_time = None
            
            return {
                "port_listening": port_listening,
                "http_responsive": http_responsive,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
                "healthy": port_listening and http_responsive
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def cleanup_deployment(self, deployment_name: str) -> None:
        """Remove deployment files from Jetson device."""
        deployment_path = f"{self.config.deployment_base_path}/{deployment_name}"
        
        logger.info(f"Cleaning up deployment: {deployment_path}")
        
        cleanup_cmd = f"rm -rf {deployment_path}"
        result = self.execute_command(cleanup_cmd)
        
        if not result["success"]:
            logger.warning(f"Cleanup failed: {result['stderr']}")
        else:
            logger.info("Cleanup completed successfully")
    
    def close(self) -> None:
        """Close SSH connection."""
        if self._ssh_client:
            self._ssh_client.close()
            self._ssh_client = None
            logger.info("SSH connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()