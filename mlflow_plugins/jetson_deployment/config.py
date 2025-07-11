# mlflow_plugins/jetson_deployment/config.py

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

@dataclass
class JetsonConfig:
    """Configuration for Jetson device deployment."""
    device_ip: str
    username: str = "newcastleuni"  # Updated default username
    ssh_key_path: Optional[str] = None
    password: Optional[str] = None
    deployment_base_path: str = "/home/newcastleuni/mlflow_deployments"  # Updated default path
    timeout: int = 120
    max_retries: int = 3
    
    @classmethod
    def from_uri(cls, target_uri: str) -> 'JetsonConfig':
        """
        Create config from MLflow target URI.
        
        Supported formats:
        - jetson://192.168.2.100
        - jetson://simple_jetson.yaml
        - jetson://config_file.yaml
        """
        parsed = urlparse(target_uri)
        
        if parsed.scheme != "jetson":
            raise ValueError(f"Invalid scheme. Expected 'jetson', got '{parsed.scheme}'")
        
        identifier = parsed.netloc
        
        # Check if it's a config file
        if identifier.endswith('.yaml') or identifier.endswith('.yml'):
            return cls.from_config_file(identifier)
        
        # Assume it's an IP address - create default config
        return cls(device_ip=identifier)
    
    @classmethod
    def from_config_file(cls, config_name: str) -> 'JetsonConfig':
        """Load configuration from YAML file."""
        # Look for config file in deployment_configs directory
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "deployment_configs" / config_name
        
        # If not found, try current directory
        if not config_path.exists():
            config_path = Path(config_name)
        
        # If still not found, try absolute path
        if not config_path.exists():
            config_path = Path(config_name).expanduser().resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_name}\n"
                f"Searched in:\n"
                f"  - {project_root}/deployment_configs/{config_name}\n"
                f"  - {Path(config_name).resolve()}\n"
                f"  - {Path(config_name).expanduser().resolve()}"
            )
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Expand SSH key path if it starts with ~
        if config_data.get('ssh_key_path', '').startswith('~'):
            config_data['ssh_key_path'] = os.path.expanduser(config_data['ssh_key_path'])
        
        return cls(**config_data)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.device_ip:
            raise ValueError("device_ip is required")
        
        # Validate IP format (basic check)
        ip_parts = self.device_ip.split('.')
        if len(ip_parts) == 4:
            try:
                for part in ip_parts:
                    int_part = int(part)
                    if not 0 <= int_part <= 255:
                        raise ValueError(f"Invalid IP address: {self.device_ip}")
            except ValueError:
                # Could be hostname instead of IP
                pass
        
        # Check SSH authentication
        if not self.ssh_key_path and not self.password:
            # Try to find default SSH key
            default_key_paths = [
                "~/.ssh/jetson_key",
                "~/.ssh/id_rsa",
                "~/.ssh/id_ed25519"
            ]
            
            for key_path in default_key_paths:
                expanded_path = Path(key_path).expanduser()
                if expanded_path.exists():
                    self.ssh_key_path = str(expanded_path)
                    break
            
            if not self.ssh_key_path:
                raise ValueError(
                    "Either ssh_key_path or password must be provided.\n"
                    "No default SSH key found in ~/.ssh/\n"
                    "Please either:\n"
                    "  1. Set ssh_key_path in config\n"
                    "  2. Set password in config\n"
                    "  3. Create SSH key: ssh-keygen -t rsa -f ~/.ssh/jetson_key"
                )
        
        # Validate SSH key exists
        if self.ssh_key_path:
            key_path = Path(self.ssh_key_path).expanduser()
            if not key_path.exists():
                raise FileNotFoundError(
                    f"SSH key not found: {self.ssh_key_path}\n"
                    f"Expanded path: {key_path}\n"
                    f"Please create SSH key or update ssh_key_path in config"
                )
        
        # Validate timeout
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        
        # Validate max_retries
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "device_ip": self.device_ip,
            "username": self.username,
            "ssh_key_path": self.ssh_key_path,
            "password": self.password,
            "deployment_base_path": self.deployment_base_path,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
    
    def __str__(self) -> str:
        """String representation (hide password)."""
        config_dict = self.to_dict()
        if config_dict["password"]:
            config_dict["password"] = "***hidden***"
        return f"JetsonConfig({config_dict})"