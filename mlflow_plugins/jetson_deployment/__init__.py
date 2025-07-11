"""
Simple MLflow Jetson Deployment Plugin
"""

__version__ = "0.1.0"

# Import the main class and required functions
from .simple_plugin import SimpleJetsonDeploymentTarget

# Import required functions and make them available at module level
from .simple_plugin import run_local, target_help

# Expose everything MLflow needs
__all__ = [
    'SimpleJetsonDeploymentTarget',
    'run_local', 
    'target_help'
]

# Define the deployment target function for plugin discovery
def get_deployment_target():
    return SimpleJetsonDeploymentTarget