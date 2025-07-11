from setuptools import setup, find_packages

setup(
    name="mlflow-jetson-simple-plugin",
    version="0.1.0",
    description="Simple MLflow plugin for transferring files to Jetson devices",
    packages=find_packages(),
    install_requires=[
        "mlflow>=2.0.0",
        "paramiko>=2.7.0",
        "scp>=0.14.0",
        "pyyaml>=5.4.0"
    ],
    entry_points={
        "mlflow.deployments": [
            "jetson=jetson_deployment"
        ]
    },
    python_requires=">=3.7",
)