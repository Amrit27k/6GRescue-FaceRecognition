import mlflow
import os
import sqlite3
from datetime import datetime

class MLflowConfig:
    """Configure MLflow for both Edge and Jetson environments"""
    
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
            "few_shot_learning"
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
            deployment_status TEXT
        )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"Edge MLflow configured with SQLite backend: {tracking_uri}")
        return tracking_uri
    
    @staticmethod
    def setup_jetson_mlflow(db_path="mlflow_jetson.db"):
        """Setup MLflow for Jetson Nano"""
        # Create MLflow directory structure
        mlflow_dir = "mlruns_jetson"
        os.makedirs(mlflow_dir, exist_ok=True)
        
        # Set tracking URI
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiments
        experiments = [
            "face_recognition_inference",
            "performance_benchmarks",
            "model_updates"
        ]
        
        for exp_name in experiments:
            try:
                mlflow.create_experiment(exp_name)
                print(f"Created experiment: {exp_name}")
            except:
                print(f"Experiment {exp_name} already exists")
        
        # Initialize database schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create custom metrics table for inference tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            fps REAL,
            inference_time_ms REAL,
            faces_detected INTEGER,
            recognized_faces INTEGER,
            unknown_faces INTEGER,
            cpu_usage REAL,
            memory_usage REAL,
            gpu_usage REAL
        )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"Jetson MLflow configured with SQLite backend: {tracking_uri}")
        return tracking_uri
    
    @staticmethod
    def sync_metrics(edge_db="mlflow_edge.db", jetson_db="mlflow_jetson.db"):
        """Sync metrics between Edge and Jetson (for monitoring)"""
        # This would typically be done over network
        # For now, we'll create a simple export/import mechanism
        
        sync_dir = "mlflow_sync"
        os.makedirs(sync_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export Edge metrics
        edge_conn = sqlite3.connect(edge_db)
        edge_cursor = edge_conn.cursor()
        
        edge_cursor.execute("SELECT * FROM runs")
        edge_runs = edge_cursor.fetchall()
        
        # Save to sync file
        import json
        sync_data = {
            "timestamp": timestamp,
            "edge_runs_count": len(edge_runs),
            "source": "edge",
            "runs": edge_runs
        }
        
        with open(os.path.join(sync_dir, f"edge_sync_{timestamp}.json"), 'w') as f:
            json.dump(sync_data, f, default=str)
        
        edge_conn.close()
        
        print(f"Exported {len(edge_runs)} runs from Edge MLflow")
        
    @staticmethod
    def create_mlflow_ui_script():
        """Create script to launch MLflow UI"""
        script_content = """#!/bin/bash
# Launch MLflow UI for Edge or Jetson

if [ "$1" == "edge" ]; then
    echo "Starting MLflow UI for Edge server..."
    mlflow ui --backend-store-uri sqlite:///mlflow_edge.db --port 5000
elif [ "$1" == "jetson" ]; then
    echo "Starting MLflow UI for Jetson Nano..."
    mlflow ui --backend-store-uri sqlite:///mlflow_jetson.db --port 5001
else
    echo "Usage: ./mlflow_ui.sh [edge|jetson]"
    echo "  edge   - Start MLflow UI for Edge server on port 5000"
    echo "  jetson - Start MLflow UI for Jetson Nano on port 5001"
fi
"""
        
        with open("mlflow_ui.sh", 'w') as f:
            f.write(script_content)
        
        os.chmod("mlflow_ui.sh", 0o755)
        print("Created mlflow_ui.sh script")


# Model Registry utilities
class ModelRegistry:
    """Manage model versions and deployments"""
    
    def __init__(self, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, run_id, model_name="face_recognition_model"):
        """Register a model from a run"""
        model_uri = f"runs:/{run_id}/model"
        
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
    
    def promote_model(self, model_name, version):
        """Promote model from staging to production"""
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
        
        print(f"Promoted model {model_name} version {version} to Production")
    
    def get_production_model(self, model_name):
        """Get current production model"""
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return versions[0]
        return None
    
    def list_models(self):
        """List all registered models"""
        models = self.client.search_registered_models()
        for model in models:
            print(f"\nModel: {model.name}")
            for version in model.latest_versions:
                print(f"  Version {version.version}: {version.current_stage}")


if __name__ == "__main__":
    # Setup configurations
    print("Setting up MLflow configurations...")
    
    # Setup Edge MLflow
    edge_uri = MLflowConfig.setup_edge_mlflow()
    
    # Setup Jetson MLflow
    jetson_uri = MLflowConfig.setup_jetson_mlflow()
    
    # Create UI launch script
    MLflowConfig.create_mlflow_ui_script()
    
    print("\nMLflow setup complete!")
    print("To view MLflow UI:")
    print("  Edge:   ./mlflow_ui.sh edge")
    print("  Jetson: ./mlflow_ui.sh jetson")