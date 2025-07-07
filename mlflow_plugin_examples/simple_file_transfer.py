# examples/simple_file_transfer.py
"""
Simple MLflow plugin usage example.
Just transfers files to Jetson without running anything.
"""

import mlflow
from mlflow.deployments import get_deploy_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleJetsonFileTransfer:
    """Simple file transfer to Jetson using MLflow plugin."""
    
    def __init__(self, jetson_ip: str = "192.168.2.100", mlflow_uri: str = "http://localhost:5000"):
        """Initialize simple file transfer."""
        self.jetson_ip = jetson_ip
        self.mlflow_uri = mlflow_uri
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Get simple deployment client
        self.deploy_client = get_deploy_client(f"jetson://{self.jetson_ip}")
        
        logger.info(f"Simple file transfer initialized for Jetson: {jetson_ip}")
    
    def transfer_face_recognition_files(self, model_name: str = "face_recognition_model", 
                                      version: str = "latest") -> dict:
        """
        Transfer face recognition files to Jetson.
        
        This will transfer:
        - Model from MLflow registry
        - face_features.pkl
        - face_database.json  
        - model_params.json
        - inference_server.py
        - client.py
        - Dockerfiles (if present)
        
        But WON'T run anything - just copies files.
        """
        logger.info(f"📁 Transferring face recognition files: {model_name}:{version}")
        
        try:
            deployment = self.deploy_client.create_deployment(
                name="face_recognition_files",
                model_uri=f"models:/{model_name}/{version}",
                config={
                    "transfer_only": True,
                    "deployment_type": "file_transfer",
                    "model_name": model_name
                }
            )
            
            logger.info("✅ File transfer completed!")
            logger.info(f"📍 Files location: {deployment['jetson_path']}")
            logger.info(f"📦 Transfer size: {deployment['size_mb']:.2f} MB")
            logger.info(f"📄 Files transferred: {len(deployment['transferred_files'])}")
            
            # Show what files were transferred
            logger.info("📋 Transferred files:")
            for file_path in deployment['transferred_files']:
                logger.info(f"   ✓ {file_path}")
            
            return deployment
            
        except Exception as e:
            logger.error(f"❌ File transfer failed: {str(e)}")
            raise
    
    def check_transferred_files(self, deployment_name: str = "face_recognition_files") -> dict:
        """Check what files are on the Jetson."""
        logger.info(f"🔍 Checking files for: {deployment_name}")
        
        try:
            deployment = self.deploy_client.get_deployment(deployment_name)
            file_status = deployment.get("file_status", {})
            
            logger.info("📊 File Status:")
            logger.info(f"   📁 Total files: {file_status.get('file_count', 0)}")
            logger.info(f"   🤖 Has model: {'✓' if file_status.get('has_model') else '✗'}")
            logger.info(f"   👤 Has face features: {'✓' if file_status.get('has_face_features') else '✗'}")
            logger.info(f"   📊 Has face database: {'✓' if file_status.get('has_face_database') else '✗'}")
            logger.info(f"   ⚙️ Has model params: {'✓' if file_status.get('has_model_params') else '✗'}")
            logger.info(f"   🐍 Has inference script: {'✓' if file_status.get('has_inference_script') else '✗'}")
            logger.info(f"   📍 Jetson path: {deployment.get('path', 'unknown')}")
            
            return deployment
            
        except Exception as e:
            logger.error(f"❌ Status check failed: {str(e)}")
            raise
    
    def update_files(self, model_name: str = "face_recognition_model", 
                    version: str = "latest") -> dict:
        """Update files on Jetson with new version."""
        logger.info(f"🔄 Updating files with {model_name}:{version}")
        
        try:
            updated_deployment = self.deploy_client.update_deployment(
                name="face_recognition_files",
                model_uri=f"models:/{model_name}/{version}"
            )
            
            logger.info("✅ Files updated successfully!")
            return updated_deployment
            
        except Exception as e:
            logger.error(f"❌ File update failed: {str(e)}")
            raise
    
    def list_all_transfers(self) -> list:
        """List all file transfers on Jetson."""
        logger.info("📋 Listing all file transfers...")
        
        try:
            transfers = self.deploy_client.list_deployments()
            
            if not transfers:
                logger.info("   No file transfers found")
                return []
            
            logger.info(f"   Found {len(transfers)} file transfers:")
            for transfer in transfers:
                status = "✓" if transfer["status"] == "files_present" else "✗"
                logger.info(f"   {status} {transfer['name']} ({transfer['file_count']} files)")
            
            return transfers
            
        except Exception as e:
            logger.error(f"❌ Failed to list transfers: {str(e)}")
            return []
    
    def remove_files(self, deployment_name: str = "face_recognition_files") -> None:
        """Remove files from Jetson."""
        logger.info(f"🗑️ Removing files: {deployment_name}")
        
        try:
            self.deploy_client.delete_deployment(deployment_name)
            logger.info("✅ Files removed successfully!")
            
        except Exception as e:
            logger.error(f"❌ File removal failed: {str(e)}")
            raise


def main():
    """Simple file transfer workflow example."""
    print("=" * 60)
    print("🚀 Simple MLflow Jetson File Transfer")
    print("=" * 60)
    
    # Initialize file transfer (update with your IPs)
    transfer_manager = SimpleJetsonFileTransfer(
        jetson_ip="192.168.2.100",  # Your Jetson IP
        mlflow_uri="http://localhost:5000"  # Your MLflow server
    )
    
    try:
        # Step 1: Transfer files
        print("\n📁 Step 1: Transferring face recognition files...")
        deployment = transfer_manager.transfer_face_recognition_files()
        
        # Step 2: Check what was transferred
        print("\n🔍 Step 2: Checking transferred files...")
        status = transfer_manager.check_transferred_files()
        
        # Step 3: List all transfers
        print("\n📋 Step 3: Listing all file transfers...")
        all_transfers = transfer_manager.list_all_transfers()
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ FILE TRANSFER COMPLETE!")
        print("=" * 60)
        print(f"📍 Files location on Jetson: {deployment['jetson_path']}")
        print(f"📦 Transfer size: {deployment['size_mb']:.2f} MB")
        print(f"📄 Files transferred: {len(deployment['transferred_files'])}")
        print(f"📊 Status: {deployment['status']}")
        
        print("\n🔧 Next Steps:")
        print("1. SSH to your Jetson device:")
        print(f"   ssh newcastleuni@192.168.2.100")
        print("2. Navigate to files:")
        print(f"   cd {deployment['jetson_path']}")
        print("3. Check files:")
        print("   ls -la")
        print("4. Run inference server manually:")
        print("   cd scripts && python3 inference_server.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Transfer workflow failed: {str(e)}")
        return False


def example_update_workflow():
    """Example of updating files with new model version."""
    print("\n" + "=" * 60)
    print("🔄 File Update Example")
    print("=" * 60)
    
    transfer_manager = SimpleJetsonFileTransfer("192.168.2.100")
    
    try:
        # Update with new model version
        print("\n🔄 Updating files with new model version...")
        updated_deployment = transfer_manager.update_files(
            model_name="face_recognition_model", 
            version="2"  # Update to version 2
        )
        
        # Check updated files
        print("\n🔍 Checking updated files...")
        transfer_manager.check_transferred_files()
        
        print("\n✅ Files updated successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Update failed: {str(e)}")
        return False


def example_cleanup():
    """Example of removing files from Jetson."""
    print("\n" + "=" * 60)
    print("🗑️ Cleanup Example")
    print("=" * 60)
    
    transfer_manager = SimpleJetsonFileTransfer("192.168.2.100")
    
    try:
        # Remove files
        transfer_manager.remove_files("face_recognition_files")
        
        # Verify removal
        transfers = transfer_manager.list_all_transfers()
        print(f"📊 Remaining transfers: {len(transfers)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🎯 Starting Simple MLflow Jetson File Transfer Examples...")
    
    # Main transfer workflow
    success1 = main()
    
    if success1:
        print("\n" + "="*60)
        print("🔧 Additional Examples Available:")
        print("="*60)
        print("1. Update files: example_update_workflow()")
        print("2. Cleanup files: example_cleanup()")
        print("\nUncomment the lines below to run additional examples:")
        print("# example_update_workflow()")
        print("# example_cleanup()")
        
        # Uncomment these to run additional examples:
        # example_update_workflow()
        # example_cleanup()
    
    print("\n🎉 Examples completed!")
