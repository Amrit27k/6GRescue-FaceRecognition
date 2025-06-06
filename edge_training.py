import cv2
import numpy as np
import time
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import sqlite3
import json
import pickle
from datetime import datetime
import joblib

class EdgeFaceRecognitionTrainer:
    def __init__(self, images_dir="images", mlflow_uri="sqlite:///mlflow_edge.db"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("face_recognition_training")
        self.mlflow_client = MlflowClient()
        
        # Input images directory
        self.images_dir = images_dir
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=True)
            print(f"Created images directory: {self.images_dir}")
        
        # Create directories for few-shot learning
        self.few_shot_dir = "few_shot_examples"
        self.unknown_dir = os.path.join(self.few_shot_dir, "unknown")
        self.models_dir = "models"
        os.makedirs(self.few_shot_dir, exist_ok=True)
        os.makedirs(self.unknown_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load YOLOv8 model
        self.load_yolov8_model()
        
        # Face database for recognition
        self.face_database = self.load_face_database()
        
        # Feature extractor (foundation model)
        self.feature_extractor = self.load_feature_extractor()
        
        # Random Forest model for classification
        self.rf_model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        
        # Face features cache
        self.face_features = {}
        self.load_face_features()
        
        # Few-shot learning parameters
        self.confidence_threshold = 0.6  # Changed from similarity_threshold
        self.min_examples = 3
        
        # Random Forest parameters
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        # Model version tracking
        self.current_model_version = self.get_latest_model_version()
        
        # Load existing Random Forest model if available
        self.load_rf_model()

    def get_latest_model_version(self):
        """Get the latest model version from MLflow"""
        try:
            models = self.mlflow_client.search_registered_models(
                filter_string="name='face_recognition_model'"
            )
            if models:
                latest_version = max([
                    int(v.version) for model in models 
                    for v in model.latest_versions
                ])
                return latest_version
            return 0
        except:
            return 0

    def load_yolov8_model(self):
        """Load YOLOv8 model for face detection"""
        print("Loading YOLOv8 model...")
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n-face.pt")  # Use face-specific model if available
            self.model_type = "v8"
            print("Loaded YOLOv8 model successfully!")
        except Exception as e:
            print(f"YOLOv8 not available: {e}")
            print("Falling back to YOLOv5...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model_type = "v5"
            self.model.classes = [0]  # Person class only
            self.model.conf = 0.5
            print("Loaded YOLOv5 model successfully!")

    def load_feature_extractor(self):
        """Load foundation model for feature extraction"""
        print("Loading feature extraction model...")
        try:
            # Using MobileNetV3 for edge deployment
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=True)
            # Remove the last layer
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_extractor.to(self.device)
            feature_extractor.eval()
            print("Feature extraction model loaded successfully!")
            return feature_extractor
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
            return None

    def load_face_database(self):
        """Load face database from disk"""
        db_path = os.path.join(self.models_dir, "face_database.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    return json.load(f)
            except:
                print("Error loading face database. Creating new database.")
        return {}

    def load_face_features(self):
        """Load cached face features from disk"""
        features_path = os.path.join(self.models_dir, "face_features.pkl")
        if os.path.exists(features_path):
            try:
                with open(features_path, 'rb') as f:
                    self.face_features = pickle.load(f)
                print(f"Loaded features for {len(self.face_features)} identities.")
            except:
                print("Error loading face features. Starting with empty cache.")

    def load_rf_model(self):
        """Load Random Forest model from disk"""
        rf_path = os.path.join(self.models_dir, "random_forest_model.pkl")
        encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
        
        if os.path.exists(rf_path) and os.path.exists(encoder_path):
            try:
                self.rf_model = joblib.load(rf_path)
                with open(encoder_path, 'rb') as f:
                    encoders = pickle.load(f)
                    self.label_encoder = encoders['label_encoder']
                    self.reverse_label_encoder = encoders['reverse_label_encoder']
                print("Loaded Random Forest model successfully!")
            except Exception as e:
                print(f"Error loading Random Forest model: {e}")

    def save_face_database(self):
        """Save face database to disk"""
        db_path = os.path.join(self.models_dir, "face_database.json")
        with open(db_path, 'w') as f:
            json.dump(self.face_database, f)

    def save_face_features(self):
        """Save face features to disk"""
        features_path = os.path.join(self.models_dir, "face_features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(self.face_features, f)

    def save_rf_model(self):
        """Save Random Forest model to disk"""
        if self.rf_model is not None:
            rf_path = os.path.join(self.models_dir, "random_forest_model.pkl")
            encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
            
            joblib.dump(self.rf_model, rf_path)
            
            encoders = {
                'label_encoder': self.label_encoder,
                'reverse_label_encoder': self.reverse_label_encoder
            }
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoders, f)

    def extract_features(self, face_img):
        """Extract features from a face image using foundation model"""
        if self.feature_extractor is None:
            # Simplified feature extraction
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        
        # Preprocess image for the feature extractor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = transform(face_img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = features.squeeze().cpu().numpy()
            
        return features

    def detect_faces(self, frame):
        """Detect faces in frame using YOLO"""
        faces = []
        
        try:
            if self.model_type == "v8":
                results = self.model(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            conf = float(box.conf.cpu().numpy()[0])
                            
                            faces.append({
                                "box": (x, y, w, h),
                                "confidence": conf
                            })
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(rgb_frame)
                detections = results.xyxy[0].cpu().numpy()
                
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    if int(cls) == 0 and conf > self.model.conf:
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        
                        faces.append({
                            "box": (x, y, w, h),
                            "confidence": float(conf)
                        })
        except Exception as e:
            print(f"Error in face detection: {e}")
            
        return faces

    def prepare_training_data(self):
        """Prepare training data for Random Forest"""
        X = []  # Features
        y = []  # Labels
        
        # Create label encoder
        unique_persons = list(set(self.face_database.values()))  # Remove duplicates
        unique_persons = [p for p in unique_persons if p != "Unknown"]  # Remove Unknown
        
        if not unique_persons:
            return np.array([]), np.array([])
        
        self.label_encoder = {person: idx for idx, person in enumerate(unique_persons)}
        self.reverse_label_encoder = {idx: person for person, idx in self.label_encoder.items()}
        
        # Prepare training data
        for person_id, features_list in self.face_features.items():
            person_name = self.face_database.get(person_id, "Unknown")
            if person_name == "Unknown" or person_name not in self.label_encoder:
                continue
                
            person_label = self.label_encoder[person_name]
            
            for features in features_list:
                X.append(features.flatten())
                y.append(person_label)
        
        return np.array(X), np.array(y)

    def train_random_forest(self):
        """Train Random Forest classifier"""
        print("Training Random Forest classifier...")
        
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("No training data available!")
            return False
        
        # Split data for validation
        if len(X) > 10:  # Only split if we have enough data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # Validate model
        if len(X_val) > 0:
            y_pred = self.rf_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Random Forest Validation Accuracy: {accuracy:.3f}")
            
            # Print classification report
            target_names = [self.reverse_label_encoder[i] for i in sorted(self.reverse_label_encoder.keys())]
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))
        
        # Save model
        self.save_rf_model()
        print("Random Forest model trained and saved successfully!")
        return True

    def train_few_shot_model(self, person_name, image_paths):
        """Train few-shot learning model with MLflow tracking"""
        with mlflow.start_run(run_name=f"few_shot_training_{person_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("person_name", person_name)
            mlflow.log_param("num_examples", len(image_paths))
            mlflow.log_param("confidence_threshold", self.confidence_threshold)
            mlflow.log_param("device", self.device)
            mlflow.log_params(self.rf_params)
            
            # Create new person ID
            if self.face_database:
                new_id = str(max(int(k) for k in self.face_database.keys()) + 1)
            else:
                new_id = "0"
            
            # Initialize metrics
            successful_extractions = 0
            total_faces_detected = 0
            
            # Process each image
            feature_list = []
            for img_path in image_paths:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                    
                faces = self.detect_faces(frame)
                total_faces_detected += len(faces)
                
                # Find largest face
                if faces:
                    largest_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                    x, y, w, h = largest_face["box"]
                    
                    if w > 60 and h > 60:  # Minimum face size
                        face_roi = frame[y:y+h, x:x+w]
                        features = self.extract_features(face_roi)
                        feature_list.append(features)
                        successful_extractions += 1
                        
                        # Save face example
                        person_dir = os.path.join(self.few_shot_dir, new_id)
                        os.makedirs(person_dir, exist_ok=True)
                        timestamp = int(time.time() * 1000)
                        cv2.imwrite(os.path.join(person_dir, f"{timestamp}.jpg"), face_roi)
            
            # Log metrics
            mlflow.log_metric("successful_extractions", successful_extractions)
            mlflow.log_metric("total_faces_detected", total_faces_detected)
            mlflow.log_metric("extraction_success_rate", 
                            successful_extractions / len(image_paths) if image_paths else 0)
            
            if successful_extractions >= self.min_examples:
                # Update face database
                self.face_database[new_id] = person_name
                self.face_features[new_id] = feature_list
                
                # Save models
                self.save_face_database()
                self.save_face_features()
                
                # Train Random Forest with updated data
                rf_success = self.train_random_forest()
                
                if rf_success:
                    # Package model artifacts
                    model_artifacts = {
                        "face_database": self.face_database,
                        "face_features": self.face_features,
                        "feature_extractor_state": self.feature_extractor.state_dict() if self.feature_extractor else None,
                        "confidence_threshold": self.confidence_threshold,
                        "model_type": "mobilenet_v3_small_with_random_forest",
                        "rf_params": self.rf_params
                    }
                    
                    # Save model package
                    model_path = os.path.join(self.models_dir, f"face_model_v{self.current_model_version + 1}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_artifacts, f)
                    
                    # Log model to MLflow
                    mlflow.log_artifact(model_path)
                    mlflow.log_artifact(os.path.join(self.models_dir, "face_database.json"))
                    mlflow.log_artifact(os.path.join(self.models_dir, "random_forest_model.pkl"))
                    
                    # Log Random Forest model
                    mlflow.sklearn.log_model(
                        sk_model=self.rf_model,
                        artifact_path="random_forest_classifier",
                        registered_model_name="face_recognition_rf_model"
                    )
                    
                    # Register feature extractor
                    if self.feature_extractor:
                        mlflow.pytorch.log_model(
                            pytorch_model=self.feature_extractor,
                            artifact_path="feature_extractor",
                            registered_model_name="face_recognition_model"
                        )
                    
                    self.current_model_version += 1
                    mlflow.log_metric("model_version", self.current_model_version)
                    
                    print(f"Successfully trained model for {person_name} with {successful_extractions} examples")
                    return True, new_id
                else:
                    print("Failed to train Random Forest model")
                    return False, None
            else:
                print(f"Insufficient examples for {person_name}. Got {successful_extractions}, need {self.min_examples}")
                return False, None

    def evaluate_model(self, test_image_paths):
        """Evaluate model performance"""
        with mlflow.start_run(run_name=f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("num_test_images", len(test_image_paths))
            mlflow.log_param("model_version", self.current_model_version)
            
            correct_predictions = 0
            total_predictions = 0
            unknown_faces = 0
            
            for img_path in test_image_paths:
                # Extract ground truth from filename or path
                # Assuming format: person_name_*.jpg
                filename = os.path.basename(img_path)
                ground_truth = filename.split('_')[0]
                
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                    
                faces = self.detect_faces(frame)
                if faces:
                    largest_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                    x, y, w, h = largest_face["box"]
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Recognize face
                    prediction = self.recognize_face(face_roi)
                    
                    total_predictions += 1
                    if prediction["name"] == ground_truth:
                        correct_predictions += 1
                    elif prediction["name"] == "Unknown":
                        unknown_faces += 1
            
            # Calculate metrics
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            unknown_rate = unknown_faces / total_predictions if total_predictions > 0 else 0
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("unknown_rate", unknown_rate)
            mlflow.log_metric("total_predictions", total_predictions)
            
            print(f"Model Evaluation: Accuracy={accuracy:.2f}, Unknown Rate={unknown_rate:.2f}")
            return accuracy

    def recognize_face(self, face_roi):
        """Recognize face using Random Forest classifier"""
        if self.rf_model is None:
            return {"name": "Unknown", "confidence": 0, "person_id": None}
        
        # Extract features
        query_features = self.extract_features(face_roi).flatten().reshape(1, -1)
        
        try:
            # Get prediction probabilities
            probabilities = self.rf_model.predict_proba(query_features)[0]
            predicted_class = self.rf_model.predict(query_features)[0]
            confidence = probabilities[predicted_class]
            
            # Check if confidence meets threshold
            if confidence < self.confidence_threshold:
                return {"name": "Unknown", "confidence": 0, "person_id": None}
            
            # Get person name
            person_name = self.reverse_label_encoder.get(predicted_class, "Unknown")
            
            # Find person_id
            person_id = None
            for pid, name in self.face_database.items():
                if name == person_name:
                    person_id = pid
                    break
            
            return {
                "name": person_name,
                "confidence": confidence * 100,
                "person_id": person_id
            }
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return {"name": "Unknown", "confidence": 0, "person_id": None}

    def deploy_to_jetson(self, jetson_ip="192.168.50.94", jetson_user="newcastleuni"):
        """Deploy trained model to Jetson Nano"""
        print(f"Deploying model version {self.current_model_version} to Jetson Nano...")
        
        # Create deployment package
        deployment_dir = "jetson_deployment"
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Copy model files
        shutil.copy(os.path.join(self.models_dir, f"face_model_v{self.current_model_version}.pkl"), 
                   os.path.join(deployment_dir, "face_model.pkl"))
        shutil.copy(os.path.join(self.models_dir, "face_database.json"), deployment_dir)
        shutil.copy(os.path.join(self.models_dir, "random_forest_model.pkl"), deployment_dir)
        shutil.copy(os.path.join(self.models_dir, "label_encoder.pkl"), deployment_dir)
        
        # Create deployment script
        with open(os.path.join(deployment_dir, "deploy.sh"), 'w') as f:
            f.write(f"""#!/bin/bash
# Deployment script for Jetson Nano
echo "Deploying face recognition model with Random Forest to Jetson Nano..."

# Copy files to Jetson
scp -r ../jetson_deployment/* {jetson_user}@{jetson_ip}:~/face_recognition/models/

echo "Deployment complete!"
""")
        
        os.chmod(os.path.join(deployment_dir, "deploy.sh"), 0o755)
        print(f"Deployment package created in {deployment_dir}")
        print(f"Run './deploy.sh' from the {deployment_dir} directory to deploy to Jetson")


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = EdgeFaceRecognitionTrainer()
    
    # Example: Register a new person
    person_name = "amrit"
    image_paths = [f"images/{person_name}/{person_name}_1.jpg", f"images/{person_name}/{person_name}_2.jpg", f"images/{person_name}/{person_name}_3.jpg", f"images/{person_name}/{person_name}_4.jpg", f"images/{person_name}/{person_name}_5.jpg"]
    
    success, person_id = trainer.train_few_shot_model(person_name, image_paths)
    if success:
        print(f"Successfully registered {person_name} with ID: {person_id}")
