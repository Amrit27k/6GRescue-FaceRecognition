#!/usr/bin/env python3
"""
Complete Object Recognition Training and Testing Script for Notebook
Combines training, model loading, and inference testing in one script
"""

import cv2
import numpy as np
import time
import os
import torch
import torchvision
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
import json
import pickle
from datetime import datetime
import joblib
import glob
from IPython.display import display, Image as IPImage
import warnings
warnings.filterwarnings('ignore')

class CompleteObjectRecognitionSystem:
    def __init__(self, images_dir="images", models_dir="models/v3"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Directories
        self.images_dir = images_dir
        self.models_dir = models_dir
        self.few_shot_dir = "few_shot_examples"
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.few_shot_dir, exist_ok=True)
        
        # Models and data
        self.yolo_model = None
        self.feature_extractor = None
        self.rf_model = None
        self.face_database = {}
        self.face_features = {}
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        
        # Parameters - ADJUSTED FOR BETTER PERFORMANCE
        self.confidence_threshold = 0.4  # Lowered from 0.6 to reduce Unknown predictions
        self.min_examples = 2  # Reduced to allow training with fewer examples
        self.yolo_conf_threshold = 0.01  # Very low YOLO confidence
        
        # RF parameters - TUNED FOR SMALL DATASETS
        self.rf_params = {
            'n_estimators': 50,  # Reduced for faster training
            'max_depth': 6,      # Reduced to prevent overfitting
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'  # Handle class imbalance
        }
        
        # Load models
        self.load_yolo_model()
        self.load_feature_extractor()
        self.load_existing_model()
        
    def load_yolo_model(self):
        """Load YOLO model for object detection"""
        print("Loading YOLO model...")
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            print("YOLO model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
    
    def load_feature_extractor(self):
        """Load feature extraction model"""
        print("Loading feature extractor...")
        try:
            model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            print("Feature extractor loaded successfully!")
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
    
    def load_existing_model(self):
        """Load existing trained model if available"""
        try:
            db_path = os.path.join(self.models_dir, "face_database.json")
            features_path = os.path.join(self.models_dir, "face_features.pkl")
            rf_path = os.path.join(self.models_dir, "random_forest_model.pkl")
            encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
            
            if all(os.path.exists(p) for p in [db_path, features_path, rf_path, encoder_path]):
                # Load database
                with open(db_path, 'r') as f:
                    self.face_database = json.load(f)
                
                # Load features
                with open(features_path, 'rb') as f:
                    self.face_features = pickle.load(f)
                
                # Load RF model
                self.rf_model = joblib.load(rf_path)
                
                # Load encoders
                with open(encoder_path, 'rb') as f:
                    encoders = pickle.load(f)
                    self.label_encoder = encoders['label_encoder']
                    self.reverse_label_encoder = encoders['reverse_label_encoder']
                
                print(f"Loaded existing model with {len(self.face_database)} classes:")
                for class_name in self.label_encoder.keys():
                    print(f"  - {class_name}")
                return True
        except Exception as e:
            print(f"No existing model found or error loading: {e}")
        return False
    
    def detect_objects(self, frame, debug=False):
        """Detect objects using YOLO with fallback"""
        objects = []
        
        if self.yolo_model is None:
            if debug:
                print("YOLO model not available, using fallback")
        else:
            try:
                results = self.yolo_model(frame, verbose=False, conf=self.yolo_conf_threshold)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            conf = float(box.conf.cpu().numpy()[0])
                            
                            if w > 20 and h > 20:  # Minimum size
                                objects.append({
                                    "box": (x, y, w, h),
                                    "confidence": conf
                                })
                                if debug:
                                    print(f"YOLO detected object: {w}x{h} at ({x},{y}) conf={conf:.3f}")
                                    
            except Exception as e:
                if debug:
                    print(f"YOLO detection error: {e}")
        
        # Fallback: If YOLO detects nothing, create center region detection
        if len(objects) == 0:
            if debug:
                print("Using center region fallback")
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            fallback_w, fallback_h = int(w * 0.8), int(h * 0.8)  # Larger region
            fallback_x = max(0, center_x - fallback_w // 2)
            fallback_y = max(0, center_y - fallback_h // 2)
            
            objects.append({
                "box": (fallback_x, fallback_y, fallback_w, fallback_h),
                "confidence": 0.5
            })
                    
        return objects
    
    def extract_features(self, object_img):
        """Extract features from object image"""
        if self.feature_extractor is None:
            # Fallback feature extraction
            gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])  # Smaller histogram
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            input_tensor = transform(object_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().cpu().numpy()
                
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Fallback to simple features
            gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            return cv2.normalize(hist, hist).flatten()
    
    def prepare_training_data(self):
        """Prepare training data with REDUCED synthetic unknown generation"""
        X = []
        y = []
        
        # Get unique classes
        unique_classes = list(set(self.face_database.values()))
        unique_classes = [c for c in unique_classes if c != "Unknown"]
        
        if not unique_classes:
            return np.array([]), np.array([])
        
        # Add Unknown class
        unique_classes.append("Unknown")
        
        self.label_encoder = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.reverse_label_encoder = {idx: cls for cls, idx in self.label_encoder.items()}
        
        # Add real training data
        for obj_id, features_list in self.face_features.items():
            obj_name = self.face_database.get(obj_id, "Unknown")
            if obj_name not in self.label_encoder:
                continue
                
            obj_label = self.label_encoder[obj_name]
            
            for features in features_list:
                X.append(features.flatten())
                y.append(obj_label)
        
        # REDUCED synthetic unknown generation
        if len(X) > 0:
            X_known = np.array(X)
            unknown_label = self.label_encoder["Unknown"]
            
            # Generate FEWER synthetic unknowns - only 25% of known samples
            num_unknowns = max(1, len(X) // 4)  # Much less synthetic data
            
            for _ in range(num_unknowns):
                base_idx = np.random.randint(0, len(X_known))
                base_features = X_known[base_idx].copy()
                
                # LESS aggressive noise - smaller noise scale
                noise_scale = np.std(base_features) * 1.0  # Reduced from 2.0
                noise = np.random.normal(0, noise_scale, base_features.shape)
                synthetic_unknown = base_features + noise
                
                X.append(synthetic_unknown)
                y.append(unknown_label)
        
        print(f"Training data prepared: {len(X)} samples")
        class_counts = np.bincount(y)
        for i, count in enumerate(class_counts):
            if i in self.reverse_label_encoder:
                print(f"  {self.reverse_label_encoder[i]}: {count} samples")
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("No training data available!")
            return False
        
        # Train with all data (no validation split for small datasets)
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.rf_model.fit(X, y)
        
        # Calculate training accuracy
        y_pred = self.rf_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Training Accuracy: {accuracy:.3f}")
        
        # Print detailed classification report
        target_names = [self.reverse_label_encoder[i] for i in sorted(self.reverse_label_encoder.keys())]
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=target_names, zero_division=0))
        
        # Save model
        self.save_model()
        print("Model trained and saved successfully!")
        return True
    
    def save_model(self):
        """Save all model components"""
        # Save database
        db_path = os.path.join(self.models_dir, "face_database.json")
        with open(db_path, 'w') as f:
            json.dump(self.face_database, f)
        
        # Save features
        features_path = os.path.join(self.models_dir, "face_features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(self.face_features, f)
        
        # Save RF model
        if self.rf_model:
            rf_path = os.path.join(self.models_dir, "random_forest_model.pkl")
            joblib.dump(self.rf_model, rf_path)
            
            # Save encoders
            encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
            encoders = {
                'label_encoder': self.label_encoder,
                'reverse_label_encoder': self.reverse_label_encoder
            }
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoders, f)
    
    def train_on_class(self, class_name, debug=True):
        """Train model on a specific class"""
        class_dir = os.path.join(self.images_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Error: Directory {class_dir} not found!")
            return False
        
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        if not image_files:
            print(f"No images found in {class_dir}")
            return False
        
        print(f"Training on {class_name} with {len(image_files)} images...")
        
        # Create new class ID
        if self.face_database:
            new_id = str(max(int(k) for k in self.face_database.keys()) + 1)
        else:
            new_id = "0"
        
        successful_extractions = 0
        feature_list = []
        
        for img_path in image_files:
            if debug:
                print(f"Processing: {os.path.basename(img_path)}")
            
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            # Detect objects
            objects = self.detect_objects(frame, debug=debug)
            
            if objects:
                # Use largest detection
                largest_obj = max(objects, key=lambda x: x["box"][2] * x["box"][3])
                x, y, w, h = largest_obj["box"]
                
                if w > 30 and h > 30:
                    object_roi = frame[y:y+h, x:x+w]
                    features = self.extract_features(object_roi)
                    feature_list.append(features)
                    successful_extractions += 1
                    
                    if debug:
                        print(f"  Extracted features: {features.shape}")
        
        if successful_extractions >= self.min_examples:
            # Update database
            self.face_database[new_id] = class_name
            self.face_features[new_id] = feature_list
            
            print(f"Successfully processed {class_name}: {successful_extractions} examples")
            return True
        else:
            print(f"Insufficient examples for {class_name}: got {successful_extractions}, need {self.min_examples}")
            return False
    
    def recognize_object(self, object_roi, debug=False):
        """Recognize object using trained model"""
        if self.rf_model is None:
            return {"name": "Unknown", "confidence": 0, "object_id": None}
        
        # Extract features
        query_features = self.extract_features(object_roi).flatten().reshape(1, -1)
        
        try:
            # Get predictions
            probabilities = self.rf_model.predict_proba(query_features)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            if debug:
                print(f"Prediction probabilities: {dict(zip(self.reverse_label_encoder.values(), probabilities))}")
                print(f"Predicted class: {predicted_class}, confidence: {confidence:.3f}")
            
            # Get object name
            object_name = self.reverse_label_encoder.get(predicted_class, "Unknown")
            
            # Apply confidence threshold ONLY for non-Unknown predictions
            if object_name != "Unknown" and confidence < self.confidence_threshold:
                if debug:
                    print(f"Confidence {confidence:.3f} below threshold {self.confidence_threshold}, returning Unknown")
                return {"name": "Unknown", "confidence": confidence * 100, "object_id": None}
            
            # Find object_id
            object_id = None
            for oid, name in self.face_database.items():
                if name == object_name:
                    object_id = oid
                    break
            
            return {
                "name": object_name,
                "confidence": confidence * 100,
                "object_id": object_id
            }
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return {"name": "Error", "confidence": 0, "object_id": None}
    
    def test_image(self, image_path, debug=True, show_image=True):
        """Test recognition on a single image"""
        print(f"\n--- Testing: {os.path.basename(image_path)} ---")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error loading image!")
            return None
        
        # Detect objects
        objects = self.detect_objects(frame, debug=debug)
        print(f"Detected {len(objects)} object(s)")
        
        results = []
        result_frame = frame.copy()
        
        for i, obj in enumerate(objects):
            x, y, w, h = obj["box"]
            
            if w > 30 and h > 30:
                object_roi = frame[y:y+h, x:x+w]
                recognition = self.recognize_object(object_roi, debug=debug)
                
                result = {
                    "bbox": (x, y, w, h),
                    "recognition": recognition
                }
                results.append(result)
                
                print(f"Object {i}: {recognition['name']} ({recognition['confidence']:.1f}%)")
                
                # Draw results
                color = (0, 255, 0) if recognition['name'] != 'Unknown' else (0, 0, 255)
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
                label = f"{recognition['name']}: {recognition['confidence']:.1f}%"
                cv2.putText(result_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show result
        if show_image:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Recognition Result: {os.path.basename(image_path)}")
            plt.axis('off')
            plt.show()
        
        return results

# Initialize the system
print("Initializing Complete Object Recognition System...")
system = CompleteObjectRecognitionSystem()

# Training function
def train_all_classes(class_names):
    """Train on multiple classes"""
    print("=== TRAINING PHASE ===")
    
    success_count = 0
    for class_name in class_names:
        print(f"\n--- Training on {class_name} ---")
        if system.train_on_class(class_name, debug=True):
            success_count += 1
    
    if success_count > 0:
        print(f"\n--- Training Random Forest on {success_count} classes ---")
        system.train_model()
        return True
    else:
        print("No classes successfully processed!")
        return False

# Testing function
def test_images(test_image_paths, debug=False):
    """Test on multiple images"""
    print("\n=== TESTING PHASE ===")
    
    all_results = {}
    prediction_counts = {}
    
    for image_path in test_image_paths:
        results = system.test_image(image_path, debug=debug, show_image=True)
        all_results[image_path] = results
        
        # Count predictions
        if results:
            for result in results:
                name = result['recognition']['name']
                if name not in prediction_counts:
                    prediction_counts[name] = 0
                prediction_counts[name] += 1
    
    # Summary
    print("\n=== PREDICTION SUMMARY ===")
    for name, count in sorted(prediction_counts.items()):
        print(f"{name}: {count} predictions")
    
    return all_results

# ==================== USAGE EXAMPLES ====================

# Example 1: Train on your 4 classes
print("\n" + "="*50)
print("EXAMPLE: Train on 4 object classes")
print("="*50)

class_names = ["spark-robot", "Leave-Warning-Sign", "Fire-extinguisher", "High_Voltage_Sign"]

# Uncomment to train:
# train_success = train_all_classes(class_names)

# Example 2: Test on custom images
print("\n" + "="*50)
print("EXAMPLE: Test on custom images")
print("="*50)

# Add your test image paths here
test_images = [
    # "path/to/test_image1.jpg",
    # "path/to/test_image2.jpg",
]

# Uncomment to test:
# if test_images:
#     test_results = test_images(test_images, debug=True)

print("\n" + "="*50)
print("SYSTEM READY!")
print("="*50)
print("To use:")
print("1. Place training images in folders: images/spark-robot/, images/Leave-Warning-Sign/, etc.")
print("2. Run: train_success = train_all_classes(class_names)")
print("3. Add test image paths to test_images list")
print("4. Run: test_results = test_images(test_images, debug=True)")
print("="*50)