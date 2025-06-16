#!/usr/bin/env python3
"""Lightweight model server without OpenCV dependency"""
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
import logging
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class LightweightFaceRecognitionModel:
    def __init__(self):
        self.rf_model = None
        self.scaler = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load model artifacts and train RandomForest if needed"""
        try:
            # Load face database
            with open('face_database.json', 'r') as f:
                self.face_database = json.load(f)
            
            # Load face features
            with open('face_features.pkl', 'rb') as f:
                self.face_features = pickle.load(f)
            
            # Load model parameters
            if os.path.exists('model_params.json'):
                with open('model_params.json', 'r') as f:
                    params = json.load(f)
                    self.confidence_threshold = params.get("confidence_threshold", 0.6)
                    self.min_examples = params.get("min_examples", 3)
            else:
                self.confidence_threshold = 0.6
                self.min_examples = 3
            
            # Train RandomForest model
            self.train_random_forest()
            
            logger.info(f"Loaded model with {len(self.face_database)} identities")
            logger.info(f"RandomForest trained with {len(self.label_encoder)} classes")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            # Create empty model if files don't exist
            self.face_database = {}
            self.face_features = {}
            self.confidence_threshold = 0.6
            self.min_examples = 3
            self.rf_model = None
            logger.warning("Using empty model")
    
    def train_random_forest(self):
        """Train RandomForest classifier from loaded features"""
        if not self.face_features:
            logger.warning("No features available for training")
            return
        
        X = []  # Features
        y = []  # Labels
        
        # Prepare training data
        label_id = 0
        for person_id, features_list in self.face_features.items():
            if len(features_list) >= self.min_examples:
                person_name = self.face_database.get(person_id, f"Person_{person_id}")
                
                # Store label mapping
                self.label_encoder[person_name] = label_id
                self.reverse_label_encoder[label_id] = person_name
                
                # Add features to training set
                for features in features_list:
                    X.append(features)
                    y.append(label_id)
                
                label_id += 1
        
        if len(X) < 2:
            logger.warning("Not enough training data for RandomForest")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train RandomForest
        self.rf_model = RandomForestClassifier(
            n_estimators=50,  # Reduced for lighter model
            max_depth=8,      # Reduced depth
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1  # Single thread for consistency
        )
        
        self.rf_model.fit(X_scaled, y)
        
        # Calculate training accuracy
        train_accuracy = self.rf_model.score(X_scaled, y)
        logger.info(f"RandomForest training accuracy: {train_accuracy:.3f}")
    
    def extract_features_pil(self, face_img_pil):
        """Extract features using PIL only (no OpenCV)"""
        try:
            # Resize to standard size
            face_resized = face_img_pil.resize((64, 64))
            
            # Convert to grayscale
            if face_resized.mode != 'L':
                gray = face_resized.convert('L')
            else:
                gray = face_resized
            
            # Convert to numpy array
            gray_array = np.array(gray)
            
            # Histogram features
            hist, _ = np.histogram(gray_array, bins=64, range=(0, 256))
            hist = hist / (np.sum(hist) + 1e-7)  # Normalize
            
            # Simple texture features (without LBP)
            h, w = gray_array.shape
            texture_features = []
            
            # Sample texture at regular intervals
            for i in range(4, h-4, 8):  # Sample every 8th pixel
                for j in range(4, w-4, 8):
                    # Get local patch
                    patch = gray_array[i-2:i+3, j-2:j+3]
                    if patch.shape == (5, 5):
                        # Simple texture measures
                        texture_features.extend([
                            np.std(patch),           # Local standard deviation
                            np.max(patch) - np.min(patch),  # Local range
                            np.mean(patch > np.mean(patch))  # Binary threshold ratio
                        ])
            
            # Pad or truncate texture features to fixed size
            texture_features = texture_features[:60]  # Take first 60
            while len(texture_features) < 60:
                texture_features.append(0)  # Pad with zeros
            
            # Statistical features
            stats = [
                np.mean(gray_array), 
                np.std(gray_array), 
                np.min(gray_array), 
                np.max(gray_array),
                np.percentile(gray_array, 25), 
                np.percentile(gray_array, 75),
                np.median(gray_array),
                len(gray_array[gray_array > np.mean(gray_array)]) / gray_array.size  # Above-mean ratio
            ]
            
            # Combine all features
            features = np.concatenate([hist, texture_features, stats])
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return zero features if extraction fails
            return np.zeros(64 + 60 + 8)  # hist + texture + stats
    
    def predict(self, face_img_pil):
        """Predict face identity using RandomForest"""
        if self.rf_model is None:
            return {
                "name": "Unknown",
                "confidence": 0,
                "person_id": None
            }
        
        try:
            # Extract features
            query_features = self.extract_features_pil(face_img_pil)
            
            # Scale features
            query_features_scaled = self.scaler.transform([query_features])
            
            # Get prediction probabilities
            probabilities = self.rf_model.predict_proba(query_features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Check if confidence meets threshold
            if confidence >= self.confidence_threshold:
                predicted_name = self.reverse_label_encoder[predicted_class]
                
                # Find person_id for the predicted name
                person_id = None
                for pid, name in self.face_database.items():
                    if name == predicted_name:
                        person_id = pid
                        break
                
                return {
                    "name": predicted_name,
                    "confidence": confidence * 100,
                    "person_id": person_id
                }
            else:
                return {
                    "name": "Unknown",
                    "confidence": confidence * 100,
                    "person_id": None
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "name": "Error",
                "confidence": 0,
                "person_id": None
            }

# Initialize model
model = LightweightFaceRecognitionModel()

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    model_status = "trained" if model.rf_model is not None else "empty"
    return jsonify({
        "status": "pong",
        "model_status": model_status,
        "classes": len(model.label_encoder),
        "classifier": "RandomForest-Lightweight"
    })

@app.route('/invocations', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        instances = data.get('instances', [])
        
        results = []
        for instance in instances:
            # Decode base64 image
            face_b64 = instance.get('face_image', '')
            if not face_b64:
                results.append({
                    "name": "Error",
                    "confidence": 0,
                    "person_id": None,
                    "error": "No image data"
                })
                continue
                
            try:
                # Decode using PIL instead of OpenCV
                face_data = base64.b64decode(face_b64)
                face_img = Image.open(io.BytesIO(face_data))
                
                # Convert to RGB if needed
                if face_img.mode in ('RGBA', 'P'):
                    face_img = face_img.convert('RGB')
                
                # Predict
                result = model.predict(face_img)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Image processing error: {e}")
                results.append({
                    "name": "Error",
                    "confidence": 0,
                    "person_id": None,
                    "error": str(e)
                })
        
        return jsonify({"predictions": results})
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        "model_type": "RandomForest Face Recognition (Lightweight)",
        "classes": list(model.label_encoder.keys()),
        "num_classes": len(model.label_encoder),
        "confidence_threshold": model.confidence_threshold,
        "min_examples": model.min_examples,
        "trained": model.rf_model is not None,
        "dependencies": ["PIL", "scikit-learn", "numpy"]
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model.rf_model is not None,
        "database_size": len(model.face_database),
        "trained_classes": len(model.label_encoder)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)