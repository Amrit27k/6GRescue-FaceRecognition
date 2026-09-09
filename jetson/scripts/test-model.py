#!/usr/bin/env python3
"""Lightweight model server without OpenCV dependency - Updated for v4 models"""
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
import logging
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class LightweightFaceRecognitionModel:
    def __init__(self, models_dir="data"):  # Updated to v4
        self.models_dir = models_dir
        self.rf_model = None
        self.scaler = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.load_model_artifacts()

    def load_model_artifacts(self):
        """Load model artifacts and train RandomForest if needed"""
        try:
            # Load face database
            db_path = os.path.join(self.models_dir, 'face_database.json')
            with open(db_path, 'r') as f:
                self.face_database = json.load(f)
            logger.info(f"Loaded face database with {len(self.face_database)} identities")
            
            # Load face features
            features_path = os.path.join(self.models_dir, 'face_features.pkl')
            with open(features_path, 'rb') as f:
                self.face_features = pickle.load(f)
            logger.info(f"Loaded face features with {len(self.face_features)} feature vectors")
            
            # Load Random Forest model directly
            rf_path = os.path.join(self.models_dir, 'random_forest_model.pkl')
            if os.path.exists(rf_path):
                #with open(encoder_path, 'rb') as f:
                import joblib
                self.rf_model = joblib.load(f)
                logger.info("Loaded pre-trained Random Forest model")
                
                # Load encoders
                encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
                with open(encoder_path, 'rb') as f:
                    encoders = pickle.load(f)
                    self.label_encoder = encoders['label_encoder']
                    self.reverse_label_encoder = encoders['reverse_label_encoder']
                
                logger.info(f"Loaded encoders with {len(self.label_encoder)} classes")
            else:
                # Fallback to training if no pre-trained model
                logger.warning("No pre-trained RF model found, training from features")
                self.train_random_forest()
            
            # Load model parameters if available
            params_path = os.path.join(self.models_dir, 'model_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    logger.info("Loading model parameters from file")
                    params = pickle.load(f)
                    self.confidence_threshold = params.get("confidence_threshold", 0.2)  # Updated threshold
                    self.min_examples = params.get("min_examples", 2)  # Updated min examples
            else:
                logger.info("Using default model parameters")
                self.confidence_threshold = 0.2  # Lowered threshold
                self.min_examples = 2  # Reduced min examples

            logger.info(f"Model loaded with confidence threshold: {self.confidence_threshold}")
            logger.info(f"Available classes: {list(self.label_encoder.keys())}")

        except Exception as e:
            logger.error(f"Error loading modeld artifacts: {e}")
            # Create empty model if files don't exist
            self.face_database = {}
            self.face_features = {}
            self.confidence_threshold = 0.2
            self.min_examples = 2
            self.rf_model = None
            logger.warning("Using empty model")

    def train_random_forest(self):
        """Train RandomForest classifier from loaded features (fallback)"""
        if not self.face_features:
            logger.warning("No features available for training")
            return

        X = []  # Features
        y = []  # Labels

        # Get unique classes
        unique_classes = list(set(self.face_database.values()))
        unique_classes = [c for c in unique_classes if c != "Unknown"]
        
        if not unique_classes:
            logger.warning("No valid classes found")
            return
        
        # Add Unknown class
        unique_classes.append("Unknown")
        
        self.label_encoder = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.reverse_label_encoder = {idx: cls for cls, idx in self.label_encoder.items()}

        # Prepare training data
        for person_id, features_list in self.face_features.items():
            person_name = self.face_database.get(person_id, "Unknown")
            if person_name not in self.label_encoder:
                continue
                
            person_label = self.label_encoder[person_name]
            
            for features in features_list:
                X.append(features.flatten())
                y.append(person_label)

        # Add minimal synthetic unknowns (matching training script logic)
        if len(X) > 0:
            X_known = np.array(X)
            unknown_label = self.label_encoder["Unknown"]
            
            num_unknowns = max(1, len(X) // 6)  # Match training script
            
            for _ in range(num_unknowns):
                base_idx = np.random.randint(0, len(X_known))
                base_features = X_known[base_idx].copy()
                
                noise_scale = np.std(base_features) * 0.5  # Match training script
                noise = np.random.normal(0, noise_scale, base_features.shape)
                synthetic_unknown = base_features + noise
                
                X.append(synthetic_unknown)
                y.append(unknown_label)

        if len(X) < 2:
            logger.warning("Not enough training data for RandomForest")
            return

        X = np.array(X)
        y = np.array(y)

        # Train RandomForest with same parameters as training script
        self.rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced',
            n_jobs=1
        )

        self.rf_model.fit(X, y)

        # Calculate training accuracy
        train_accuracy = self.rf_model.score(X, y)
        logger.info(f"RandomForest training accuracy: {train_accuracy:.3f}")

    def predict(self, query_features):
        """Predict object identity using RandomForest"""
        if self.rf_model is None:
            return {
                "name": "Model Not Loaded",
                "confidence": 0,
                "object_id": None  # Changed from person_id to object_id
            }

        try:
            query_features = np.array(query_features).flatten().reshape(1, -1)
            
            # Get prediction probabilities
            probabilities = self.rf_model.predict_proba(query_features)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            logger.debug(f"Prediction probabilities: {dict(zip(self.reverse_label_encoder.values(), probabilities))}")
            logger.debug(f"Predicted class: {predicted_class}, confidence: {confidence:.3f}")
            
            # Get object name
            object_name = self.reverse_label_encoder.get(predicted_class, "Unknown")
            
            # Apply confidence threshold logic matching training script
            if object_name != "Unknown" and confidence < self.confidence_threshold:
                logger.debug(f"Confidence {confidence:.3f} below threshold {self.confidence_threshold}, returning Unknown")
                return {
                    "name": "Unknown",
                    "confidence": confidence * 100,
                    "object_id": None
                }

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
            logger.error(f"Prediction error: {e}")
            return {
                "name": "Error",
                "confidence": 0,
                "object_id": None
            }

# Initialize model
model = LightweightFaceRecognitionModel(models_dir="models/v4")  # Updated path

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    model_status = "trained" if model.rf_model is not None else "empty"
    return jsonify({
        "status": "pong",
        "model_status": model_status,
        "classes": len(model.label_encoder),
        "classifier": "RandomForest-ObjectRecognition",  # Updated name
        "confidence_threshold": model.confidence_threshold
    })

@app.route('/invocations', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        instances = data.get('instances', [])

        results = []
        for instance in instances:
            # Get feature vector
            query_features = instance.get('image_feature_vector', [])
            if not query_features:
                results.append({
                    "name": "Error",
                    "confidence": 0,
                    "object_id": None,
                    "error": "No feature vector provided"
                })
                continue

            try:
                result = model.predict(query_features)
                results.append(result)

            except Exception as e:
                logger.error(f"Prediction processing error: {e}")
                results.append({
                    "name": "Error",
                    "confidence": 0,
                    "object_id": None,
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
        "model_type": "RandomForest Object Recognition",  # Updated
        "classes": list(model.label_encoder.keys()),
        "num_classes": len(model.label_encoder),
        "confidence_threshold": model.confidence_threshold,
        "min_examples": model.min_examples,
        "trained": model.rf_model is not None,
        "model_version": "v4",  # Added version
        "dependencies": ["scikit-learn", "numpy", "joblib"]
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model.rf_model is not None,
        "database_size": len(model.face_database),
        "trained_classes": len(model.label_encoder),
        "models_directory": model.models_dir
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)