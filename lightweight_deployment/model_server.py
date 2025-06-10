#!/usr/bin/env python3
"""
Lightweight MLflow model server for Jetson deployment
No heavy dependencies - only essential packages
"""
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class LightweightFaceRecognitionModel:
    def __init__(self):
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load model artifacts"""
        try:
            # Load face database
            with open('face_database.json', 'r') as f:
                self.face_database = json.load(f)
            
            # Load face features
            with open('face_features.pkl', 'rb') as f:
                self.face_features = pickle.load(f)
            
            # Load model parameters
            with open('model_params.json', 'r') as f:
                params = json.load(f)
                self.similarity_threshold = params["similarity_threshold"]
                self.min_examples = params["min_examples"]
            
            logger.info(f"Loaded model with {len(self.face_database)} identities")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def extract_features_simple(self, face_img_array):
        """Simple feature extraction using histogram (fallback)"""
        import cv2
        gray = cv2.cvtColor(face_img_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def predict(self, face_img_array):
        """Predict face identity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_features = self.extract_features_simple(face_img_array)
        
        best_match = "Unknown"
        best_similarity = -1
        best_confidence = 0
        
        for person_id, features_list in self.face_features.items():
            if len(features_list) < self.min_examples:
                continue
                
            similarities = []
            for features in features_list:
                if features.shape != query_features.shape:
                    continue
                    
                sim = cosine_similarity([query_features], [features])[0][0]
                similarities.append(sim)
            
            if similarities:
                similarities.sort(reverse=True)
                top_n = min(3, len(similarities))
                avg_similarity = sum(similarities[:top_n]) / top_n
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_confidence = avg_similarity * 100
                    if avg_similarity >= self.similarity_threshold:
                        best_match = self.face_database.get(person_id, "Unknown")
        
        return {
            "name": best_match,
            "confidence": best_confidence,
            "person_id": person_id if best_match != "Unknown" else None
        }

# Initialize model
model = LightweightFaceRecognitionModel()

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        instances = data.get('instances', [])
        
        results = []
        for instance in instances:
            # Decode base64 image
            import base64
            face_b64 = instance.get('face_image', '')
            face_data = base64.b64decode(face_b64)
            
            # Convert to numpy array
            import cv2
            nparr = np.frombuffer(face_data, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Predict
            result = model.predict(face_img)
            results.append(result)
        
        return jsonify({"predictions": results})
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
