import pickle
import json
import numpy as np
import pandas as pd
from typing import Any
import mlflow
from mlflow.pyfunc import PythonModel

class FaceRecognitionModel(PythonModel):
    def load_context(self, context):
        """Load model artifacts"""
        print("Loading face recognition model...")
        
        # Load face database
        with open(context.artifacts["face_database"], 'r') as f:
            self.face_database = json.load(f)
        
        # Load face features
        with open(context.artifacts["face_features"], 'rb') as f:
            self.face_features = pickle.load(f)
        
        # Load model parameters
        with open(context.artifacts["model_params"], 'r') as f:
            params = json.load(f)
            self.similarity_threshold = params["similarity_threshold"]
            self.min_examples = params["min_examples"]
        
        print(f"Loaded model with {len(self.face_database)} identities")
    
    def extract_features_simple(self, face_img_array):
        """Simple feature extraction using histogram (fallback)"""
        try:
            import cv2
            gray = cv2.cvtColor(face_img_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return dummy features if OpenCV fails
            return np.random.random(128)
    
    def predict(self, context, model_input):
        """Predict function for MLflow model serving"""
        print(f"Received prediction request with {len(model_input)} instances")
        
        results = []
        
        # Handle both DataFrame and dict inputs
        if isinstance(model_input, pd.DataFrame):
            inputs = model_input.to_dict('records')
        else:
            inputs = model_input if isinstance(model_input, list) else [model_input]
        
        for instance in inputs:
            try:
                # Extract face image from input (base64 encoded)
                if isinstance(instance, dict):
                    face_b64 = instance.get('face_image', '')
                else:
                    face_b64 = instance
                
                if not face_b64:
                    results.append({
                        "name": "Unknown",
                        "confidence": 0.0,
                        "person_id": None,
                        "error": "No face_image provided"
                    })
                    continue
                
                # Decode base64 image
                import base64
                import cv2
                face_data = base64.b64decode(face_b64)
                nparr = np.frombuffer(face_data, np.uint8)
                face_img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if face_img_array is None:
                    results.append({
                        "name": "Unknown", 
                        "confidence": 0.0,
                        "person_id": None,
                        "error": "Invalid image data"
                    })
                    continue
                
                # Extract features and predict
                query_features = self.extract_features_simple(face_img_array)
                result = self.recognize_face(query_features)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing instance: {e}")
                results.append({
                    "name": "Unknown",
                    "confidence": 0.0, 
                    "person_id": None,
                    "error": str(e)
                })
        
        return results
    
    def recognize_face(self, query_features):
        """Recognize face from features"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        best_match = "Unknown"
        best_similarity = -1
        best_confidence = 0.0
        best_person_id = None
        
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
                    best_confidence = float(avg_similarity * 100)
                    if avg_similarity >= self.similarity_threshold:
                        best_match = self.face_database.get(person_id, "Unknown")
                        best_person_id = person_id
        
        return {
            "name": best_match,
            "confidence": best_confidence,
            "person_id": best_person_id
        }