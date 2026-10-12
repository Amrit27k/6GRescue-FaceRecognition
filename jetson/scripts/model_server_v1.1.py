#!/usr/bin/env python3
"""Serving-only model server. Loads the artifacts produced by the training
script and never trains anything.

Required in ARTIFACT_DIR (default /app/artifacts):
  random_forest_model.pkl   (joblib, from training)
  label_encoder.pkl         (pickle with 'label_encoder' and 'reverse_label_encoder')
  face_database.json        (id -> class name)
Optional:
  model_params.json         {"confidence_threshold": 0.2}
"""
import json
import logging
import os
import pickle

import joblib
import numpy as np
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/app/models")
UNKNOWN = "Unknown"

app = Flask(__name__)


class ServedModel:
    def __init__(self, artifact_dir):
        self.dir = artifact_dir
        # Fail fast: a missing artifact must stop the container, not
        # silently produce an empty model that answers Unknown forever.
        self.rf = joblib.load(os.path.join(artifact_dir, "random_forest_model.pkl"))
        with open(os.path.join(artifact_dir, "label_encoder.pkl"), "rb") as f:
            enc = pickle.load(f)
        self.reverse = {int(k): v for k, v in enc["reverse_label_encoder"].items()}
        with open(os.path.join(artifact_dir, "face_database.json")) as f:
            db = json.load(f)
        self.name_to_id = {name: pid for pid, name in db.items()}

        self.threshold = 0.2  # same default as the training script
        params_path = os.path.join(artifact_dir, "model_params.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                self.threshold = float(json.load(f).get("confidence_threshold", self.threshold))
        self.threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", self.threshold))

        self.n_features = int(self.rf.n_features_in_)
        logger.info("Loaded RF: %d trees, %d features, classes=%s, threshold=%.2f",
                    self.rf.n_estimators, self.n_features,
                    [self.reverse.get(int(c)) for c in self.rf.classes_], self.threshold)

    def predict(self, features):
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.shape[1]}")
        probs = self.rf.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        name = self.reverse.get(int(self.rf.classes_[idx]), UNKNOWN)
        top3 = sorted(((self.reverse.get(int(c), "?"), float(p))
                       for c, p in zip(self.rf.classes_, probs)),
                      key=lambda t: -t[1])[:3]
        logger.info("pred=%s conf=%.3f top3=%s", name, conf,
                    [(n, round(p, 2)) for n, p in top3])
        # Same rule as the training script's recognize_object()
        if name != UNKNOWN and conf < self.threshold:
            name = UNKNOWN
        pid = self.name_to_id.get(name) if name != UNKNOWN else None
        return {"name": name, "confidence": conf * 100, "person_id": pid}


model = ServedModel(ARTIFACT_DIR)


@app.route("/ping")
def ping():
    return jsonify({"status": "pong", "classes": len(model.reverse)})


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True,
                    "n_features": model.n_features,
                    "threshold": model.threshold})


@app.route("/info")
def info():
    return jsonify({"model_type": "RandomForest (served from training artifacts)",
                    "classes": sorted(v for v in model.reverse.values()),
                    "n_estimators": int(model.rf.n_estimators),
                    "n_features": model.n_features,
                    "confidence_threshold": model.threshold,
                    "artifact_dir": model.dir})


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        data = request.get_json(force=True)
        results = []
        for inst in data.get("instances", []):
            vec = inst.get("image_feature_vector")
            if vec is None or len(vec) == 0:
                results.append({"name": "Error", "confidence": 0,
                                "person_id": None, "error": "no feature vector"})
                continue
            try:
                results.append(model.predict(vec))
            except Exception as e:
                logger.error("predict failed: %s", e)
                results.append({"name": "Error", "confidence": 0,
                                "person_id": None, "error": str(e)})
        return jsonify({"predictions": results})
    except Exception as e:
        logger.error("endpoint error: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
