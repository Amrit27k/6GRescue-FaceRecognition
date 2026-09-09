from flask import Flask, request, jsonify
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class GGUFModelServer:
    def __init__(self):
        self.model = None
        self.model_path = os.environ.get('MODEL_PATH', 'model.gguf')
        self.model_name = os.environ.get('MODEL_NAME', 'unknown')
        self.load_model()

    def load_model(self):
        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model from {self.model_path}...")
            start = time.time()
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=512,        # Small context, enough for health check
                n_threads=4,      # Limit threads for Jetson
                n_gpu_layers=0,   # CPU only — keeps Stage-1 simple
                verbose=False
            )
            load_time = time.time() - start
            logger.info(f"Model loaded in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Error loading GGUF model: {e}")

    def predict(self, prompt="Hello"):
        if self.model is None:
            return {"error": "Model not loaded"}
        start = time.time()
        # Minimal generation — just enough to confirm model works
        output = self.model(
            prompt,
            max_tokens=10,
            echo=False
        )
        latency_ms = (time.time() - start) * 1000
        return {
            "model": self.model_name,
            "inference_latency_ms": round(latency_ms, 3)
        }

model_server = GGUFModelServer()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "pong",
        "model_loaded": model_server.model is not None,
        "model_type": "GGUF",
        "model_name": model_server.model_name
    })

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'Hello') if not data.get('dummy') else 'Hello'
        result = model_server.predict(prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "model_type": "GGUF LLM",
        "model_name": model_server.model_name,
        "model_path": model_server.model_path,
        "loaded": model_server.model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)