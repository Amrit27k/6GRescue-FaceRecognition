import numpy as np
from flask import Flask, request, jsonify
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class TFLiteModelServer:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()

    def load_model(self):
        model_path = os.environ.get('MODEL_PATH', 'model.tflite')
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Loaded TFLite model from {model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
        except Exception as e:
            logger.error(f"Error loading TFLite model: {e}")

    def predict(self, input_data):
        if self.interpreter is None:
            return {"error": "Model not loaded"}
        start = time.time()
        input_data = np.array(input_data, dtype=np.float32)
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data
        )
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        latency_ms = (time.time() - start) * 1000
        return {
            "output_shape": list(output.shape),
            "inference_latency_ms": round(latency_ms, 3)
        }

model_server = TFLiteModelServer()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "pong",
        "model_loaded": model_server.interpreter is not None,
        "model_type": "TFLite"
    })

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Accept either raw input_data array or dummy flag
        if data.get('dummy'):
            # For timing experiments - just run dummy inference
            input_shape = model_server.input_details[0]['shape']
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
            result = model_server.predict(dummy_input)
        else:
            input_data = data.get('input_data')
            result = model_server.predict(input_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)