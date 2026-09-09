import numpy as np
from flask import Flask, request, jsonify
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class ONNXModelServer:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()

    def load_model(self):
        model_path = os.environ.get('MODEL_PATH', 'model.onnx')
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Loaded ONNX model from {model_path}")
            logger.info(f"Input: {self.input_name}, shape: {input_shape}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")

    def predict(self, input_data):
        if self.session is None:
            return {"error": "Model not loaded"}
        start = time.time()
        input_array = np.array(input_data, dtype=np.float32)
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_array}
        )
        latency_ms = (time.time() - start) * 1000
        return {
            "output_shape": list(outputs[0].shape),
            "inference_latency_ms": round(latency_ms, 3)
        }

model_server = ONNXModelServer()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "pong",
        "model_loaded": model_server.session is not None,
        "model_type": "ONNX"
    })

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data.get('dummy'):
            input_shape = model_server.session.get_inputs()[0].shape
            # Replace dynamic dims with 1
            input_shape = [1 if isinstance(d, str) or d is None 
                          else d for d in input_shape]
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