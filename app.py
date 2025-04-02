from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
MODEL_PATH = os.path.join(config.CONFIG['data_dir'], config.CONFIG['final_model_filename'])
model = None

def get_model():
    global model
    if model is None:
        logger.info("Loading model from path: %s", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

dt = config.CONFIG['dt']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        model = get_model()
        data = request.get_json(force=True)

        if not isinstance(data, dict):
            logger.error("Invalid JSON format: expected a dictionary")
            return jsonify({"error": "Invalid JSON format"}), 400

        required_fields = ["input"]
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        input_data = data.get("input")
        if input_data is None:
            logger.error("No input data provided.")
            return jsonify({"error": "No input data provided."}), 400

        input_array = np.array(input_data, dtype=np.float32)
        expected_shape = (None, 5)
        if len(input_array.shape) != 2 or input_array.shape[1] != expected_shape[1]:
            logger.error("Invalid input shape: %s", input_array.shape)
            return jsonify({"error": f"Input must have shape {expected_shape}"}), 400

        power_pred = model.predict(input_array)
        energy_pred = power_pred * dt
        
        logger.info("Prediction successful, input shape: %s", input_array.shape)
        
        return jsonify({
            "power_predictions": power_pred.tolist(),
            "energy_predictions": energy_pred.tolist(),
            "time_step_seconds": dt
        })
    except ValueError as e:
        logger.error("Invalid input format: %s", str(e), exc_info=True)
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
    except tf.errors.OpError as e:
        logger.error("TensorFlow model error: %s", str(e), exc_info=True)
        return jsonify({"error": f"TensorFlow model error: {str(e)}"}), 500
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        model = get_model()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/')
def index():
    return "Energy Prediction Model API is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
