
import os

# Limit TensorFlow CPU usage
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import traceback
from PIL import Image
from pathlib import Path

app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Allow your deployed frontend
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://animal-classification-1-0lng.onrender.com"
            ]
        }
    }
)

# Limit TensorFlow CPU threads
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load model
model = tf.keras.models.load_model(
    MODEL_DIR / "animal_model_v2.keras"
)

# Load class names
with open(MODEL_DIR / "class_names_mobilenet.json", "r") as f:
    class_names = json.load(f)

print("MobileNetV2 model loaded successfully!")
print("Total animal classes:", len(class_names))


# Home route
@app.route("/", methods=["GET"])
def home():
    return "Animal Classification Backend is Running!"


# Prediction route
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        return "", 200

    print("Prediction request received!")

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read and process image
        img = Image.open(image_file.stream).convert("RGB")

        # Resize image
        img = img.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert to NumPy array
        img_array = np.asarray(img, dtype=np.float32)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Run direct model inference
        predictions = model(
            img_array,
            training=False
        ).numpy()

        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]

        top_predictions = []

        for index in top_indices:
            top_predictions.append({
                "animal": class_names[int(index)],
                "confidence": round(
                    float(predictions[0][index]) * 100, 2
                )
            })

        predicted_animal = top_predictions[0]["animal"]
        confidence = top_predictions[0]["confidence"]

        print("Prediction successful!")
        print("Top predictions:", top_predictions)

        return jsonify({
            "prediction": predicted_animal,
            "confidence": confidence,
            "top_predictions": top_predictions
        }), 200

    except Exception as e:
        print("Prediction error:", str(e))
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# Handle oversized uploads
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run()
