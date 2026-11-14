from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

app = Flask(__name__)

# Enable frontend-backend communication
CORS(app)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load MobileNetV2 model
model = tf.keras.models.load_model(
    MODEL_DIR / "animal_model_v2.keras"
)

# Load MobileNetV2 class names
with open(MODEL_DIR / "class_names_mobilenet.json", "r") as f:
    class_names = json.load(f)

print("MobileNetV2 model loaded successfully!")
print("Total animal classes:", len(class_names))


# Home route
@app.route("/")
def home():
    return "Animal Classification Backend is Running!"


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:
        # Read uploaded image
        img = Image.open(image_file.stream).convert("RGB")

        # Resize image
        img = img.resize((224, 224))

        # Convert image into array
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)

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

        # Best prediction
        predicted_animal = top_predictions[0]["animal"]
        confidence = top_predictions[0]["confidence"]

        print("Top predictions:", top_predictions)

        return jsonify({
            "prediction": predicted_animal,
            "confidence": confidence,
            "top_predictions": top_predictions
        })

    except Exception as e:
        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)

