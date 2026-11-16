
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

app = Flask(__name__)

# Allow the deployed frontend to communicate with the backend
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://animal-classification-1-0lng.onrender.com"
            ]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load existing animal classification model
model = tf.keras.models.load_model(
    MODEL_DIR / "animal_model_v2.keras"
)

# Load exact 64 dataset classes
with open(MODEL_DIR / "class_names_mobilenet.json", "r") as f:
    class_names = json.load(f)

DATASET_CLASSES = set(name.lower() for name in class_names)

# Load pretrained ImageNet screening model
screening_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True
)

# Map ImageNet descriptions to your dataset classes
ALIASES = {
    "tabby": "cat",
    "tiger_cat": "cat",
    "persian_cat": "cat",
    "siamese_cat": "cat",
    "egyptian_cat": "cat",
    "golden_retriever": "dog",
    "labrador_retriever": "dog",
    "german_shepherd": "dog",
    "poodle": "dog",
    "beagle": "dog",
    "rottweiler": "dog",
    "doberman": "dog",
    "boxer": "dog",
    "chihuahua": "dog",
    "pug": "dog",
    "african_elephant": "elephant",
    "indian_elephant": "elephant",
    "tusker": "elephant",
    "brown_bear": "bear",
    "american_black_bear": "bear",
    "ice_bear": "bear",
    "sloth_bear": "bear",
    "red_fox": "fox",
    "grey_fox": "fox",
    "arctic_fox": "fox",
    "timber_wolf": "wolf",
    "white_wolf": "wolf",
    "grey_wolf": "wolf",
    "african_hunting_dog": "hyena",
    "hyena": "hyena",
    "leopard": "leopard",
    "jaguar": "jaguar",
    "cheetah": "cheetah",
    "cougar": "lynx",
    "lynx": "lynx",
    "wallaby": "kangaroo",
    "koala": "koala",
    "chimpanzee": "chimpanzee",
    "gorilla": "gorilla",
    "orangutan": "chimpanzee",
    "macaque": "lemur",
    "sloth": "beaver",
    "otter": "otter",
    "beaver": "beaver",
    "mongoose": "mongoose",
    "meerkat": "mongoose",
    "red_panda": "panda",
    "giant_panda": "panda",
    "lesser_panda": "panda",
    "gazelle": "antelope",
    "impala": "antelope",
    "ibex": "antelope",
    "bighorn": "antelope",
    "hartebeest": "antelope",
    "warthog": "bison",
    "bison": "bison",
    "water_buffalo": "buffalo",
    "ox": "buffalo",
    "black_and_tan_coonhound": "dog",
    "african_grey": "parrot"
}

def identify_dataset_animal(description):

    label = description.lower().replace(" ", "_")

    # Check direct class match
    if label in DATASET_CLASSES:
        return label

    # Check known ImageNet aliases
    if label in ALIASES:
        mapped_class = ALIASES[label]

        if mapped_class in DATASET_CLASSES:
            return mapped_class

    # Check whether a dataset class appears in the label
    for animal in DATASET_CLASSES:
        if animal in label.split("_"):
            return animal

    return None


print("Animal model loaded successfully!")
print("Image screening model loaded successfully!")
print("Dataset classes:", len(class_names))


@app.route("/")
def home():
    return "Animal Classification Backend is Running!"


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
        img = img.resize((224, 224))

        # Prepare image for existing animal model
        img_array = np.array(img, dtype=np.float32)
        batch_array = np.expand_dims(img_array, axis=0)

        # Screen image using ImageNet
        screening_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            batch_array.copy()
        )

        screening_predictions = screening_model.predict(
            screening_array, verbose=0
        )

        # Get top 5 screening predictions
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
            screening_predictions,
            top=5
        )[0]

        print("Screening predictions:", decoded)

        # Check whether ImageNet identifies a dataset animal
        matched_animal = None

        for _, description, score in decoded:

            candidate = identify_dataset_animal(description)

            if candidate is not None and score >= 0.10:
                matched_animal = candidate
                break

        # Reject images not matching the dataset
        if matched_animal is None:

            return jsonify({
                "prediction": "Could Not Predict",
                "confidence": 0,
                "rejected": True,
                "message": "This image does not appear to be a recognizable animal from our dataset.",
                "top_predictions": []
            })

        # Run original animal classification model
        predictions = model.predict(batch_array, verbose=0)[0]

        best_index = int(np.argmax(predictions))

        confidence = round(
            float(predictions[best_index]) * 100, 2
        )

        predicted_animal = class_names[best_index]

        print("Predicted animal:", predicted_animal)
        print("Confidence:", confidence)

        # Confidence check
        if confidence < 60:

            return jsonify({
                "prediction": "Could Not Predict",
                "confidence": confidence,
                "rejected": True,
                "message": "Animal could not be recognized reliably.",
                "top_predictions": []
            })

        # Get top 5 animal predictions
        top_indices = np.argsort(predictions)[-5:][::-1]

        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "animal": class_names[int(index)],
                "confidence": round(
                    float(predictions[index]) * 100, 2
                )
            })

        return jsonify({
            "prediction": predicted_animal,
            "confidence": confidence,
            "rejected": False,
            "top_predictions": top_predictions
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
