import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json

app = Flask(__name__)

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model("final_efficientnetb3_model.keras")

# Load class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

class_labels = {v: k for k, v in class_indices.items()}

# ------------------ TREATMENT ------------------
treatment_suggestions = {
    "Corn__Blight": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Mancozeb fungicide"],
        "Organic": ["Crop rotation"]
    },
    "Corn__Healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Organic compost"]
    }
    # 👉 You can paste full dictionary here (optional)
}

# ------------------ PREDICTION FUNCTION ------------------
from tensorflow.keras.applications.efficientnet import preprocess_input

def predict_disease(img):
    img = Image.open(img)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((300, 300))

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))

    label = class_labels[idx]
    confidence = float(preds[idx])

    treatment = treatment_suggestions.get(label, {
        "Fertilizer": ["Consult expert"],
        "Pesticide": ["Consult expert"],
        "Organic": ["Consult expert"]
    })

    return label, confidence, treatment


# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return "AgriCare API Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        label, confidence, treatment = predict_disease(file)

        return jsonify({
            "disease": label,
            "confidence": round(confidence * 100, 2),
            "fertilizer": treatment["Fertilizer"],
            "pesticide": treatment["Pesticide"],
            "organic": treatment["Organic"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)