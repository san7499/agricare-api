import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json

app = Flask(__name__)

# ✅ FIXED MODEL LOADING
model = tf.keras.models.load_model(
    "final_efficientnetb3_model.keras",
    compile=False,
    safe_mode=False
)

# Load class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

class_labels = {v: k for k, v in class_indices.items()}

# ------------------ TREATMENT ------------------
treatment_suggestions = {

    # 🌽 CORN
    "Corn__Blight": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Mancozeb fungicide"],
        "Organic": ["Crop rotation", "Remove infected leaves"]
    },
    "Corn__Common_Rust": {
        "Fertilizer": ["Nitrogen-rich fertilizer"],
        "Pesticide": ["Propiconazole"],
        "Organic": ["Neem oil spray"]
    },
    "Corn__Gray_Leaf_Spot": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Proper field sanitation"]
    },
    "Corn__Healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Organic compost"]
    },

    # 🍇 GRAPE
    "Grape___Black_rot": {
        "Fertilizer": ["Phosphorus-rich fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Remove infected fruits"]
    },
    "Grape___Esca": {
        "Fertilizer": ["Organic manure"],
        "Pesticide": ["Thiophanate-methyl"],
        "Organic": ["Prune infected wood"]
    },
    "Grape___Leaf_blight": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Copper fungicide"],
        "Organic": ["Improve air circulation"]
    },
    "Grape___healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Mulching"]
    },

    # 🥭 MANGO
    "Mango__Gall_Midge": {
        "Fertilizer": ["Organic compost"],
        "Pesticide": ["Imidacloprid"],
        "Organic": ["Sticky traps"]
    },
    "Mango__Healthy": {
        "Fertilizer": ["Farmyard manure"],
        "Pesticide": ["Not required"],
        "Organic": ["Regular pruning"]
    },
    "Mango__Powdery_Mildew": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Sulfur fungicide"],
        "Organic": ["Neem oil spray"]
    },
    "Mango__Sooty_Mould": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Dimethoate"],
        "Organic": ["Control aphids"]
    },

    # 🥜 PEANUT
    "Peanut__early_leaf_spot": {
        "Fertilizer": ["Calcium-rich fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Crop rotation"]
    },
    "Peanut__early_rust": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Remove infected plants"]
    },
    "Peanut__healthy_leaf": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Organic compost"]
    },
    "Peanut__late_leaf_spot": {
        "Fertilizer": ["Phosphorus fertilizer"],
        "Pesticide": ["Tebuconazole"],
        "Organic": ["Proper spacing"]
    },
    "Peanut__nutrition_deficiency": {
        "Fertilizer": ["Micronutrient mixture"],
        "Pesticide": ["Not required"],
        "Organic": ["Soil testing"]
    },
    "Peanut__rust": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Hexaconazole"],
        "Organic": ["Neem oil spray"]
    },

    # 🌶️ PEPPER
    "Pepper__bell___Bacterial_spot": {
        "Fertilizer": ["Calcium nitrate"],
        "Pesticide": ["Copper fungicide"],
        "Organic": ["Remove infected leaves"]
    },
    "Pepper__bell___healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Compost manure"]
    },

    # 🥔 POTATO
    "Potato___Early_blight": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Neem oil spray"]
    },
    "Potato___Late_blight": {
        "Fertilizer": ["Phosphorus fertilizer"],
        "Pesticide": ["Metalaxyl"],
        "Organic": ["Remove infected plants"]
    },
    "Potato___healthy": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not required"],
        "Organic": ["Crop rotation"]
    },

    # 🍅 TOMATO
    "Tomato_Bacterial_spot": {
        "Fertilizer": ["Calcium-rich fertilizer"],
        "Pesticide": ["Copper fungicide"],
        "Organic": ["Neem oil spray"]
    },
    "Tomato_Early_blight": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Remove infected leaves"]
    },
    "Tomato_Late_blight": {
        "Fertilizer": ["Phosphorus fertilizer"],
        "Pesticide": ["Metalaxyl"],
        "Organic": ["Crop rotation"]
    },
    "Tomato_Leaf_Mold": {
        "Fertilizer": ["Nitrogen fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Improve ventilation"]
    },
    "Tomato_Septoria_leaf_spot": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Chlorothalonil"],
        "Organic": ["Remove infected leaves"]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "Fertilizer": ["Nitrogen fertilizer"],
        "Pesticide": ["Abamectin"],
        "Organic": ["Neem oil spray"]
    },
    "Tomato__Target_Spot": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Mancozeb"],
        "Organic": ["Crop sanitation"]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "Fertilizer": ["Potassium fertilizer"],
        "Pesticide": ["Imidacloprid (for whiteflies)"],
        "Organic": ["Yellow sticky traps"]
    },
    "Tomato__Tomato_mosaic_virus": {
        "Fertilizer": ["Balanced NPK fertilizer"],
        "Pesticide": ["Not effective"],
        "Organic": ["Remove infected plants"]
    },
    "Tomato_healthy": {
        "Fertilizer": ["Balanced NPK fertilizer (20:20:20)"],
        "Pesticide": ["Not required"],
        "Organic": ["Organic compost"]
    }
}


from tensorflow.keras.applications.efficientnet import preprocess_input

# ------------------ PREDICT FUNCTION ------------------
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
