"""
Crop Recommendation — Flask API
=================================
Endpoints:
  POST /api/predict        – predict best crop from soil/climate features
  GET  /api/crops          – list all supported crops
  GET  /api/model-info     – model metadata & performance stats
  GET  /api/health         – health check
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
try:
    from flask_cors import CORS
    _cors_available = True
except ImportError:
    _cors_available = False

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
if _cors_available:
    CORS(app)

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ── Load Artifacts ─────────────────────────────────────────────────────────────
def load_artifacts():
    """Load model, scaler, label encoder, and metadata from disk."""
    with open(os.path.join(MODEL_DIR, "model.pkl"),   "rb") as f: model   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),  "rb") as f: scaler  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f: encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:   meta    = json.load(f)
    return model, scaler, encoder, meta

try:
    model, scaler, encoder, meta = load_artifacts()
    print(f" Model loaded: {meta['best_model']} ({meta['test_accuracy']}% accuracy)")
except FileNotFoundError:
    model = scaler = encoder = meta = None
    print("  Model artifacts not found — run `python model/train_model.py` first.")


# ── Crop Info ──────────────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif",  "water": "High",   "desc": "Staple grain for over half the world's population."},
    "maize":       {"emoji": "🌽", "season": "Kharif",  "water": "Medium", "desc": "Versatile crop used for food, feed & biofuel."},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",    "water": "Low",    "desc": "High-protein legume, drought tolerant."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",  "water": "Medium", "desc": "Nutritious legume rich in protein & fibre."},
    "pigeonpeas":  {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "desc": "Hardy legume used in dals and stews."},
    "mothbeans":   {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "desc": "Drought-resistant bean grown in arid regions."},
    "mungbean":    {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "desc": "Quick-growing legume with high nutritional value."},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",  "water": "Medium", "desc": "High-protein bean used in South Asian cuisine."},
    "lentil":      {"emoji": "🫘", "season": "Rabi",    "water": "Low",    "desc": "Ancient legume and key protein source."},
    "pomegranate": {"emoji": "🍎", "season": "Annual",  "water": "Low",    "desc": "Antioxidant-rich fruit, drought tolerant."},
    "banana":      {"emoji": "🍌", "season": "Annual",  "water": "High",   "desc": "Tropical fruit, one of the most traded crops."},
    "mango":       {"emoji": "🥭", "season": "Summer",  "water": "Medium", "desc": "The king of fruits, loved worldwide."},
    "grapes":      {"emoji": "🍇", "season": "Annual",  "water": "Medium", "desc": "Used fresh, dried as raisins, or for wine."},
    "watermelon":  {"emoji": "🍉", "season": "Summer",  "water": "High",   "desc": "Refreshing summer fruit, 92% water content."},
    "muskmelon":   {"emoji": "🍈", "season": "Summer",  "water": "Medium", "desc": "Sweet aromatic melon grown in warm climates."},
    "apple":       {"emoji": "🍏", "season": "Annual",  "water": "Medium", "desc": "Temperate fruit, cultivated for 4000+ years."},
    "orange":      {"emoji": "🍊", "season": "Winter",  "water": "Medium", "desc": "Vitamin C-rich citrus, globally popular."},
    "papaya":      {"emoji": "🍑", "season": "Annual",  "water": "High",   "desc": "Fast-growing tropical fruit with health benefits."},
    "coconut":     {"emoji": "🥥", "season": "Annual",  "water": "High",   "desc": "Multi-use tropical palm, 'tree of life'."},
    "cotton":      {"emoji": "🌿", "season": "Kharif",  "water": "High",   "desc": "World's most important natural textile fibre."},
    "jute":        {"emoji": "🌿", "season": "Kharif",  "water": "High",   "desc": "Natural bast fibre, biodegradable packaging."},
    "coffee":      {"emoji": "☕", "season": "Annual",  "water": "Medium", "desc": "Most traded tropical commodity after crude oil."},
}


# ── Helpers ────────────────────────────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_RANGES = {
    "N":           (0,   140, "kg/ha"),
    "P":           (5,   145, "kg/ha"),
    "K":           (5,   205, "kg/ha"),
    "temperature": (8,   43,  "°C"),
    "humidity":    (14,  99,  "%"),
    "ph":          (3.5, 9.9, ""),
    "rainfall":    (20,  300, "mm"),
}

def validate_input(data: dict):
    errors = []
    values = {}
    for feat in FEATURES:
        if feat not in data:
            errors.append(f"Missing field: '{feat}'")
            continue
        try:
            v = float(data[feat])
        except (ValueError, TypeError):
            errors.append(f"'{feat}' must be a number")
            continue
        lo, hi, _ = FEATURE_RANGES[feat]
        if not (lo <= v <= hi):
            errors.append(f"'{feat}' must be between {lo} and {hi}")
        values[feat] = v
    return values, errors


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_name":   meta["best_model"] if meta else None,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(silent=True) or {}
    values, errors = validate_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    X = np.array([[values[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)

    pred_idx = model.predict(X_scaled)[0]
    crop     = encoder.inverse_transform([pred_idx])[0]

    # Confidence / probabilities
    top_n = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        top_idx = np.argsort(probs)[::-1][:5]
        top_n = [
            {"crop": encoder.inverse_transform([i])[0], "confidence": round(float(probs[i]) * 100, 2)}
            for i in top_idx
        ]

    info = CROP_INFO.get(crop, {})

    return jsonify({
        "recommended_crop": crop,
        "emoji":            info.get("emoji", "🌱"),
        "confidence":       top_n[0]["confidence"] if top_n else None,
        "top_5":            top_n,
        "crop_info":        info,
        "input_summary":    values,
    })


@app.route("/api/crops")
def list_crops():
    crops = [
        {"name": c, **CROP_INFO.get(c, {})}
        for c in (meta["crops"] if meta else list(CROP_INFO.keys()))
    ]
    return jsonify({"total": len(crops), "crops": crops})


@app.route("/api/model-info")
def model_info():
    if meta is None:
        return jsonify({"error": "Model metadata not available"}), 503
    return jsonify(meta)


@app.route("/api/feature-ranges")
def feature_ranges():
    return jsonify({
        k: {"min": v[0], "max": v[1], "unit": v[2]}
        for k, v in FEATURE_RANGES.items()
    })


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
