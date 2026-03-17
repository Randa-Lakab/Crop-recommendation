"""
Crop Recommendation API — Flask Application
============================================
Endpoints
---------
GET  /                  → Web UI
GET  /api/health        → Health check
GET  /api/model-info    → Model metadata & performance
POST /api/predict       → Single prediction
POST /api/predict/batch → Batch predictions (JSON array)
GET  /api/crops         → List of supported crops
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "model")
FEATURES   = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Load artefacts ─────────────────────────────────────────────────────────
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "model.pkl"),   "rb") as f: model   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),  "rb") as f: scaler  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f: encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metadata.json"))     as f: meta    = json.load(f)
    return model, scaler, encoder, meta

try:
    model, scaler, encoder, meta = load_artifacts()
    print(f"[INFO] Model loaded: {meta['best_model']}  "
          f"(accuracy={meta['test_accuracy']}  crops={meta['n_classes']})")
except FileNotFoundError:
    print("[WARN] Trained model not found. Run  python model/train.py  first.")
    model = scaler = encoder = meta = None


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
VALID_RANGES = {
    "N":           (0,   200,  "mg/kg"),
    "P":           (0,   200,  "mg/kg"),
    "K":           (0,   300,  "mg/kg"),
    "temperature": (-10,  55,  "°C"),
    "humidity":    (0,   100,  "%"),
    "ph":          (0,    14,  ""),
    "rainfall":    (0,   500,  "mm"),
}

CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif",  "water": "High"},
    "maize":       {"emoji": "🌽", "season": "Kharif",  "water": "Medium"},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",    "water": "Low"},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",  "water": "Medium"},
    "pigeonpeas":  {"emoji": "🌿", "season": "Kharif",  "water": "Low"},
    "mothbeans":   {"emoji": "🌱", "season": "Kharif",  "water": "Low"},
    "mungbean":    {"emoji": "🫛", "season": "Kharif",  "water": "Medium"},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",  "water": "Low"},
    "lentil":      {"emoji": "🫘", "season": "Rabi",    "water": "Low"},
    "pomegranate": {"emoji": "🍎", "season": "Perennial","water": "Low"},
    "banana":      {"emoji": "🍌", "season": "Perennial","water": "High"},
    "mango":       {"emoji": "🥭", "season": "Perennial","water": "Medium"},
    "grapes":      {"emoji": "🍇", "season": "Perennial","water": "Medium"},
    "watermelon":  {"emoji": "🍉", "season": "Summer",  "water": "Medium"},
    "muskmelon":   {"emoji": "🍈", "season": "Summer",  "water": "Medium"},
    "apple":       {"emoji": "🍎", "season": "Winter",  "water": "Medium"},
    "orange":      {"emoji": "🍊", "season": "Winter",  "water": "Medium"},
    "papaya":      {"emoji": "🍈", "season": "Perennial","water": "Medium"},
    "coconut":     {"emoji": "🥥", "season": "Perennial","water": "High"},
    "cotton":      {"emoji": "🪴", "season": "Kharif",  "water": "Medium"},
    "jute":        {"emoji": "🌿", "season": "Kharif",  "water": "High"},
    "coffee":      {"emoji": "☕", "season": "Perennial","water": "Medium"},
}

def validate_input(data: dict) -> tuple[dict | None, str | None]:
    """Return (cleaned_dict, error_message). One of them will be None."""
    cleaned = {}
    for feat in FEATURES:
        if feat not in data:
            return None, f"Missing field: '{feat}'"
        try:
            val = float(data[feat])
        except (ValueError, TypeError):
            return None, f"'{feat}' must be a number, got: {data[feat]!r}"
        lo, hi, unit = VALID_RANGES[feat]
        if not (lo <= val <= hi):
            return None, f"'{feat}' out of range [{lo}–{hi} {unit}], got {val}"
        cleaned[feat] = val
    return cleaned, None


def predict_one(cleaned: dict) -> dict:
    X = np.array([[cleaned[f] for f in FEATURES]])
    X_scaled  = scaler.transform(X)
    proba     = model.predict_proba(X_scaled)[0]
    top3_idx  = np.argsort(proba)[::-1][:3]
    top_crop  = encoder.classes_[top3_idx[0]]
    info      = CROP_INFO.get(top_crop, {})
    return {
        "crop":        top_crop,
        "confidence":  round(float(proba[top3_idx[0]]) * 100, 2),
        "emoji":       info.get("emoji", "🌱"),
        "season":      info.get("season", "N/A"),
        "water_needs": info.get("water", "N/A"),
        "top3": [
            {
                "crop":       encoder.classes_[i],
                "confidence": round(float(proba[i]) * 100, 2),
                "emoji":      CROP_INFO.get(encoder.classes_[i], {}).get("emoji", "🌱"),
            }
            for i in top3_idx
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html", meta=meta)


@app.route("/api/health")
def health():
    return jsonify({
        "status":    "ok" if model else "model_not_loaded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model":     meta["best_model"] if meta else None,
    })


@app.route("/api/model-info")
def model_info():
    if not meta:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(meta)


@app.route("/api/crops")
def crops():
    if not encoder:
        return jsonify({"error": "Model not loaded"}), 503
    crops_list = [
        {**{"name": c}, **CROP_INFO.get(c, {})}
        for c in encoder.classes_
    ]
    return jsonify({"crops": crops_list, "total": len(crops_list)})


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    cleaned, err = validate_input(data)
    if err:
        return jsonify({"error": err}), 422

    result = predict_one(cleaned)
    result["input"] = cleaned
    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    if not model:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(force=True, silent=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of objects"}), 400
    if len(data) > 500:
        return jsonify({"error": "Batch size limit is 500"}), 400

    results = []
    for i, row in enumerate(data):
        cleaned, err = validate_input(row)
        if err:
            results.append({"index": i, "error": err})
        else:
            r = predict_one(cleaned)
            r["index"] = i
            results.append(r)

    return jsonify({"count": len(results), "results": results})


@app.route("/static/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(os.path.join(BASE_DIR, "..", "static", "plots"), filename)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
