#  CropSense-Intelligent Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A production-ready machine learning web application that recommends the optimal crop based on soil nutrients and climate conditions.

---

## Features

- **Multi-model training** — 6 algorithms trained and compared automatically
- **REST API** — clean Flask endpoints with full input validation
- **Interactive UI** — responsive web interface with animated confidence bars
- **Top-5 predictions** — confidence scores for the top 5 candidate crops
- **Model transparency** — `/api/model-info` exposes accuracy, CV scores, and feature importances
- **22 crops** supported across cereals, legumes, fruits and cash crops

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10, Flask 3 |
| ML | scikit-learn 1.5 |
| Data | NumPy, Pandas |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
crop-recommendation/
├── app.py                  # Flask API & routes
├── requirements.txt
├── README.md
├── model/
│   ├── train_model.py      # Training pipeline
│   ├── model.pkl           # Saved best model  (generated)
│   ├── scaler.pkl          # StandardScaler    (generated)
│   ├── encoder.pkl         # LabelEncoder      (generated)
│   ├── model_meta.json     # Performance stats (generated)
│   └── crop_data.csv       # Training data     (generated)
└── templates/
    └── index.html          # Web UI
```

---

## Setup & Run

```bash
# 1. Clone & enter directory
git clone https://github.com/YOUR_USERNAME/crop-recommendation.git
cd crop-recommendation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates model artifacts)
python model/train_model.py

# 5. Start the server
python app.py
```

Open **http://localhost:5000**

---

## API Reference

### `POST /api/predict`
```json
{
  "N": 80, "P": 45, "K": 40,
  "temperature": 24.5,
  "humidity": 85.0,
  "ph": 6.2,
  "rainfall": 200.0
}
```
Returns recommended crop, confidence score, top-5 candidates, and crop metadata.

### `GET /api/model-info` — Performance metrics & feature importances
### `GET /api/crops` — All 22 supported crops
### `GET /api/health` — Health check

---

## Input Features

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| `N` | Nitrogen | 0–140 | kg/ha |
| `P` | Phosphorus | 5–145 | kg/ha |
| `K` | Potassium | 5–205 | kg/ha |
| `temperature` | Avg temperature | 8–43 | °C |
| `humidity` | Relative humidity | 14–99 | % |
| `ph` | Soil pH | 3.5–9.9 | — |
| `rainfall` | Annual rainfall | 20–300 | mm |

---

## Model Comparison

| Model | Test Accuracy | CV Accuracy |
|-------|:---:|:---:|
| **Random Forest** | **~99%** | **~99%** |
| Gradient Boosting | ~98% | ~98% |
| SVM | ~97% | ~97% |
| KNN | ~96% | ~96% |
| Decision Tree | ~95% | ~95% |
| Naive Bayes | ~88% | ~87% |

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Author

**Randa LAKAB** · [LinkedIn](https://www.linkedin.com/in/randa-lakab-4b9125389/) · [GitHub](https://github.com/Randa-Lakab)

> Built as a Data Science portfolio project demonstrating end-to-end ML deployment with Flask.
