"""
Crop Recommendation Model Training
====================================
Trains multiple ML classifiers on the Crop Recommendation dataset,
evaluates them, and saves the best model + preprocessing artifacts.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


# ── 1. Synthetic Dataset (mirrors Kaggle Crop Recommendation dataset) ──────────

def generate_dataset(n_samples: int = 2200, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic crop recommendation dataset."""
    rng = np.random.default_rng(seed)

    crops = {
        "rice":        dict(N=(60,100),  P=(30,60),  K=(30,60),  temp=(20,27), hum=(80,90), ph=(5.5,7.0), rain=(150,250)),
        "maize":       dict(N=(60,100),  P=(50,80),  K=(50,80),  temp=(18,27), hum=(55,75), ph=(5.8,7.0), rain=(50,100)),
        "chickpea":    dict(N=(30,60),   P=(60,90),  K=(70,100), temp=(18,29), hum=(15,30), ph=(6.0,8.0), rain=(30,60)),
        "kidneybeans": dict(N=(15,35),   P=(60,90),  K=(15,30),  temp=(15,22), hum=(18,28), ph=(5.5,7.0), rain=(80,120)),
        "pigeonpeas":  dict(N=(15,40),   P=(55,90),  K=(15,40),  temp=(18,29), hum=(40,60), ph=(5.5,7.0), rain=(50,100)),
        "mothbeans":   dict(N=(15,35),   P=(40,60),  K=(15,30),  temp=(24,34), hum=(25,50), ph=(3.5,6.5), rain=(30,60)),
        "mungbean":    dict(N=(15,40),   P=(40,60),  K=(15,30),  temp=(25,35), hum=(80,90), ph=(6.2,7.2), rain=(40,80)),
        "blackgram":   dict(N=(30,50),   P=(55,80),  K=(15,30),  temp=(24,35), hum=(60,80), ph=(5.5,7.5), rain=(40,80)),
        "lentil":      dict(N=(15,35),   P=(60,80),  K=(15,30),  temp=(15,25), hum=(65,75), ph=(6.0,7.0), rain=(25,60)),
        "pomegranate": dict(N=(15,20),   P=(10,18),  K=(35,50),  temp=(18,25), hum=(85,95), ph=(5.5,7.5), rain=(100,140)),
        "banana":      dict(N=(90,120),  P=(60,80),  K=(50,65),  temp=(25,30), hum=(75,85), ph=(5.5,7.0), rain=(100,150)),
        "mango":       dict(N=(15,25),   P=(10,18),  K=(30,50),  temp=(24,32), hum=(40,60), ph=(5.5,7.5), rain=(90,120)),
        "grapes":      dict(N=(15,25),   P=(10,25),  K=(30,50),  temp=(8,17),  hum=(80,90), ph=(5.5,7.5), rain=(65,80)),
        "watermelon":  dict(N=(80,120),  P=(10,15),  K=(45,60),  temp=(24,35), hum=(80,90), ph=(5.8,7.5), rain=(40,50)),
        "muskmelon":   dict(N=(90,110),  P=(10,15),  K=(50,60),  temp=(28,35), hum=(85,95), ph=(6.0,7.0), rain=(20,30)),
        "apple":       dict(N=(15,25),   P=(125,145),K=(195,210),temp=(20,24), hum=(90,95), ph=(5.5,7.0), rain=(100,120)),
        "orange":      dict(N=(15,20),   P=(5,12),   K=(8,20),   temp=(10,18), hum=(90,95), ph=(6.0,7.5), rain=(110,120)),
        "papaya":      dict(N=(45,55),   P=(55,65),  K=(40,55),  temp=(25,35), hum=(90,95), ph=(6.5,7.5), rain=(150,200)),
        "coconut":     dict(N=(5,10),    P=(5,10),   K=(30,50),  temp=(26,34), hum=(90,95), ph=(5.0,8.0), rain=(150,220)),
        "cotton":      dict(N=(100,140), P=(30,55),  K=(15,25),  temp=(24,28), hum=(55,65), ph=(6.0,7.5), rain=(60,100)),
        "jute":        dict(N=(60,80),   P=(40,60),  K=(38,55),  temp=(24,37), hum=(70,90), ph=(6.0,7.0), rain=(150,200)),
        "coffee":      dict(N=(80,120),  P=(15,25),  K=(15,35),  temp=(22,28), hum=(80,90), ph=(6.0,6.5), rain=(150,250)),
    }

    rows = []
    per_crop = n_samples // len(crops)
    for crop, r in crops.items():
        for _ in range(per_crop):
            rows.append({
                "N":           rng.uniform(*r["N"]),
                "P":           rng.uniform(*r["P"]),
                "K":           rng.uniform(*r["K"]),
                "temperature": rng.uniform(*r["temp"]),
                "humidity":    rng.uniform(*r["hum"]),
                "ph":          rng.uniform(*r["ph"]),
                "rainfall":    rng.uniform(*r["rain"]),
                "label":       crop,
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ── 2. Training Pipeline ───────────────────────────────────────────────────────

def train(save_dir: str = "model") -> None:
    os.makedirs(save_dir, exist_ok=True)

    print("📦  Generating dataset …")
    df = generate_dataset()
    df.to_csv(os.path.join(save_dir, "crop_data.csv"), index=False)
    print(f"    {len(df)} rows, {df['label'].nunique()} crops\n")

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Model zoo ────────────────────────────────────────────────────────────
    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "SVM":                 SVC(kernel="rbf", C=10, probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":       DecisionTreeClassifier(max_depth=15, random_state=42),
        "Naive Bayes":         GaussianNB(),
    }

    results = {}
    best_name, best_acc, best_model = "", 0.0, None

    print("🤖  Training models …\n")
    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        cv  = cross_val_score(clf, X_train_s, y_train, cv=5, scoring="accuracy").mean()
        results[name] = {"test_accuracy": round(acc * 100, 2), "cv_accuracy": round(cv * 100, 2)}
        print(f"  {name:<22}  test={acc*100:.2f}%   cv={cv*100:.2f}%")
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, clf

    print(f"\n✅  Best model: {best_name} ({best_acc*100:.2f}%)\n")

    # ── Detailed report for best model ───────────────────────────────────────
    y_pred_best = best_model.predict(X_test_s)
    report = classification_report(
        y_test, y_pred_best,
        target_names=le.classes_,
        output_dict=True
    )

    # ── Feature importances (RF / GB only) ───────────────────────────────────
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
        feature_importance = dict(zip(features, [round(float(v), 4) for v in fi]))

    # ── Save artifacts ────────────────────────────────────────────────────────
    with open(os.path.join(save_dir, "model.pkl"),   "wb") as f: pickle.dump(best_model, f)
    with open(os.path.join(save_dir, "scaler.pkl"),  "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(save_dir, "encoder.pkl"), "wb") as f: pickle.dump(le, f)

    meta = {
        "best_model":         best_name,
        "test_accuracy":      round(best_acc * 100, 2),
        "features":           features,
        "crops":              le.classes_.tolist(),
        "model_results":      results,
        "feature_importance": feature_importance,
        "classification_report": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
    }
    with open(os.path.join(save_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(" Artifacts saved:")
    for fn in ["model.pkl", "scaler.pkl", "encoder.pkl", "model_meta.json", "crop_data.csv"]:
        print(f"    model/{fn}")


if __name__ == "__main__":
    train()
