"""
Crop Recommendation - Model Training Pipeline
=============================================
Trains and evaluates multiple ML classifiers, saves the best model.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "crop_recommendation.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "model")
PLOTS_DIR  = os.path.join(BASE_DIR, "..", "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Feature / Label names ──────────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA GENERATION  (used when real CSV is absent)

def generate_synthetic_data(n_samples: int = 2200) -> pd.DataFrame:
    """Generate realistic synthetic crop data."""
    np.random.seed(42)
    crops = {
        "rice":        dict(N=(60,100),  P=(30,60),   K=(30,60),   T=(20,30),  H=(70,90),  pH=(5.5,6.5), R=(150,300)),
        "maize":       dict(N=(60,100),  P=(40,70),   K=(40,70),   T=(18,27),  H=(55,75),  pH=(5.8,7.0), R=(50,120)),
        "chickpea":    dict(N=(0,40),    P=(40,80),   K=(40,80),   T=(15,25),  H=(14,25),  pH=(6.0,8.0), R=(30,80)),
        "kidneybeans": dict(N=(0,40),    P=(40,80),   K=(40,80),   T=(15,27),  H=(18,25),  pH=(5.5,7.0), R=(50,150)),
        "pigeonpeas":  dict(N=(0,40),    P=(40,80),   K=(40,80),   T=(18,30),  H=(30,70),  pH=(5.0,7.0), R=(60,150)),
        "mothbeans":   dict(N=(0,40),    P=(30,60),   K=(30,60),   T=(24,35),  H=(25,65),  pH=(3.5,6.5), R=(30,80)),
        "mungbean":    dict(N=(0,40),    P=(30,60),   K=(30,60),   T=(25,35),  H=(80,90),  pH=(6.2,7.2), R=(40,100)),
        "blackgram":   dict(N=(30,60),   P=(40,70),   K=(30,60),   T=(25,35),  H=(60,80),  pH=(5.0,7.5), R=(50,100)),
        "lentil":      dict(N=(0,30),    P=(40,80),   K=(20,60),   T=(15,25),  H=(60,80),  pH=(6.0,8.0), R=(30,80)),
        "pomegranate": dict(N=(0,30),    P=(10,30),   K=(40,70),   T=(20,35),  H=(85,95),  pH=(5.5,7.5), R=(110,200)),
        "banana":      dict(N=(80,120),  P=(60,100),  K=(40,70),   T=(22,30),  H=(70,90),  pH=(5.5,7.0), R=(100,200)),
        "mango":       dict(N=(0,20),    P=(10,30),   K=(30,60),   T=(24,35),  H=(45,65),  pH=(5.5,7.5), R=(50,120)),
        "grapes":      dict(N=(10,30),   P=(100,150), K=(190,240), T=(8,42),   H=(80,90),  pH=(5.5,7.0), R=(50,80)),
        "watermelon":  dict(N=(80,130),  P=(10,30),   K=(40,70),   T=(24,35),  H=(80,90),  pH=(6.0,7.0), R=(40,100)),
        "muskmelon":   dict(N=(80,130),  P=(10,30),   K=(50,80),   T=(28,38),  H=(90,100), pH=(6.0,7.0), R=(20,50)),
        "apple":       dict(N=(0,20),    P=(100,150), K=(130,180), T=(0,25),   H=(90,100), pH=(5.5,6.5), R=(100,200)),
        "orange":      dict(N=(0,20),    P=(5,20),    K=(5,20),    T=(10,30),  H=(90,100), pH=(6.0,7.5), R=(100,200)),
        "papaya":      dict(N=(40,70),   P=(5,20),    K=(40,70),   T=(25,40),  H=(90,100), pH=(6.5,7.5), R=(100,200)),
        "coconut":     dict(N=(0,30),    P=(0,30),    K=(30,60),   T=(20,35),  H=(90,100), pH=(5.0,8.0), R=(100,300)),
        "cotton":      dict(N=(100,140), P=(20,60),   K=(20,60),   T=(24,30),  H=(75,85),  pH=(6.0,8.0), R=(60,120)),
        "jute":        dict(N=(60,100),  P=(30,60),   K=(30,60),   T=(24,35),  H=(70,90),  pH=(6.0,7.5), R=(150,250)),
        "coffee":      dict(N=(80,120),  P=(30,60),   K=(30,60),   T=(20,30),  H=(55,65),  pH=(6.0,7.0), R=(150,250)),
    }
    n_each  = n_samples // len(crops)
    records = []
    for crop, r in crops.items():
        for _ in range(n_each):
            records.append({
                "N":           np.random.uniform(*r["N"]),
                "P":           np.random.uniform(*r["P"]),
                "K":           np.random.uniform(*r["K"]),
                "temperature": np.random.uniform(*r["T"]),
                "humidity":    np.random.uniform(*r["H"]),
                "ph":          np.random.uniform(*r["pH"]),
                "rainfall":    np.random.uniform(*r["R"]),
                "label":       crop,
            })
    df = pd.DataFrame(records)
    # add small Gaussian noise
    for col in FEATURES:
        df[col] += np.random.normal(0, df[col].std() * 0.03, len(df))
        df[col] = df[col].clip(lower=0)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TRAINING

def train():
    # ── Load or generate data ──────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[INFO] Loaded dataset: {DATA_PATH}  ({len(df)} rows)")
    else:
        print("[INFO] Dataset not found – generating synthetic data …")
        df = generate_synthetic_data()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"[INFO] Saved synthetic dataset → {DATA_PATH}")

    # ── Pre-processing ─────────────────────────────────────────────────────
    X = df[FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET].values)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Models to compare ─────────────────────────────────────────────────
    models = {
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(max_depth=10, random_state=42),
        "KNN":                  KNeighborsClassifier(n_neighbors=5),
        "SVM":                  SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "Naive Bayes":          GaussianNB(),
    }

    results   = {}
    best_acc  = 0.0
    best_name = ""
    best_model = None

    print("\n{:<22} {:>10} {:>10} {:>10}".format("Model", "Train Acc", "Test Acc", "CV Mean"))
    print("─" * 56)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc  = accuracy_score(y_test,  clf.predict(X_test))
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy", n_jobs=-1)
        f1        = f1_score(y_test, clf.predict(X_test), average="weighted")

        results[name] = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy":  round(test_acc,  4),
            "cv_mean":        round(cv_scores.mean(), 4),
            "cv_std":         round(cv_scores.std(),  4),
            "f1_score":       round(f1, 4),
        }
        print(f"{name:<22} {train_acc:>10.4f} {test_acc:>10.4f} {cv_scores.mean():>10.4f}")

        if test_acc > best_acc:
            best_acc   = test_acc
            best_name  = name
            best_model = clf

    print(f"\n✅  Best model: {best_name}  (test accuracy = {best_acc:.4f})")

    # ── Classification report ──────────────────────────────────────────────
    y_pred  = best_model.predict(X_test)
    report  = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_model_comparison(results)
    _plot_confusion_matrix(y_test, y_pred, le.classes_, best_name)
    _plot_feature_importance(best_model, best_name)

    # ── Save artefacts ─────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb")  as f: pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f: pickle.dump(scaler,     f)
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as f: pickle.dump(le,        f)

    meta = {
        "best_model":     best_name,
        "test_accuracy":  round(best_acc, 4),
        "crops":          le.classes_.tolist(),
        "features":       FEATURES,
        "n_classes":      int(len(le.classes_)),
        "model_results":  results,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n📦  Saved: model.pkl · scaler.pkl · encoder.pkl · metadata.json")
    print("🖼   Plots saved to static/plots/")
    return meta


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _plot_model_comparison(results: dict):
    names   = list(results.keys())
    accs    = [v["test_accuracy"] for v in results.values()]
    f1s     = [v["f1_score"]      for v in results.values()]
    cv_means = [v["cv_mean"]      for v in results.values()]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - 0.25, accs,     0.25, label="Test Accuracy", color="#2ECC71")
    bars2 = ax.bar(x,         f1s,     0.25, label="F1 Score",      color="#3498DB")
    bars3 = ax.bar(x + 0.25,  cv_means, 0.25, label="CV Mean",     color="#9B59B6")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in [*bars1, *bars2, *bars3]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, classes, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn",
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_feature_importance(model, model_name: str):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(importances[idx] / importances.max())
    bars   = ax.barh([FEATURES[i] for i in idx], importances[idx], color=colors)
    ax.set_title(f"Feature Importances – {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    for bar, val in zip(bars, importances[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    train()
