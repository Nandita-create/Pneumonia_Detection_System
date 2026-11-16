import re

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import FEATURE_COLUMNS as FEATURES

# Minimum fraction of NORMAL images that must be called NORMAL
# (0.97 -> about 3% false positives). Raise it for fewer false positives,
# at the cost of missing more pneumonia cases.
TARGET_SPECIFICITY = 0.97


def patient_id(name):
    """Images of the same patient must never be split across folds."""
    for pat in (r"(person\d+)", r"(IM-\d+)"):
        m = re.search(pat, name, re.I)
        if m:
            return m.group(1).lower()
    return name


def pick_threshold(y_true, p, target_spec):
    """Smallest threshold that keeps specificity >= target.
    A sample is called PNEUMONIA only if p > threshold."""
    return float(np.quantile(p[y_true == 0], target_spec, method="higher"))


def metrics(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision": tp / max(tp + fp, 1),
        "Recall": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
    }


data = pd.read_csv("feature_data.csv")
train = data[data["split"] == "train"].reset_index(drop=True)
test = data[data["split"] == "test"].reset_index(drop=True)

X_train, y_train = train[FEATURES], train["label"].values
X_test, y_test = test[FEATURES], test["label"].values
groups = train["image_name"].map(patient_id)

models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, class_weight="balanced",
            probability=True, random_state=42)),
    "Random Forest": make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
    "Gradient Boosting": make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            class_weight="balanced", max_depth=4,
            learning_rate=0.05, random_state=42)),
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

rows, bundle = [], {}

for name, pipe in models.items():
    print(f"Training {name} ...")

    # Out-of-fold probabilities (patient-grouped) -> honest threshold choice
    oof = cross_val_predict(
        pipe, X_train, y_train, cv=cv, groups=groups, method="predict_proba"
    )[:, 1]

    threshold = pick_threshold(y_train, oof, TARGET_SPECIFICITY)
    cv_recall = metrics(y_train, (oof > threshold).astype(int))["Recall"]

    pipe.fit(X_train, y_train)
    pred = (pipe.predict_proba(X_test)[:, 1] > threshold).astype(int)
    m = metrics(y_test, pred)

    rows.append({"Model": name, "Threshold": threshold,
                 "CV_Recall_at_target": cv_recall, **m})
    bundle[name] = {"model": pipe, "threshold": threshold}

results = pd.DataFrame(rows)
results.to_csv("model_comparison.csv", index=False)

# Choose the best model using TRAINING-fold results only (not the test set)
best_name = results.loc[results["CV_Recall_at_target"].idxmax(), "Model"]

joblib.dump(
    {
        "model": bundle[best_name]["model"],
        "threshold": bundle[best_name]["threshold"],
        "features": FEATURES,
        "model_name": best_name,
        "target_specificity": TARGET_SPECIFICITY,
    },
    "BestModel.pkl",
)
joblib.dump({"bundle": bundle, "features": FEATURES}, "AllModels.pkl")

pd.set_option("display.width", 200)
print(results.round(3).to_string(index=False))
print("\nBest model:", best_name)