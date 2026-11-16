import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

saved = joblib.load("AllModels.pkl")
features = saved["features"]

data = pd.read_csv("feature_data.csv")
test = data[data["split"] == "test"]
X_test, y_test = test[features], test["label"]

rows = []
for name, item in saved["bundle"].items():
    pred = (item["model"].predict_proba(X_test)[:, 1] > item["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    rows.append([name, (tp + tn) / len(y_test), tp / max(tp + fp, 1),
                 tp / max(tp + fn, 1), tn / max(tn + fp, 1), fp, fn])

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, pred, labels=[0, 1]), annot=True, fmt="d",
                xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix_" + name.lower().replace(" ", "_") + ".png")
    plt.close()

results = pd.DataFrame(rows, columns=[
    "Model", "Accuracy", "Precision", "Recall (Sensitivity)",
    "Specificity", "False Positives", "False Negatives"])
results.to_csv("model_comparison.csv", index=False)
print(results.round(3).to_string(index=False))