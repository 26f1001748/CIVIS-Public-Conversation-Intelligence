
"""
PROJECT: CIVIS
MODULE: PrepareTableauMetrics.py

PURPOSE:
Export current validation performance metrics
for visualisation in Tableau.

INPUT:
models/sentiment_model.joblib
data/X_validation.npz
data/validation.csv

OUTPUT:
tableau_data/classification_metrics.csv
tableau_data/confusion_matrix.csv
"""

import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score
)

# 1. Define labels

LABELS = ["negative", "neutral", "positive"]

# 2. Load latest model and validation data

model = joblib.load("models/sentiment_model.joblib")

X_validation = load_npz("data/X_validation.npz")

df = pd.read_csv("data/validation.csv")

y_actual = df["sentiment"]
y_predicted = model.predict(X_validation)

# 3. Calculate class-wise metrics

precision, recall, f1, support = precision_recall_fscore_support(
    y_actual,
    y_predicted,
    labels=LABELS,
    zero_division=0
)

metrics_df = pd.DataFrame({
    "Sentiment": LABELS,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Support": support
})

# 4. Calculate overall metrics

accuracy = accuracy_score(y_actual, y_predicted)

print("Validation accuracy:", round(accuracy, 4))

# 5. Generate confusion matrix

cm = confusion_matrix(
    y_actual,
    y_predicted,
    labels=LABELS
)

confusion_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
)

confusion_df.index.name = "Actual Sentiment"

confusion_df = confusion_df.reset_index()

# 6. Save Tableau files

output_dir = Path("tableau_data")
output_dir.mkdir(exist_ok=True)

metrics_df.to_csv(
    output_dir / "classification_metrics.csv",
    index=False
)

confusion_df.to_csv(
    output_dir / "confusion_matrix.csv",
    index=False
)

# 7. Display results

print("\nCLASSIFICATION METRICS")
print(metrics_df.to_string(index=False))

print("\nCONFUSION MATRIX")
print(confusion_df.to_string(index=False))

print("\nFiles generated:")
print("tableau_data/classification_metrics.csv")
print("tableau_data/confusion_matrix.csv")
