"""
PROJECT: CIVIS
MODULE: PretrainedModel.py

PURPOSE:
Test a pretrained tweet sentiment model on the validation data.
No training is done here. We only measure the score and
compare it with the TF-IDF baseline.

INPUT:
data/validation.csv

OUTPUT:
reports/pretrained_results.txt
reports/pretrained_confusion_matrix.png

INSTALL (once):
pip install transformers torch
"""

import re
import html
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]

Path("reports").mkdir(exist_ok=True)


# 1. Prepare text the way this model expects:
#    usernames -> @user, links -> http

def prepare(text):
    text = html.unescape(str(text))
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    return text.strip()


validation_df = pd.read_csv("data/validation.csv")
texts = [prepare(t) for t in validation_df["text"]]
y_true = validation_df["sentiment"]


# 2. Load the model (uses the GPU if available)

device = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device == 0 else "Using CPU (this will be slower)")

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    device=device,
    truncation=True,
    max_length=128
)


# 3. Predict

print("Predicting on validation data...")
outputs = classifier(texts, batch_size=32)
y_pred = [o["label"].lower() for o in outputs]


# 4. Evaluate

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

report = classification_report(
    y_true, y_pred, labels=LABELS, zero_division=0
)

print("\nPRETRAINED MODEL RESULTS")
print("=" * 45)
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print(report)


# 5. Confusion matrix

cm = confusion_matrix(y_true, y_pred, labels=LABELS)
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
    cmap="Blues", values_format="d"
)
plt.title("Pretrained Model - Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/pretrained_confusion_matrix.png", dpi=150)
plt.show()


# 6. Save results

with open("reports/pretrained_results.txt", "w") as file:
    file.write(f"Model: {MODEL_NAME}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n\n")
    file.write(report)

print("\nResults saved.")