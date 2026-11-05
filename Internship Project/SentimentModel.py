
"""
PROJECT: CIVIS
MODULE: SentimentModel.py

PURPOSE:
Train and compare Logistic Regression and LinearSVC
models for three-class sentiment classification.

Model selection is based on validation macro-F1.

INPUT:
data/X_train.npz
data/X_validation.npz
data/cleaned_train.csv
data/validation.csv

OUTPUT:
models/sentiment_model.joblib
reports/validation_results.txt
reports/tuning_results.csv
reports/confusion_matrix.png

IMPORTANT:
- Models are fitted using training data only.
- Validation data is used for model selection.
- Test data is not used in training or selection.
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# STEP 2: CONFIGURATION
# ============================================================

LABELS = ["negative", "neutral", "positive"]

Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)


# ============================================================
# STEP 3: LOAD FEATURE MATRICES
# ============================================================

print("Loading feature matrices...")

X_train = load_npz("data/X_train.npz")
X_validation = load_npz("data/X_validation.npz")


# ============================================================
# STEP 4: LOAD SENTIMENT LABELS
# ============================================================

print("Loading sentiment labels...")

train_df = pd.read_csv("data/cleaned_train.csv")
validation_df = pd.read_csv("data/validation.csv")

y_train = train_df["sentiment"]
y_validation = validation_df["sentiment"]


# ============================================================
# STEP 5: VERIFY DATA ALIGNMENT
# ============================================================

if X_train.shape[0] != len(y_train):
    raise ValueError(
        "Training feature and label counts do not match."
    )

if X_validation.shape[0] != len(y_validation):
    raise ValueError(
        "Validation feature and label counts do not match."
    )

if set(y_train.unique()) != set(LABELS):
    raise ValueError("Unexpected sentiment labels in training data.")

if set(y_validation.unique()) != set(LABELS):
    raise ValueError("Unexpected sentiment labels in validation data.")

print("Training rows:", X_train.shape[0])
print("Validation rows:", X_validation.shape[0])


# ============================================================
# STEP 6: TRAIN AND COMPARE MODELS
# ============================================================

print("\nTraining and comparing models...\n")

results = []
trained_models = {}


# ------------------------------------------------------------
# 6A. LOGISTIC REGRESSION
# ------------------------------------------------------------

for class_weight in [None, "balanced"]:

    for C in [0.1, 0.5, 1, 3, 10]:

        name = f"LogReg | C={C} | class_weight={class_weight}"

        model_candidate = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )

        model_candidate.fit(X_train, y_train)

        predictions = model_candidate.predict(X_validation)

        results.append({
            "model": name,
            "accuracy": accuracy_score(
                y_validation, predictions
            ),
            "macro_f1": f1_score(
                y_validation,
                predictions,
                average="macro"
            )
        })

        trained_models[name] = model_candidate


# ------------------------------------------------------------
# 6B. LINEAR SUPPORT VECTOR CLASSIFIER
# ------------------------------------------------------------

for class_weight in [None, "balanced"]:

    for C in [0.1, 0.5, 1]:

        name = f"LinearSVC | C={C} | class_weight={class_weight}"

        model_candidate = LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=42
        )

        model_candidate.fit(X_train, y_train)

        predictions = model_candidate.predict(X_validation)

        results.append({
            "model": name,
            "accuracy": accuracy_score(
                y_validation, predictions
            ),
            "macro_f1": f1_score(
                y_validation,
                predictions,
                average="macro"
            )
        })

        trained_models[name] = model_candidate


# ============================================================
# STEP 7: COMPARE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False
).reset_index(drop=True)

results_df = results_df.round(4)

print("\nMODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/tuning_results.csv",
    index=False
)


# ============================================================
# STEP 8: SELECT BEST MODEL
# ============================================================

best_name = results_df.loc[0, "model"]

model = trained_models[best_name]

print("\nSelected model:", best_name)


# ============================================================
# STEP 9: DETAILED VALIDATION EVALUATION
# ============================================================

y_predictions = model.predict(X_validation)

accuracy = accuracy_score(
    y_validation,
    y_predictions
)

macro_f1 = f1_score(
    y_validation,
    y_predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_validation,
    y_predictions,
    average="weighted"
)

report = classification_report(
    y_validation,
    y_predictions,
    labels=LABELS,
    zero_division=0
)

print("\nVALIDATION RESULTS")
print("=" * 55)

print("Selected model:", best_name)
print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))
print("Weighted-F1:", round(weighted_f1, 4))

print("\nClassification Report:")
print(report)


# ============================================================
# STEP 10: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_validation,
    y_predictions,
    labels=LABELS
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("CIVIS - Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 11: SAVE BEST MODEL
# ============================================================

model_path = "models/sentiment_model.joblib"

joblib.dump(
    model,
    model_path
)


# ============================================================
# STEP 12: SAVE VALIDATION REPORT
# ============================================================

with open(
    "reports/validation_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("CIVIS - SENTIMENT MODEL VALIDATION RESULTS\n\n")

    file.write(f"Selected model: {best_name}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro-F1: {macro_f1:.4f}\n")
    file.write(f"Weighted-F1: {weighted_f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write(report)

    file.write("\n\nCONFUSION MATRIX\n")
    file.write(str(cm))


# ============================================================
# STEP 13: COMPLETION
# ============================================================

print("\n" + "=" * 55)
print("MODEL TRAINING AND EVALUATION COMPLETED")
print("=" * 55)

print("Selected model:", best_name)
print("Saved model:", model_path)
print("Saved comparison: reports/tuning_results.csv")
print("Saved report: reports/validation_results.txt")
print("Saved confusion matrix: reports/confusion_matrix.png")
