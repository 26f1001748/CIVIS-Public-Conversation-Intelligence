
"""
PROJECT: CIVIS
MODULE: ErrorAnalysis.py

PURPOSE:
Inspect incorrect predictions made by the selected
LinearSVC sentiment classifier.

INPUT:
data/X_validation.npz
data/validation.csv
models/sentiment_model.joblib

OUTPUT:
reports/misclassified_tweets.csv
reports/validation_predictions.csv

OPERATIONS:
1. Load the selected model.
2. Load validation features and labels.
3. Generate predictions.
4. Identify incorrect predictions.
5. Save misclassified tweets.
6. Save all validation predictions.

IMPORTANT:
- Uses validation data for error analysis.
- Does not train or modify the model.
- Does not use the test dataset.
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import joblib

from scipy.sparse import load_npz
from pathlib import Path

from sklearn.metrics import accuracy_score


# ============================================================
# STEP 2: DEFINE FILE PATHS
# ============================================================

model_path = Path("models/sentiment_model.joblib")
features_path = Path("data/X_validation.npz")
validation_path = Path("data/validation.csv")

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)


# ============================================================
# STEP 3: CHECK REQUIRED FILES
# ============================================================

for path in [model_path, features_path, validation_path]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# STEP 4: LOAD MODEL AND VALIDATION DATA
# ============================================================

print("Loading selected sentiment model...")

model = joblib.load(model_path)

print("Loading validation features...")

X_validation = load_npz(features_path)

print("Loading validation dataset...")

df = pd.read_csv(validation_path)


# ============================================================
# STEP 5: VERIFY DATA ALIGNMENT
# ============================================================

if X_validation.shape[0] != len(df):

    raise ValueError(
        "Validation feature count does not match dataset row count."
    )

if "text" not in df.columns or "sentiment" not in df.columns:

    raise ValueError(
        "Validation dataset must contain text and sentiment columns."
    )


# ============================================================
# STEP 6: GENERATE PREDICTIONS
# ============================================================

print("\nGenerating sentiment predictions...")

predictions = model.predict(X_validation)

df["predicted_sentiment"] = predictions


# ============================================================
# STEP 7: IDENTIFY INCORRECT PREDICTIONS
# ============================================================

errors = df[
    df["sentiment"] != df["predicted_sentiment"]
].copy()

correct = len(df) - len(errors)

accuracy = accuracy_score(
    df["sentiment"],
    df["predicted_sentiment"]
)


# ============================================================
# STEP 8: SAVE ALL VALIDATION PREDICTIONS
# ============================================================

df.to_csv(
    reports_dir / "validation_predictions.csv",
    index=False
)


# ============================================================
# STEP 9: SAVE MISCLASSIFIED TWEETS
# ============================================================

errors[
    ["text", "sentiment", "predicted_sentiment"]
].to_csv(
    reports_dir / "misclassified_tweets.csv",
    index=False
)


# ============================================================
# STEP 10: DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 50)
print("VALIDATION ERROR ANALYSIS")
print("=" * 50)

print("Total validation records:", len(df))
print("Correct predictions:", correct)
print("Incorrect predictions:", len(errors))
print("Validation accuracy:", round(accuracy, 4))

print("\nSAMPLE MISCLASSIFICATIONS")
print("=" * 50)

if len(errors) > 0:

    print(
        errors[
            ["text", "sentiment", "predicted_sentiment"]
        ].head(15).to_string(index=False)
    )

else:

    print("No misclassifications found.")


# ============================================================
# STEP 11: COMPLETION
# ============================================================

print("\nError analysis completed.")

print("Saved files:")
print("- reports/misclassified_tweets.csv")
print("- reports/validation_predictions.csv")
