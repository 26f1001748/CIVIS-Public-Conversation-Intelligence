
# ============================================================
# CIVIS - FINAL MODEL EVALUATION
#
# Purpose:
# 1. Load the saved, selected LinearSVC model.
# 2. Load the untouched test dataset.
# 3. Generate predictions for all test tweets.
# 4. Calculate final evaluation metrics.
# 5. Generate confusion matrix data.
# 6. Save misclassified tweets for error analysis.
#
# IMPORTANT:
# The test dataset is NOT used for model training or tuning.
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ------------------------------------------------------------
# STEP 2: DEFINE FILE PATHS
# ------------------------------------------------------------

MODEL_PATH = "models/logistic_regression.joblib"

TEST_DATA_PATH = "data/test.csv"

TEST_FEATURES_PATH = "data/X_test.npz"

OUTPUT_FOLDER = "reports/test_evaluation"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# STEP 3: DEFINE SENTIMENT MAPPING
# ------------------------------------------------------------

# Standardise numerical and text labels into one format.

def standardise_sentiment(value):

    if pd.isna(value):
        raise ValueError("Missing sentiment label detected.")

    value = str(value).strip().lower()

    mapping = {
        "0": "Negative",
        "negative": "Negative",
        "1": "Neutral",
        "neutral": "Neutral",
        "2": "Positive",
        "positive": "Positive"
    }

    if value not in mapping:
        raise ValueError(
            f"Unknown sentiment label: {value}"
        )

    return mapping[value]


# Numerical representation for sklearn evaluation
numeric_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

sentiment_names = [
    "Negative",
    "Neutral",
    "Positive"
]


# ------------------------------------------------------------
# STEP 4: LOAD SAVED MODEL AND TEST DATA
# ------------------------------------------------------------

print("Loading selected model...")

model = joblib.load(MODEL_PATH)

print("Loading untouched test dataset...")

test_df = pd.read_csv(TEST_DATA_PATH)

print("Loading test TF-IDF features...")

X_test = load_npz(TEST_FEATURES_PATH)


# ------------------------------------------------------------
# STEP 5: IDENTIFY ACTUAL SENTIMENT COLUMN
# ------------------------------------------------------------

# Detect the label column regardless of capitalisation.

column_lookup = {
    column.strip().lower().replace(" ", "_"): column
    for column in test_df.columns
}

if "sentiment" in column_lookup:
    label_column = column_lookup["sentiment"]

elif "label" in column_lookup:
    label_column = column_lookup["label"]

else:
    raise ValueError(
        "Could not find Sentiment or Label column."
    )

print("Actual label column:", label_column)


# ------------------------------------------------------------
# STEP 6: CHECK DATA DIMENSIONS
# ------------------------------------------------------------

if len(test_df) != X_test.shape[0]:

    raise ValueError(
        "Test CSV and TF-IDF matrix have different row counts."
    )

print("Number of test tweets:", len(test_df))


# ------------------------------------------------------------
# STEP 7: GENERATE FINAL PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions on untouched test data...")

# The model is loaded in its already-trained state.
# No fitting or parameter tuning happens here.

raw_predictions = model.predict(X_test)

# Standardise actual sentiment labels
actual_sentiment = test_df[label_column].apply(
    standardise_sentiment
)

# Standardise model predictions
predicted_sentiment = pd.Series(
    raw_predictions
).apply(standardise_sentiment)

# Convert both to numerical labels for evaluation
y_true = actual_sentiment.map(numeric_mapping).to_numpy()

y_pred = predicted_sentiment.map(numeric_mapping).to_numpy()


# ------------------------------------------------------------
# STEP 8: CALCULATE FINAL EVALUATION METRICS
# ------------------------------------------------------------

print("\nCalculating final evaluation metrics...")

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=sentiment_names,
    output_dict=True,
    zero_division=0
)

metrics_df = pd.DataFrame(report).T

# Save detailed classification report
metrics_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "test_classification_report.csv"
    )
)


# ------------------------------------------------------------
# STEP 9: GENERATE CONFUSION MATRIX
# ------------------------------------------------------------

print("Generating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

confusion_data = []

for i, actual_name in enumerate(sentiment_names):

    for j, predicted_name in enumerate(sentiment_names):

        confusion_data.append({
            "Actual Sentiment": actual_name,
            "Predicted Sentiment": predicted_name,
            "Tweet Count": int(cm[i, j])
        })

confusion_df = pd.DataFrame(confusion_data)

confusion_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "test_confusion_matrix.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# STEP 10: IDENTIFY MISCLASSIFIED TWEETS
# ------------------------------------------------------------

print("Identifying misclassified tweets...")

results_df = test_df.copy()

results_df["Actual Sentiment"] = actual_sentiment.values

results_df["Predicted Sentiment"] = predicted_sentiment.values

results_df["Correct Prediction"] = (
    y_true == y_pred
)

results_df["Prediction Status"] = np.where(
    results_df["Correct Prediction"],
    "Correct",
    "Misclassified"
)

# Keep only incorrect predictions
misclassified_df = results_df[
    results_df["Correct Prediction"] == False
].copy()

misclassified_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "test_misclassified_tweets.csv"
    ),
    index=False
)

# Save predictions for all test tweets
results_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "test_predictions.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# STEP 11: DISPLAY FINAL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("CIVIS - FINAL TEST EVALUATION RESULTS")
print("=" * 55)

print("\nTotal test tweets:", len(test_df))

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print(f"Macro F1-Score: {macro_f1:.4f}")

print(f"Weighted F1-Score: {weighted_f1:.4f}")

print(
    "\nCorrect predictions:",
    int(np.sum(y_true == y_pred))
)

print(
    "Incorrect predictions:",
    int(np.sum(y_true != y_pred))
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=sentiment_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(cm)

print("\nMisclassified tweets:", len(misclassified_df))

print("\nFiles saved in:", OUTPUT_FOLDER)

print("1. test_classification_report.csv")
print("2. test_confusion_matrix.csv")
print("3. test_misclassified_tweets.csv")
print("4. test_predictions.csv")

print("\nFINAL TEST EVALUATION COMPLETED.")

# ============================================================
# END OF SCRIPT
# ============================================================
