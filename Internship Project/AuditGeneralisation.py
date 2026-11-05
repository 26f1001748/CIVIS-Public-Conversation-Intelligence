
# ============================================================
# CIVIS - GENERALISATION AND DATASET AUDIT
#
# Checks:
# 1. Dataset sizes and sentiment distributions
# 2. Preprocessed text availability
# 3. Vocabulary coverage and unseen words
# 4. Average tweet lengths
# 5. Validation versus test model performance
#
# This script does NOT retrain or modify the model.
# ============================================================

import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report


# ------------------------------------------------------------
# STEP 1: FILE PATHS
# ------------------------------------------------------------

FILES = {
    "Training": "data/processed_train.csv",
    "Validation": "data/processed_validation.csv",
    "Test": "data/processed_test.csv"
}

VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

VALIDATION_PREDICTIONS = "tableau_data/civis_predictions.csv"

TEST_PREDICTIONS = "reports/test_evaluation/test_predictions.csv"

OUTPUT_FOLDER = "reports/generalisation_audit"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# STEP 2: LOAD DATA AND VECTORIZER
# ------------------------------------------------------------

print("\nLoading datasets...")

datasets = {}

for name, path in FILES.items():

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    datasets[name] = pd.read_csv(path)

    print(f"{name}: {len(datasets[name])} tweets")

print("\nLoading fitted TF-IDF vectorizer...")

vectorizer = joblib.load(VECTORIZER_PATH)

# Vocabulary learned from training data only
training_vocabulary = set(vectorizer.vocabulary_.keys())

print("Training vocabulary size:", len(training_vocabulary))


# ------------------------------------------------------------
# STEP 3: IDENTIFY DATA COLUMNS
# ------------------------------------------------------------

def find_column(df, possible_names):

    lookup = {
        c.strip().lower().replace(" ", "_"): c
        for c in df.columns
    }

    for name in possible_names:
        if name in lookup:
            return lookup[name]

    return None


# ------------------------------------------------------------
# STEP 4: STANDARDISE SENTIMENT LABELS
# ------------------------------------------------------------

def standardise_sentiment(value):

    mapping = {
        "0": "Negative",
        "1": "Neutral",
        "2": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive"
    }

    return mapping.get(str(value).strip().lower(), "Unknown")


# ------------------------------------------------------------
# STEP 5: AUDIT DATASET DISTRIBUTIONS
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("DATASET DISTRIBUTION AUDIT")
print("=" * 55)

distribution_results = []

for name, df in datasets.items():

    label_column = find_column(
        df, ["sentiment", "label"]
    )

    if label_column is None:
        raise ValueError(f"No sentiment column found in {name}")

    labels = df[label_column].apply(standardise_sentiment)

    print(f"\n{name} sentiment distribution:")

    percentages = labels.value_counts(normalize=True) * 100

    print(percentages.round(2))

    for sentiment in ["Negative", "Neutral", "Positive"]:

        distribution_results.append({
            "Dataset": name,
            "Sentiment": sentiment,
            "Percentage": round(
                percentages.get(sentiment, 0), 2
            )
        })

distribution_df = pd.DataFrame(distribution_results)

distribution_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "sentiment_distributions.csv"),
    index=False
)


# ------------------------------------------------------------
# STEP 6: AUDIT PREPROCESSED TEXT
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("PREPROCESSING AUDIT")
print("=" * 55)

text_column = "processed_text"

preprocessing_results = []

for name, df in datasets.items():

    if text_column not in df.columns:
        raise ValueError(
            f"{name} is missing the processed_text column. "
            "Check your preprocessing output."
        )

    text = df[text_column].fillna("").astype(str)

    empty_count = (text.str.strip() == "").sum()

    average_length = text.str.len().mean()

    url_tokens = text.str.count(r"\bURL\b").sum()

    user_tokens = text.str.count(r"\bUSER\b").sum()

    print(f"\n{name}:")
    print("Empty processed texts:", empty_count)
    print("Average character length:", round(average_length, 2))
    print("Normalised URL tokens:", url_tokens)
    print("Normalised USER tokens:", user_tokens)

    preprocessing_results.append({
        "Dataset": name,
        "Total Tweets": len(df),
        "Empty Texts": int(empty_count),
        "Average Character Length": round(average_length, 2),
        "URL Tokens": int(url_tokens),
        "USER Tokens": int(user_tokens)
    })

pd.DataFrame(preprocessing_results).to_csv(
    os.path.join(OUTPUT_FOLDER, "preprocessing_audit.csv"),
    index=False
)


# ------------------------------------------------------------
# STEP 7: AUDIT VOCABULARY COVERAGE
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("VOCABULARY COVERAGE AUDIT")
print("=" * 55)

vocabulary_results = []

for name, df in datasets.items():

    texts = df[text_column].fillna("").astype(str)

    total_tokens = 0
    known_tokens = 0
    unseen_tokens = 0

    unique_tokens = set()

    for text in texts:

        # Extract tokens using a pattern compatible with
        # the default TF-IDF word tokenisation.
        tokens = re.findall(r"(?u)\b\w\w+\b", text)

        total_tokens += len(tokens)

        for token in tokens:

            if token in training_vocabulary:
                known_tokens += 1
            else:
                unseen_tokens += 1

            unique_tokens.add(token)

    coverage = (
        known_tokens / total_tokens * 100
        if total_tokens else 0
    )

    unseen_rate = (
        unseen_tokens / total_tokens * 100
        if total_tokens else 0
    )

    unseen_unique = unique_tokens - training_vocabulary

    print(f"\n{name}:")
    print("Total tokens:", total_tokens)
    print("Known tokens:", known_tokens)
    print("Unseen tokens:", unseen_tokens)
    print("Vocabulary coverage:", round(coverage, 2), "%")
    print("Unique unseen words:", len(unseen_unique))

    vocabulary_results.append({
        "Dataset": name,
        "Total Tokens": total_tokens,
        "Known Tokens": known_tokens,
        "Unseen Tokens": unseen_tokens,
        "Vocabulary Coverage (%)": round(coverage, 2),
        "Unseen Token Rate (%)": round(unseen_rate, 2),
        "Unique Unseen Words": len(unseen_unique)
    })

pd.DataFrame(vocabulary_results).to_csv(
    os.path.join(OUTPUT_FOLDER, "vocabulary_audit.csv"),
    index=False
)


# ------------------------------------------------------------
# STEP 8: COMPARE VALIDATION AND TEST PERFORMANCE
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("VALIDATION VS TEST PERFORMANCE")
print("=" * 55)

performance_results = []

for name, path in [
    ("Validation", VALIDATION_PREDICTIONS),
    ("Test", TEST_PREDICTIONS)
]:

    if not os.path.exists(path):
        print(f"{name} predictions not found. Skipping.")
        continue

    df = pd.read_csv(path)

    actual_column = find_column(
        df, ["actual_sentiment"]
    )

    predicted_column = find_column(
        df, ["predicted_sentiment"]
    )

    if actual_column is None or predicted_column is None:
        print(f"Missing prediction columns in {name}.")
        continue

    y_true = df[actual_column].apply(standardise_sentiment)

    y_pred = df[predicted_column].apply(standardise_sentiment)

    report = classification_report(
        y_true,
        y_pred,
        labels=["Negative", "Neutral", "Positive"],
        output_dict=True,
        zero_division=0
    )

    print(f"\n{name}:")

    print("Accuracy:", round(report["accuracy"] * 100, 2), "%")

    print(
        "Macro-F1:",
        round(report["macro avg"]["f1-score"], 4)
    )

    for sentiment in ["Negative", "Neutral", "Positive"]:

        performance_results.append({
            "Dataset": name,
            "Sentiment": sentiment,
            "Precision": report[sentiment]["precision"],
            "Recall": report[sentiment]["recall"],
            "F1 Score": report[sentiment]["f1-score"]
        })

pd.DataFrame(performance_results).to_csv(
    os.path.join(OUTPUT_FOLDER, "performance_comparison.csv"),
    index=False
)


# ------------------------------------------------------------
# STEP 9: FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("GENERALISATION AUDIT COMPLETED")
print("=" * 55)

print("\nReports saved in:", OUTPUT_FOLDER)

print("1. sentiment_distributions.csv")
print("2. preprocessing_audit.csv")
print("3. vocabulary_audit.csv")
print("4. performance_comparison.csv")

print("\nIMPORTANT:")
print("Vocabulary coverage is a diagnostic, not a measure")
print("of sentiment understanding or model accuracy.")

# ============================================================
# END OF AUDIT
# ============================================================
