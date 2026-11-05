
import pandas as pd
from pathlib import Path

# 1. Load all three dataset splits

data_dir = Path("data")

train_df = pd.read_csv(data_dir / "train.csv")
validation_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

datasets = {
    "train": train_df,
    "validation": validation_df,
    "test": test_df
}

# 2. Define valid labels

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# 3. Audit each split

for name, df in datasets.items():

    print("\n" + "=" * 45)
    print(f"AUDIT: {name.upper()}")
    print("=" * 45)

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Empty or whitespace-only tweets
    empty_text = (
        df["text"].fillna("").astype(str).str.strip() == ""
    )

    print("\nEmpty tweets:", empty_text.sum())

    # Duplicate tweets
    print(
        "Duplicate tweets:",
        df["text"].duplicated().sum()
    )

    # Invalid numeric labels
    invalid_labels = ~df["label"].isin(label_map.keys())

    print("Invalid labels:", invalid_labels.sum())

    # Check whether sentiment names match numeric labels
    expected_sentiment = df["label"].map(label_map)

    mismatch = (
        df["sentiment"] != expected_sentiment
    ).fillna(True)

    print("Label mismatches:", mismatch.sum())

# 4. Check duplicates across splits

print("\n" + "=" * 45)
print("CROSS-SPLIT DUPLICATE AUDIT")
print("=" * 45)

train_texts = set(train_df["text"].dropna())
validation_texts = set(validation_df["text"].dropna())
test_texts = set(test_df["text"].dropna())

print(
    "Train-validation overlap:",
    len(train_texts & validation_texts)
)

print(
    "Train-test overlap:",
    len(train_texts & test_texts)
)

print(
    "Validation-test overlap:",
    len(validation_texts & test_texts)
)

print("\nData quality audit completed.")

"""
PROJECT: CIVIS
MODULE: DataAudit.py

PURPOSE:
Perform a data quality audit on the official TweetEval
training, validation and test splits.

INPUT:
data/train.csv
data/validation.csv
data/test.csv

AUDIT CHECKS:
1. Missing values in each column.
2. Empty or whitespace-only tweets.
3. Duplicate tweets within each split.
4. Invalid numerical sentiment labels.
5. Mismatches between numerical labels and
   their corresponding sentiment names.
6. Duplicate text overlap between different splits.

CURRENT AUDIT RESULTS:
Training:
- Missing values: 0
- Empty tweets: 0
- Duplicate texts: 29
- Invalid labels: 0
- Label mismatches: 0

Validation:
- Missing values: 0
- Empty tweets: 0
- Duplicate texts: 0
- Invalid labels: 0
- Label mismatches: 0

Test:
- Missing values: 0
- Empty tweets: 0
- Duplicate texts: 0
- Invalid labels: 0
- Label mismatches: 0

CROSS-SPLIT OVERLAP:
- Train-validation: 0
- Train-test: 0
- Validation-test: 0

IMPORTANT:
- This module identifies data quality issues.
- It does not delete or modify any records.
- The 29 training duplicates must be investigated
  before deciding on a cleaning strategy.
- Original validation and test splits must be preserved.

PROJECT STAGE:
Stage 3 - Data Quality Audit

NEXT MODULE:
DataCleaning.py
"""
