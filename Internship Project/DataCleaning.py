
"""
PROJECT: CIVIS
MODULE: DataCleaning.py

PURPOSE:
Clean the TweetEval training dataset by handling
duplicate records and conflicting sentiment annotations.

INPUT:
data/train.csv

OPERATIONS:
1. Identify duplicate text records.
2. Identify texts with conflicting sentiment labels.
3. Separate conflicting groups for audit.
4. Remove conflicting groups from model training data.
5. Remove redundant identical-label duplicates.
6. Preserve one copy of each valid training example.

OUTPUT:
data/cleaned_train.csv
data/duplicate_audit.csv

IMPORTANT:
- Original train.csv is never modified.
- Validation and test datasets are untouched.
- Conflicting annotations are not assigned an
  arbitrary replacement label.
- Cleaning decisions are recorded for reproducibility.

PROJECT STAGE:
Stage 5 - Data Cleaning

NEXT MODULE:
TextPreprocessing.py
"""

import pandas as pd

# 1. Load original training dataset

df = pd.read_csv("data/train.csv")

print("Original dataset shape:", df.shape)

# 2. Identify texts with conflicting labels

label_counts = df.groupby("text")["label"].nunique()

conflicting_texts = label_counts[
    label_counts > 1
].index

# 3. Save conflicting records separately

conflicting_records = df[
    df["text"].isin(conflicting_texts)
].copy()

conflicting_records.to_csv(
    "data/duplicate_audit.csv",
    index=False
)

# 4. Exclude conflicting texts from training

clean_df = df[
    ~df["text"].isin(conflicting_texts)
].copy()

# 5. Remove exact duplicate records,
# keeping the first occurrence

clean_df = clean_df.drop_duplicates(
    subset="text",
    keep="first"
)

# 6. Save cleaned training dataset

clean_df.to_csv(
    "data/cleaned_train.csv",
    index=False
)

# 7. Display results

print("\nCLEANING SUMMARY")
print("=" * 40)

print("Original records:", len(df))
print("Conflicting records:", len(conflicting_records))
print("Cleaned records:", len(clean_df))
print("Removed records:", len(df) - len(clean_df))

print("\nFinal sentiment distribution:")
print(clean_df["sentiment"].value_counts())

print("\nCleaned dataset shape:", clean_df.shape)

print("\nData cleaning completed.")

# 8. Verify cleaned dataset

print("\nCLEANING VERIFICATION")
print("=" * 40)

# Check remaining duplicate texts
print(
    "Remaining duplicates:",
    clean_df["text"].duplicated().sum()
)

# Check missing values
print(
    "\nMissing values:",
    clean_df.isnull().sum().sum()
)

# Check valid sentiment labels
valid_labels = ["negative", "neutral", "positive"]

print(
    "\nInvalid sentiment labels:",
    (~clean_df["sentiment"].isin(valid_labels)).sum()
)

# Check label consistency
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
expected = clean_df["label"].map(label_map)

print(
    "Label mismatches:",
    (clean_df["sentiment"] != expected).sum()
)

# Verify original dataset was not modified
original_df = pd.read_csv("data/train.csv")

print(
    "\nOriginal dataset shape:",
    original_df.shape
)

print(
    "Cleaned dataset shape:",
    clean_df.shape
)


# 9. Final preservation and cleaning verification

print("\nFINAL PRESERVATION VERIFICATION")
print("=" * 45)

# Reload the original dataset from disk
original_check = pd.read_csv("data/train.csv")

# Reconstruct the expected cleaned dataset independently
expected_clean = original_check[
    ~original_check["text"].isin(conflicting_texts)
].copy()

expected_clean = expected_clean.drop_duplicates(
    subset="text",
    keep="first"
)

# Compare actual and expected cleaned datasets
actual_clean = pd.read_csv("data/cleaned_train.csv")

# Reset indexes before comparison
expected_clean = expected_clean.reset_index(drop=True)
actual_clean = actual_clean.reset_index(drop=True)

print(
    "Cleaned dataset matches expected result:",
    actual_clean.equals(expected_clean)
)

# Check conflicting records saved in audit file
saved_audit = pd.read_csv("data/duplicate_audit.csv")

print(
    "Conflicting audit records saved:",
    len(saved_audit) == len(conflicting_records)
)

# Check original source file still exists and is readable
print(
    "Original dataset records:",
    len(original_check)
)

print(
    "Original dataset shape:",
    original_check.shape
)

# Final checks
assert actual_clean.equals(expected_clean)
assert len(saved_audit) == len(conflicting_records)
assert len(original_check) == 45615

print("\nALL FINAL CHECKS PASSED.")
