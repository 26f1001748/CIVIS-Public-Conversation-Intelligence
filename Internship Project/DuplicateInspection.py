
"""
PROJECT: CIVIS
MODULE: DuplicateInvestigation.py

PURPOSE:
Investigate duplicate tweets within the training dataset.

INPUT:
data/train.csv

OPERATIONS:
1. Identify repeated tweet texts.
2. Display their original records.
3. Check whether duplicate texts have identical labels.
4. Count duplicates with conflicting sentiment labels.
5. Produce a summary of duplicate records.

OUTPUT:
Terminal-based duplicate investigation report.

IMPORTANT:
This module only investigates duplicates.
It does not remove or modify any records.

PROJECT STAGE:
Stage 4 - Duplicate Investigation

NEXT MODULE:
DataCleaning.py
"""

import pandas as pd

# 1. Load training dataset
df = pd.read_csv("data/train.csv")

# 2. Identify duplicate text records
duplicates = df[
    df["text"].duplicated(keep=False)
].copy()

# 3. Display duplicate records
print("\nDUPLICATE RECORDS")
print("=" * 50)

print(
    duplicates[
        ["text", "label", "sentiment"]
    ].sort_values("text").to_string(index=False)
)

# 4. Check whether duplicate texts have conflicting labels
label_counts = duplicates.groupby("text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1]

print("\nDUPLICATE SUMMARY")
print("=" * 50)

print("Duplicate records:", len(duplicates))
print("Unique duplicated texts:", duplicates["text"].nunique())
print("Texts with conflicting labels:", len(conflicting_texts))

# 5. Display conflicting records, if any
if len(conflicting_texts) > 0:

    print("\nCONFLICTING LABEL RECORDS:")

    print(
        duplicates[
            duplicates["text"].isin(conflicting_texts.index)
        ][["text", "label", "sentiment"]]
        .sort_values("text")
        .to_string(index=False)
    )

else:
    print("\nNo conflicting labels found.")

print("\nDuplicate investigation completed.")
