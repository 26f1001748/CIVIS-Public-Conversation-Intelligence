
"""
PROJECT: CIVIS
MODULE: TextPreprocessing.py

PURPOSE:
Prepare social-media text for sentiment classification
while preserving original tweet content.

INPUT:
data/cleaned_train.csv
data/validation.csv
data/test.csv

OPERATIONS:
1. Decode HTML entities.
2. Normalise URLs.
3. Normalise usernames.
4. Preserve hashtag words while removing #.
5. Convert text to lowercase.
6. Preserve negations, emojis and punctuation.

OUTPUT:
data/processed_train.csv
data/processed_validation.csv
data/processed_test.csv

IMPORTANT:
- Original text is preserved in the text column.
- Transformed text is stored in processed_text.
- No stop-word removal or stemming is performed.
- Identical preprocessing is applied to all splits.

PROJECT STAGE:
Stage 6 - Text Preprocessing
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import re
import html

from pathlib import Path


# ============================================================
# STEP 2: DEFINE THE PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    # Handle missing values safely
    if pd.isna(text):
        return ""

    # Ensure input is a string
    text = str(text)

    # Decode HTML entities such as &amp; and &#39;
    text = html.unescape(text)

    # Replace HTTP, HTTPS and www URLs with a standard token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text,
        flags=re.IGNORECASE
    )

    # Replace usernames with a standard token
    text = re.sub(
        r"@\w+",
        " USER ",
        text
    )

    # Preserve hashtag words while removing only #
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Convert text to lowercase
    text = text.lower()

    # Remove unnecessary surrounding whitespace
    text = text.strip()

    return text


# ============================================================
# STEP 3: DEFINE DATASET PATHS
# ============================================================

data_dir = Path("data")

datasets = {
    "train": data_dir / "cleaned_train.csv",
    "validation": data_dir / "validation.csv",
    "test": data_dir / "test.csv"
}


# ============================================================
# STEP 4: LOAD AND PREPROCESS ALL DATASETS
# ============================================================

for name, input_path in datasets.items():

    print("\n" + "=" * 50)
    print(f"PROCESSING {name.upper()} DATASET")
    print("=" * 50)

    # Load original dataset
    df = pd.read_csv(input_path)

    # Check that the required text column exists
    if "text" not in df.columns:
        raise ValueError(
            f"'text' column not found in {input_path}"
        )

    # Preserve original text and create processed text
    df["processed_text"] = df["text"].apply(
        preprocess_text
    )

    # Generate output filename
    output_path = data_dir / f"processed_{name}.csv"

    # Save processed dataset
    df.to_csv(
        output_path,
        index=False
    )

    # Display dataset information
    print("Dataset shape:", df.shape)

    print("\nOriginal example:")
    print(df["text"].iloc[0])

    print("\nProcessed example:")
    print(df["processed_text"].iloc[0])

    # Count normalised URL and username tokens
    url_count = df["processed_text"].str.count(
        r"\burl\b"
    ).sum()

    user_count = df["processed_text"].str.count(
        r"\buser\b"
    ).sum()

    print("\nNormalised URLs:", int(url_count))
    print("Normalised usernames:", int(user_count))

    print("\nMissing processed texts:",
          df["processed_text"].isna().sum())

    print("Saved to:", output_path)


# ============================================================
# STEP 5: VERIFY PREPROCESSING FUNCTION
# ============================================================

print("\n" + "=" * 50)
print("PREPROCESSING FUNCTION TEST")
print("=" * 50)

test_examples = [
    "I LOVE this! https://example.com",
    "Hello @john_doe, check www.example.com",
    "This is #Amazing and #DataScience",
    "I don't like this at all.",
    "Great news! 😊"
]

for example in test_examples:

    print("\nOriginal:", example)

    print("Processed:", preprocess_text(example))


# ============================================================
# STEP 6: COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 50)
print("TEXT PREPROCESSING COMPLETED")
print("=" * 50)

print("Generated files:")

print("1. data/processed_train.csv")
print("2. data/processed_validation.csv")
print("3. data/processed_test.csv")

print("\nAll three datasets processed using the same function.")

# ============================================================
# END OF TEXT PREPROCESSING
# ============================================================
