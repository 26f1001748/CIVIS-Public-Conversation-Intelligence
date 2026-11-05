
"""
PROJECT: CIVIS
MODULE: FeatureEngineering.py

PURPOSE:
Convert processed tweets into numerical TF-IDF features
for machine-learning sentiment classification.

INPUT:
data/processed_train.csv
data/processed_validation.csv
data/processed_test.csv

OPERATIONS:
1. Load all three processed datasets.
2. Initialise a TF-IDF vectoriser.
3. Fit the vectoriser using training text only.
4. Transform training, validation and test text.
5. Save the sparse feature matrices.
6. Save the fitted vectoriser.

OUTPUT:
data/X_train.npz
data/X_validation.npz
data/X_test.npz
models/tfidf_vectorizer.joblib

IMPORTANT:
The vectoriser is fitted only on training data.
Validation and test data are transformed using
the already-fitted training vectoriser.

PROJECT STAGE:
Stage 7 - Feature Engineering
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

import joblib


# ============================================================
# STEP 2: DEFINE FILE PATHS
# ============================================================

data_dir = Path("data")
models_dir = Path("models")

train_path = data_dir / "processed_train.csv"
validation_path = data_dir / "processed_validation.csv"
test_path = data_dir / "processed_test.csv"


# ============================================================
# STEP 3: CHECK INPUT FILES
# ============================================================

for path in [train_path, validation_path, test_path]:

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


# ============================================================
# STEP 4: LOAD PROCESSED DATASETS
# ============================================================

train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)


# ============================================================
# STEP 5: VERIFY REQUIRED COLUMNS
# ============================================================

for name, df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Test", test_df)
]:

    if "processed_text" not in df.columns:
        raise ValueError(
            f"'processed_text' column missing in {name} dataset."
        )

    if df["processed_text"].isna().any():
        raise ValueError(
            f"Missing processed text found in {name} dataset."
        )


# ============================================================
# STEP 6: INITIALISE TF-IDF VECTORIZER
# ============================================================

vectorizer = TfidfVectorizer(

    max_features=50000,

    ngram_range=(1, 2),

    min_df=2,

    sublinear_tf=True,

    token_pattern=r"(?u)\b\w+\b|[^\w\s]"
)


# ============================================================
# STEP 7: FIT USING TRAINING DATA ONLY
# ============================================================

print("Fitting TF-IDF vectoriser on training data...")

X_train = vectorizer.fit_transform(
    train_df["processed_text"]
)


# ============================================================
# STEP 8: TRANSFORM VALIDATION AND TEST DATA
# ============================================================

print("Transforming validation data...")

X_validation = vectorizer.transform(
    validation_df["processed_text"]
)

print("Transforming test data...")

X_test = vectorizer.transform(
    test_df["processed_text"]
)


# ============================================================
# STEP 9: CREATE OUTPUT DIRECTORIES
# ============================================================

data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)


# ============================================================
# STEP 10: SAVE SPARSE FEATURE MATRICES
# ============================================================

save_npz(
    data_dir / "X_train.npz",
    X_train
)

save_npz(
    data_dir / "X_validation.npz",
    X_validation
)

save_npz(
    data_dir / "X_test.npz",
    X_test
)


# ============================================================
# STEP 11: SAVE FITTED VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    models_dir / "tfidf_vectorizer.joblib"
)


# ============================================================
# STEP 12: DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 50)
print("TF-IDF FEATURE ENGINEERING COMPLETED")
print("=" * 50)

print("\nDataset sizes:")

print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))

print("\nFeature matrix dimensions:")

print("Training matrix:", X_train.shape)
print("Validation matrix:", X_validation.shape)
print("Test matrix:", X_test.shape)

print("\nVocabulary size:", len(vectorizer.vocabulary_))

print("\nSaved files:")
print("- data/X_train.npz")
print("- data/X_validation.npz")
print("- data/X_test.npz")
print("- models/tfidf_vectorizer.joblib")

print("\nVectoriser fitted only on training data.")
print("Validation and test data were transformed, not fitted.")
