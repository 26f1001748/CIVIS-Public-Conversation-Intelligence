
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load all three dataset splits

data_dir = Path("data")

train_df = pd.read_csv(data_dir / "train.csv")
validation_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

datasets = {
    "Train": train_df,
    "Validation": validation_df,
    "Test": test_df
}

# 2. Inspect each dataset

for name, df in datasets.items():

    print(f"\n{'=' * 40}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 40}")

    print("\nDataset shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate texts:")
    print(df["text"].duplicated().sum())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nSentiment proportions:")
    print(
        df["sentiment"]
        .value_counts(normalize=True)
        .round(3)
    )

    # Calculate tweet length
    text_length = df["text"].dropna().astype(str).str.split().str.len()

    print("\nTweet length statistics:")
    print(text_length.describe())

# 3. Compare sentiment distributions

distribution = pd.DataFrame({
    name: df["sentiment"].value_counts(normalize=True)
    for name, df in datasets.items()
})

distribution = distribution.reindex(
    ["negative", "neutral", "positive"]
)

print("\nSENTIMENT DISTRIBUTION COMPARISON")
print(distribution.round(3))

# 4. Visualise sentiment distribution across splits

distribution.plot(kind="bar", figsize=(9, 5))

plt.title("Sentiment Distribution Across Dataset Splits")
plt.xlabel("Sentiment")
plt.ylabel("Proportion of Posts")
plt.xticks(rotation=0)
plt.legend(title="Dataset split")
plt.tight_layout()

plt.savefig("sentiment_split_comparison.png", dpi=150)
plt.show()

print("\nComplete EDA finished.")
