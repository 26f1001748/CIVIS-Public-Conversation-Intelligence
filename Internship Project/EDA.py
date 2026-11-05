
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the dataset
df = pd.read_csv("sentiment_train.csv")

# 1. Basic dataset overview
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate texts:", df["text"].duplicated().sum())

# 2. Sentiment distribution
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSentiment proportions:")
print(df["sentiment"].value_counts(normalize=True).round(3))

# 3. Text length analysis
df["text_length"] = df["text"].astype(str).str.split().str.len()

print("\nText length statistics:")
print(df["text_length"].describe())

# 4. Plot sentiment distribution
df["sentiment"].value_counts().reindex(
    ["negative", "neutral", "positive"]
).plot(kind="bar")

plt.title("TweetEval Training Data: Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of posts")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=150)
plt.show()

# 5. Plot text length distribution
df["text_length"].hist(bins=30)
plt.title("Distribution of Post Lengths")
plt.xlabel("Number of words")
plt.ylabel("Number of posts")
plt.tight_layout()
plt.savefig("text_length_distribution.png", dpi=150)
plt.show()

print("\nEDA complete. Charts saved.")
