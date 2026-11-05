
import pandas as pd
import joblib
import re
import html
import codecs

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# 1. Load data
df = pd.read_csv("data/validation.csv")
X = load_npz("data/X_validation.npz")

# 2. Decode escaped text
def clean_text(text):
    text = html.unescape(str(text))
    text = codecs.decode(text, "unicode_escape")
    return text

df["cluster_text"] = df["text"].apply(clean_text)

# 3. Create clustering-specific TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    sublinear_tf=True
)

X_cluster = vectorizer.fit_transform(df["cluster_text"])

# 4. K-Means clustering
print("Running K-Means...")

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=20,
    max_iter=300
)

df["Cluster"] = kmeans.fit_predict(X_cluster) + 1

# 5. Sentiment predictions
model = joblib.load("models/sentiment_model.joblib")
df["Predicted Sentiment"] = model.predict(X)

df["Correct Prediction"] = (
    df["sentiment"] == df["Predicted Sentiment"]
)

# 6. Reduce dimensions for Tableau
print("Creating 2D coordinates...")

svd = TruncatedSVD(
    n_components=2,
    random_state=42,
    n_iter=10
)

coordinates = svd.fit_transform(X_cluster)

df["SVD 1"] = coordinates[:, 0]
df["SVD 2"] = coordinates[:, 1]

# 7. Extract meaningful cluster keywords
terms = vectorizer.get_feature_names_out()
summary = []

for cluster in sorted(df["Cluster"].unique()):

    subset = X_cluster[df["Cluster"].values == cluster]

    average_scores = subset.mean(axis=0).A1

    top_indices = average_scores.argsort()[-10:][::-1]

    top_terms = [terms[i] for i in top_indices]

    cluster_rows = df[df["Cluster"] == cluster]

    summary.append({
        "Cluster": cluster,
        "Tweet Count": len(cluster_rows),
        "Dominant Predicted Sentiment":
            cluster_rows["Predicted Sentiment"].mode().iloc[0],
        "Top Terms": ", ".join(top_terms)
    })

summary_df = pd.DataFrame(summary)

# 8. Export Tableau files
output = Path("tableau_data")
output.mkdir(exist_ok=True)

tableau_df = df[[
    "text",
    "sentiment",
    "Predicted Sentiment",
    "Correct Prediction",
    "Cluster",
    "SVD 1",
    "SVD 2"
]].rename(columns={
    "text": "Tweet",
    "sentiment": "Actual Sentiment"
})

tableau_df.to_csv(
    output / "clustered_tweets.csv",
    index=False
)

summary_df.to_csv(
    output / "cluster_summary.csv",
    index=False
)

# 9. Results
print("\nCLUSTER SUMMARY")
print(summary_df.to_string(index=False))

print("\nSVD explained variance:")
print(svd.explained_variance_ratio_.round(4))
print("Total:", round(svd.explained_variance_ratio_.sum(), 4))

print("\nGenerated:")
print("tableau_data/clustered_tweets.csv")
print("tableau_data/cluster_summary.csv")
