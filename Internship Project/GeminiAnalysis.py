
"""
PROJECT: CIVIS
MODULE: GeminiAnalysis.py

PURPOSE:
AI-assisted interpretation of sentiment classification results.

ANALYSIS:
1. Possible sarcasm and irony
2. Ambiguous sentiment
3. Context dependence
4. Recurring themes
5. Communication research insights

INPUT:
reports/validation_predictions.csv
or
reports/misclassified_tweets.csv

OUTPUT:
reports/gemini_insights.csv
reports/gemini_review.csv
reports/gemini_theme_summary.csv

IMPORTANT:
- Gemini does not replace the sentiment classifier.
- Gemini interpretations are not ground truth.
- Human review is required before accepting insights.
- Only a sample of validation tweets is analysed.
"""

import os
import json
import time
import random

import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

SAMPLE_SIZE = 120
BATCH_SIZE = 10
RANDOM_STATE = 42

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

if not API_KEY or API_KEY == "PASTE_YOUR_ACTUAL_GEMINI_API_KEY_HERE":
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. LOAD VALIDATION PREDICTIONS
# ============================================================

def load_predictions():

    prediction_file = Path(
        "reports/validation_predictions.csv"
    )

    error_file = Path(
        "reports/misclassified_tweets.csv"
    )

    if prediction_file.exists():
        df = pd.read_csv(prediction_file)
        print("Using validation predictions.")

    elif error_file.exists():
        df = pd.read_csv(error_file)
        print("Using misclassified tweets only.")

    else:
        raise FileNotFoundError(
            "No prediction file found. Run ErrorAnalysis.py first."
        )

    # Normalise column names
    column_map = {}

    for column in df.columns:
        lower = column.lower().strip()

        if lower in ["text", "tweet"]:
            column_map[column] = "text"

        elif lower in ["sentiment", "actual_sentiment"]:
            column_map[column] = "actual_sentiment"

        elif lower in [
            "predicted_sentiment",
            "predicted sentiment"
        ]:
            column_map[column] = "predicted_sentiment"

    df = df.rename(columns=column_map)

    required = [
        "text",
        "actual_sentiment",
        "predicted_sentiment"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(subset=["text"]).copy()

    df["text"] = df["text"].astype(str)

    df["prediction_correct"] = (
        df["actual_sentiment"].astype(str).str.lower()
        ==
        df["predicted_sentiment"].astype(str).str.lower()
    )

    df = df[df["text"].str.strip() != ""]

    return df


# ============================================================
# 3. SELECT A BALANCED SAMPLE
# ============================================================

def select_sample(df):

    correct = df[df["prediction_correct"]]
    incorrect = df[~df["prediction_correct"]]

    half = SAMPLE_SIZE // 2

    correct_sample = correct.sample(
        n=min(half, len(correct)),
        random_state=RANDOM_STATE
    )

    incorrect_sample = incorrect.sample(
        n=min(half, len(incorrect)),
        random_state=RANDOM_STATE
    )

    sample = pd.concat(
        [correct_sample, incorrect_sample]
    )

    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    sample["analysis_id"] = range(1, len(sample) + 1)

    print("\nSAMPLE SELECTED")
    print("Total tweets:", len(sample))
    print("Correct predictions:", sample["prediction_correct"].sum())
    print("Incorrect predictions:", (~sample["prediction_correct"]).sum())

    return sample


# ============================================================
# 4. GEMINI ANALYSIS INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are an assistant supporting a social-media communication
research project.

Analyse the supplied tweets carefully.

For every tweet:

1. Identify whether there are possible sarcasm or irony signals.
2. Identify whether the sentiment is ambiguous.
3. Identify whether important context is missing.
4. Identify the main subject or theme.
5. Suggest a cautious communication research insight.

RULES:

- Do not assume that every negative statement is sarcastic.
- Do not assume that every positive statement is sincere.
- Do not invent context that is absent from the tweet.
- Do not infer psychological diagnoses, hidden motives or mental states.
- Distinguish evidence from interpretation.
- If uncertain, say so.
- Do not change the actual or predicted sentiment labels.
- Treat classifier predictions as information, not ground truth.
- A communication insight must be a research suggestion,
  not an unsupported claim about public opinion.
- Use short, specific explanations.
- Mark sarcasm as possible only when there is textual evidence.

Return ONLY a valid JSON array.

Each object must contain:

analysis_id
possible_sarcasm (yes/no/uncertain)
sarcasm_evidence
ambiguous_sentiment (yes/no)
context_dependence (low/medium/high)
theme
possible_interpretation
communication_insight
confidence (low/medium/high)
human_review_required (yes/no)
"""


# ============================================================
# 5. ANALYSE ONE BATCH
# ============================================================

def analyse_batch(batch):

    records = []

    for _, row in batch.iterrows():

        records.append({
            "analysis_id": int(row["analysis_id"]),
            "tweet": row["text"],
            "actual_sentiment": str(row["actual_sentiment"]),
            "classifier_prediction": str(
                row["predicted_sentiment"]
            ),
            "classifier_correct": bool(
                row["prediction_correct"]
            )
        })

    prompt = (
        SYSTEM_INSTRUCTIONS
        + "\n\nAnalyse these tweets:\n"
        + json.dumps(records, ensure_ascii=False)
    )

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            results = json.loads(response.text)

            if not isinstance(results, list):
                raise ValueError(
                    "Gemini response was not a JSON list."
                )

            return results

        except Exception as error:

            print(
                f"Gemini request failed "
                f"(attempt {attempt + 1}/3): {error}"
            )

            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    return []


# ============================================================
# 6. RUN ANALYSIS
# ============================================================

def main():

    print("=" * 55)
    print("CIVIS - GEMINI AI-ASSISTED ANALYSIS")
    print("=" * 55)

    df = load_predictions()

    sample = select_sample(df)

    all_results = []

    total_batches = (
        len(sample) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(sample), BATCH_SIZE),
        start=1
    ):

        batch = sample.iloc[
            start:start + BATCH_SIZE
        ]

        print(
            f"\nAnalysing batch {batch_number}/{total_batches}..."
        )

        results = analyse_batch(batch)

        if results:
            all_results.extend(results)

        else:
            print("This batch could not be analysed.")

    if not all_results:
        raise RuntimeError(
            "No Gemini results were returned. "
            "Check API availability and model access."
        )

    # ========================================================
    # 7. VALIDATE AND MERGE RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    if "analysis_id" not in results_df.columns:
        raise ValueError(
            "Gemini did not return analysis_id values."
        )

    results_df["analysis_id"] = pd.to_numeric(
        results_df["analysis_id"],
        errors="coerce"
    )

    results_df = results_df.dropna(
        subset=["analysis_id"]
    )

    results_df["analysis_id"] = (
        results_df["analysis_id"].astype(int)
    )

    # Keep only expected IDs and one result per tweet
    results_df = results_df[
        results_df["analysis_id"].isin(sample["analysis_id"])
    ].drop_duplicates(
        subset=["analysis_id"]
    )

    final_df = sample.merge(
        results_df,
        on="analysis_id",
        how="left"
    )

    final_df["human_review_required"] = (
        final_df["human_review_required"]
        .fillna("yes")
    )

    # ========================================================
    # 8. EXPORT FULL ANALYSIS
    # ========================================================

    final_df.to_csv(
        REPORT_DIR / "gemini_insights.csv",
        index=False
    )

    # ========================================================
    # 9. EXPORT HUMAN REVIEW QUEUE
    # ========================================================

    review_df = final_df[
        final_df["human_review_required"]
        .astype(str).str.lower().isin(["yes", "true"])
    ].copy()

    review_df["human_decision"] = ""
    review_df["reviewer_notes"] = ""

    review_columns = [
        "analysis_id",
        "text",
        "actual_sentiment",
        "predicted_sentiment",
        "possible_sarcasm",
        "sarcasm_evidence",
        "ambiguous_sentiment",
        "possible_interpretation",
        "confidence",
        "human_decision",
        "reviewer_notes"
    ]

    review_columns = [
        column for column in review_columns
        if column in review_df.columns
    ]

    review_df[review_columns].to_csv(
        REPORT_DIR / "gemini_review.csv",
        index=False
    )

    # ========================================================
    # 10. CREATE THEME SUMMARY
    # ========================================================

    valid_themes = final_df.dropna(
        subset=["theme"]
    )

    theme_summary = (
        valid_themes.groupby("theme")
        .agg(
            tweet_count=("analysis_id", "count"),
            sarcasm_flags=("possible_sarcasm",
                           lambda x: (x == "yes").sum()),
            ambiguous_tweets=("ambiguous_sentiment",
                              lambda x: (x == "yes").sum())
        )
        .reset_index()
        .sort_values(
            "tweet_count",
            ascending=False
        )
    )

    theme_summary.to_csv(
        REPORT_DIR / "gemini_theme_summary.csv",
        index=False
    )

    # ========================================================
    # 11. DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 55)
    print("GEMINI ANALYSIS COMPLETED")
    print("=" * 55)

    print("Tweets selected:", len(sample))
    print("Tweets analysed:", len(final_df))
    print(
        "Possible sarcasm flags:",
        (final_df["possible_sarcasm"] == "yes").sum()
    )
    print(
        "Ambiguous sentiment flags:",
        (final_df["ambiguous_sentiment"] == "yes").sum()
    )

    print("\nTOP THEMES")
    print(theme_summary.head(10).to_string(index=False))

    print("\nFILES GENERATED:")
    print("reports/gemini_insights.csv")
    print("reports/gemini_review.csv")
    print("reports/gemini_theme_summary.csv")


if __name__ == "__main__":
    main()
