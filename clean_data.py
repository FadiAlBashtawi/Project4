import argparse
import re
import string

import pandas as pd


VALID_SENTIMENTS = {
    "positive": "Positive",
    "pos": "Positive",
    "1": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "0": "Negative",
    "neutral": "Neutral",
    "neu": "Neutral",
    "2": "Neutral",
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_sentiment(value):
    value = str(value).strip().lower()
    return VALID_SENTIMENTS.get(value)


def clean_dataset(input_path="data.csv", output_path="cleaned_data.csv"):
    df = pd.read_csv(input_path)

    required_columns = {"review", "sentiment"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "data.csv must contain these columns: review, sentiment. "
            f"Missing: {', '.join(sorted(missing_columns))}"
        )

    df = df[["review", "sentiment"]].copy()
    df = df.dropna(subset=["review", "sentiment"])

    df["review"] = df["review"].apply(clean_text)
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment)

    df = df.dropna(subset=["sentiment"])
    df = df[df["review"].str.len() > 0]
    df = df.drop_duplicates(subset=["review", "sentiment"])
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)

    print(f"Cleaned rows: {len(df)}")
    print(f"Saved cleaned dataset to: {output_path}")
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Clean data.csv for the sentiment analyzer project."
    )
    parser.add_argument("--input", default="data.csv")
    parser.add_argument("--output", default="cleaned_data.csv")
    args = parser.parse_args()

    clean_dataset(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
