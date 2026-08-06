import argparse
import json
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from clean_data import clean_dataset


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METRICS_PATH = Path("model_metrics.pkl")
BEST_MODEL_PATH = Path("best_sentiment_model.pkl")
BEST_JOBLIB_PATH = Path("best_sentiment_model.joblib")
BEST_INFO_PATH = Path("best_model_info.json")


def prepare_folders():
    return None


def load_or_clean_data(data_path, cleaned_path):
    if not Path(cleaned_path).exists():
        clean_dataset(
            input_path=data_path,
            output_path=cleaned_path
        )

    df = pd.read_csv(cleaned_path)

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    return df


def evaluate_model(name, model, x_test, y_test):
    y_pred = model.predict(x_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(
        f"\n{name} Classification Report:"
    )
    print(report)

    return metrics, y_pred


def train_classical_models(x_train, x_test, y_train, y_test):
    models = {
        "bow_naive_bayes": Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("model", MultinomialNB()),
            ]
        ),
        "tfidf_naive_bayes": Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("model", MultinomialNB()),
            ]
        ),
        "tfidf_logistic_regression": Pipeline(
            [
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("model", LogisticRegression(max_iter=1000, C=1.0)),
            ]
        ),
        "tfidf_logistic_regression_tuned": Pipeline(
            [
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("model", LogisticRegression(max_iter=1000, C=2.0)),
            ]
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics, _ = evaluate_model(name, model, x_test, y_test)
        results.append(metrics)
        trained_models[name] = {
            "feature_type": "text_pipeline",
            "model": model,
        }

    return results, trained_models


def train_embedding_model(x_train, x_test, y_train, y_test):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    x_train_emb = embedding_model.encode(
        x_train.tolist(),
        show_progress_bar=True
    )
    x_test_emb = embedding_model.encode(
        x_test.tolist(),
        show_progress_bar=True
    )

    model = LogisticRegression(max_iter=1000, C=2.0)
    model.fit(x_train_emb, y_train)

    metrics, _ = evaluate_model(
        "embedding_logistic_regression",
        model,
        x_test_emb,
        y_test
    )

    trained_model = {
        "feature_type": "embedding",
        "model": model,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
    }

    return metrics, trained_model


def show_accuracy_chart(metrics_df):
    metrics_df = metrics_df.sort_values("accuracy", ascending=False)

    plt.figure(figsize=(11, 6))
    bars = plt.bar(metrics_df["model"], metrics_df["accuracy"], color="#2E86AB")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Sentiment Model Accuracy Comparison")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()


def save_best_model(metrics_df, trained_models):
    best_row = metrics_df.sort_values("f1_score", ascending=False).iloc[0]
    best_name = best_row["model"]

    payload = trained_models[best_name]
    payload["model_name"] = best_name
    payload["metrics"] = best_row.to_dict()

    joblib.dump(payload, BEST_JOBLIB_PATH)

    with BEST_MODEL_PATH.open("wb") as file:
        pickle.dump(
            payload,
            file
        )

    BEST_INFO_PATH.write_text(
        json.dumps(best_row.to_dict(), indent=2),
        encoding="utf-8"
    )

    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {BEST_JOBLIB_PATH}")
    print(f"Saved PKL model to: {BEST_MODEL_PATH}")


def save_all_models_as_pkl(metrics_df, trained_models):
    metrics_by_model = {
        row["model"]: row
        for row in metrics_df.to_dict(orient="records")
    }

    for model_name, payload in trained_models.items():
        model_payload = dict(payload)
        model_payload["model_name"] = model_name
        model_payload["metrics"] = metrics_by_model.get(model_name, {})

        output_path = Path(f"{model_name}.pkl")

        with output_path.open("wb") as file:
            pickle.dump(
                model_payload,
                file
            )

    print("Saved individual PKL models to the project folder.")


def save_metrics(metrics_df):
    with METRICS_PATH.open("wb") as file:
        pickle.dump(
            metrics_df,
            file
        )

    print(f"Saved model metrics to: {METRICS_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare sentiment analysis models."
    )
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--cleaned", default="cleaned_data.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--show-graph", action="store_true")
    args = parser.parse_args()

    prepare_folders()

    df = load_or_clean_data(
        args.data,
        args.cleaned
    )

    x = df["review"]
    y = df["sentiment"]

    stratify = y if y.value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify
    )

    all_metrics, trained_models = train_classical_models(
        x_train,
        x_test,
        y_train,
        y_test
    )

    embedding_metrics, embedding_model = train_embedding_model(
        x_train,
        x_test,
        y_train,
        y_test
    )

    all_metrics.append(embedding_metrics)
    trained_models["embedding_logistic_regression"] = embedding_model

    metrics_df = pd.DataFrame(all_metrics)

    save_metrics(metrics_df)
    save_all_models_as_pkl(metrics_df, trained_models)
    save_best_model(metrics_df, trained_models)

    if args.show_graph:
        show_accuracy_chart(metrics_df)

    print("\nModel metrics:")
    print(metrics_df.sort_values("f1_score", ascending=False))
    print("\nOpen the saved accuracy graph later with:")
    print("python3 show_results.py")


if __name__ == "__main__":
    main()
