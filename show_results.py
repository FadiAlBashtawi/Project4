import pickle
from pathlib import Path

import matplotlib.pyplot as plt


METRICS_PATH = Path("model_metrics.pkl")


def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "model_metrics.pkl was not found. Run train_models.py once first."
        )

    with METRICS_PATH.open("rb") as file:
        return pickle.load(file)


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


def main():
    metrics_df = load_metrics()

    print("\nSaved model metrics:")
    print(metrics_df.sort_values("accuracy", ascending=False))

    show_accuracy_chart(
        metrics_df
    )


if __name__ == "__main__":
    main()
