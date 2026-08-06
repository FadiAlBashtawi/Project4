import argparse
from pathlib import Path

from sentiment_pipeline import DEFAULT_MODEL_PATH, predict_and_store


def print_prediction(result):
    score = result["prediction_score"]

    if score is None:
        score_text = "N/A"
    else:
        score_text = f"{score:.4f}"

    print(f"Text: {result['input_text']}")
    print(f"Predicted sentiment: {result['predicted_sentiment']}")
    print(f"Prediction score: {score_text}")
    print(f"Saved prediction id: {result['id']}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict sentiment for new text and save it to the database."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Optional text to classify. If omitted, interactive input starts."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH)
    )
    args = parser.parse_args()

    model_path = Path(
        args.model
    )

    if args.text:
        text = " ".join(args.text)
        result = predict_and_store(
            text,
            model_path=model_path
        )
        print_prediction(
            result
        )
        return

    print("Sentiment Analyzer")
    print("Type a review or tweet, then press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("Enter text: ").strip()

        if text.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if not text:
            print("Please enter some text.")
            continue

        result = predict_and_store(
            text,
            model_path=model_path
        )

        print_prediction(
            result
        )
        print()


if __name__ == "__main__":
    main()
