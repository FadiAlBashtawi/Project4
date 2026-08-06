from db_operations import get_all_predictions


def format_score(score):
    if score is None:
        return "N/A"

    return f"{score:.4f}"


def show_prediction_history():
    rows = get_all_predictions()

    if not rows:
        print("No predictions saved yet.")
        return

    print("\nPrediction History")
    print("-" * 80)

    for row in rows:
        print(f"ID: {row.id}")
        print(f"Text: {row.input_text}")
        print(f"Sentiment: {row.predicted_sentiment}")
        print(f"Score: {format_score(row.prediction_score)}")
        print(f"Created at: {row.created_at}")
        print("-" * 80)


if __name__ == "__main__":
    show_prediction_history()
