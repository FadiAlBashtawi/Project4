from flask import Flask, render_template_string, request

from db_operations import get_all_predictions
from sentiment_pipeline import predict_and_store


app = Flask(__name__)


PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sentiment Analyzer</title>
    <style>
        :root {
            --bg: #f4f7fb;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #687385;
            --line: #d9e2ec;
            --accent: #2563eb;
            --accent-dark: #1d4ed8;
            --positive: #12805c;
            --negative: #b42318;
            --neutral: #805b10;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background: var(--bg);
            color: var(--ink);
            font-family: Arial, Helvetica, sans-serif;
        }

        main {
            width: min(1120px, calc(100% - 32px));
            margin: 24px auto;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 380px;
            gap: 20px;
        }

        .hero,
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(23, 32, 51, 0.07);
        }

        .hero {
            padding: 24px;
            margin-bottom: 20px;
            border-left: 6px solid var(--accent);
        }

        h1,
        h2,
        h3,
        p {
            margin-top: 0;
        }

        h1 {
            margin-bottom: 8px;
            font-size: clamp(28px, 4vw, 42px);
            line-height: 1.08;
        }

        h2 {
            font-size: 20px;
        }

        .muted {
            color: var(--muted);
            line-height: 1.5;
        }

        .panel {
            padding: 20px;
        }

        form {
            display: grid;
            gap: 14px;
        }

        textarea {
            width: 100%;
            min-height: 180px;
            resize: vertical;
            padding: 14px;
            border: 1px solid var(--line);
            border-radius: 8px;
            font: inherit;
            line-height: 1.45;
            color: var(--ink);
            background: #fbfdff;
        }

        textarea:focus {
            border-color: var(--accent);
            outline: 3px solid rgba(37, 99, 235, 0.14);
        }

        button {
            justify-self: start;
            min-height: 44px;
            padding: 0 18px;
            border: 0;
            border-radius: 7px;
            background: var(--accent);
            color: #ffffff;
            font-weight: 700;
            cursor: pointer;
        }

        button:hover {
            background: var(--accent-dark);
        }

        .result {
            margin-top: 18px;
            padding: 16px;
            border-radius: 8px;
            background: #eef4ff;
            border: 1px solid #c7d7fe;
        }

        .label {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            color: #ffffff;
            font-weight: 700;
            font-size: 14px;
        }

        .Positive {
            background: var(--positive);
        }

        .Negative {
            background: var(--negative);
        }

        .Neutral {
            background: var(--neutral);
        }

        .history {
            display: grid;
            gap: 10px;
            max-height: 680px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .history-item {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
            background: #fbfdff;
        }

        .history-item p {
            margin-bottom: 8px;
            line-height: 1.4;
        }

        .score {
            color: var(--muted);
            font-size: 14px;
        }

        .error {
            margin-top: 18px;
            padding: 14px;
            border-radius: 8px;
            background: #fff1f0;
            border: 1px solid #f5c2bd;
            color: #8f1d15;
        }

        @media (max-width: 860px) {
            main {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main>
        <section>
            <div class="hero">
                <h1>Sentiment Analyzer</h1>
                <p class="muted">Classify reviews and tweets as Positive, Negative, or Neutral. Every prediction is stored in the database.</p>
            </div>

            <div class="panel">
                <h2>Analyze Text</h2>
                <form method="post">
                    <textarea name="input_text" placeholder="Type a review or tweet here...">{{ input_text }}</textarea>
                    <button type="submit">Predict Sentiment</button>
                </form>

                {% if error %}
                    <div class="error">{{ error }}</div>
                {% endif %}

                {% if result %}
                    <div class="result">
                        <h3>Prediction Result</h3>
                        <p><span class="label {{ result.predicted_sentiment }}">{{ result.predicted_sentiment }}</span></p>
                        <p class="score">Score: {{ result.prediction_score }}</p>
                        <p class="muted">Saved prediction ID: {{ result.id }}</p>
                    </div>
                {% endif %}
            </div>
        </section>

        <aside class="panel">
            <h2>Prediction History</h2>
            <div class="history">
                {% if history %}
                    {% for item in history %}
                        <div class="history-item">
                            <p>{{ item.input_text }}</p>
                            <p><span class="label {{ item.predicted_sentiment }}">{{ item.predicted_sentiment }}</span></p>
                            <p class="score">Score: {{ item.prediction_score }} | {{ item.created_at }}</p>
                        </div>
                    {% endfor %}
                {% else %}
                    <p class="muted">No predictions saved yet.</p>
                {% endif %}
            </div>
        </aside>
    </main>
</body>
</html>
"""


def format_score(score):
    if score is None:
        return "N/A"

    return f"{score:.4f}"


def serialize_result(result):
    if result is None:
        return None

    return {
        "id": result["id"],
        "input_text": result["input_text"],
        "predicted_sentiment": result["predicted_sentiment"],
        "prediction_score": format_score(result["prediction_score"]),
        "created_at": result["created_at"],
    }


def serialize_history_row(row):
    return {
        "id": row.id,
        "input_text": row.input_text,
        "predicted_sentiment": row.predicted_sentiment,
        "prediction_score": format_score(row.prediction_score),
        "created_at": row.created_at,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get(
            "input_text",
            ""
        ).strip()

        if not input_text:
            error = "Please enter text to analyze."
        else:
            try:
                result = serialize_result(
                    predict_and_store(
                        input_text
                    )
                )
            except Exception as exc:
                error = str(exc)

    history = [
        serialize_history_row(row)
        for row in get_all_predictions()
    ]

    return render_template_string(
        PAGE,
        result=result,
        error=error,
        input_text=input_text,
        history=history
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
