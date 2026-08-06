import pickle
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

from clean_data import clean_text
from db_operations import create_prediction


DEFAULT_MODEL_PATH = Path("best_sentiment_model.pkl")


def load_saved_model(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train_models.py first."
        )

    if model_path.suffix == ".pkl":
        with model_path.open("rb") as file:
            return pickle.load(file)

    return joblib.load(
        model_path
    )


def build_features(payload, text):
    cleaned_text = clean_text(
        text
    )

    if payload["feature_type"] == "embedding":
        embedding_model = SentenceTransformer(
            payload["embedding_model_name"]
        )

        return embedding_model.encode(
            [cleaned_text]
        )

    return [
        cleaned_text
    ]


def get_prediction_score(model, features, prediction):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            features
        )[0]

        classes = list(
            model.classes_
        )

        prediction_index = classes.index(
            prediction
        )

        return float(
            probabilities[prediction_index]
        )

    return None


def predict_and_store(text, model_path=DEFAULT_MODEL_PATH):
    payload = load_saved_model(
        model_path
    )

    features = build_features(
        payload,
        text
    )

    model = payload["model"]
    prediction = model.predict(
        features
    )[0]

    score = get_prediction_score(
        model,
        features,
        prediction
    )

    row = create_prediction(
        input_text=text,
        predicted_sentiment=prediction,
        prediction_score=score
    )

    return {
        "id": row.id,
        "input_text": row.input_text,
        "predicted_sentiment": row.predicted_sentiment,
        "prediction_score": row.prediction_score,
        "created_at": row.created_at,
    }
