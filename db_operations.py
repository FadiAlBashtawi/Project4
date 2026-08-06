from sqlalchemy import insert, select

from database import engine
from db_models import metadata, sentiment_predictions


def create_tables():
    metadata.create_all(
        engine
    )


def create_prediction(input_text, predicted_sentiment, prediction_score=None):
    create_tables()

    stmt = insert(sentiment_predictions).values(
        input_text=input_text,
        predicted_sentiment=predicted_sentiment,
        prediction_score=prediction_score,
    )

    with engine.begin() as conn:
        result = conn.execute(stmt)
        prediction_id = result.inserted_primary_key[0]
        row = conn.execute(
            select(sentiment_predictions).where(
                sentiment_predictions.c.id == prediction_id
            )
        ).first()

    return row


def get_all_predictions():
    create_tables()

    stmt = (
        select(sentiment_predictions)
        .order_by(sentiment_predictions.c.created_at.desc())
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()

    return rows
