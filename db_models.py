from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table
from sqlalchemy.sql import func


metadata = MetaData()


sentiment_predictions = Table(
    "sentiment_predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("input_text", String, nullable=False),
    Column("predicted_sentiment", String, nullable=False),
    Column("prediction_score", Float, nullable=True),
    Column("created_at", DateTime, server_default=func.now(), nullable=False),
)
