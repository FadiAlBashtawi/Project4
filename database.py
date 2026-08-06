import os

from sqlalchemy import create_engine


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:1@localhost/class_pos_neg"
)


engine = create_engine(
    DATABASE_URL,
    future=True
)
