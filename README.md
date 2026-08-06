# Project 4 – Sentiment Analyzer

A machine learning application that classifies text as **Positive**, **Negative**, or **Neutral** using multiple NLP models. The best-performing model is saved and used for future predictions through the command line or a Flask web interface.

## Setup

Place `data.csv` in the project folder with the columns:

```text
review,sentiment
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

## Run the Project

1. Clean the dataset:

```bash
python3 clean_data.py
```

2. Train the models:

```bash
python3 train_models.py
```

3. Start the prediction program:

```bash
python3 predict_sentiment.py
```

Or launch the web interface:

```bash
python3 flask_app.py
```

Open:

```text
http://127.0.0.1:5000
```

All predictions are automatically saved to the project's database.
