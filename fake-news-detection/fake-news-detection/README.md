# Fake News Detection System

A machine learning application that classifies a news article as REAL or FAKE based on patterns learned from a labelled text dataset.

## Features

- News article text input
- REAL / FAKE prediction
- Prediction confidence
- TF-IDF based text processing
- Logistic Regression model
- Streamlit web interface

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Project Structure

```text
fake-news-detection/
├── app.py
├── train_model.py
├── fake_news.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

```text
News Article
     |
     v
TF-IDF Vectorization
     |
     v
Logistic Regression
     |
     v
REAL / FAKE
```

## Run Locally

Install the packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the application:

```bash
streamlit run app.py
```

## Disclaimer

This project is for educational purposes. The prediction is based on patterns learned from the training dataset and is not a replacement for professional fact-checking.

## Author

Vinuthna
