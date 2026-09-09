import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("fake_news.csv")
data = data.dropna(subset=["text", "label"])

data["label"] = data["label"].map({"REAL": 1, "FAKE": 0})
data = data.dropna(subset=["label"])

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed.")
print("Accuracy:", round(accuracy * 100, 2), "%")
print()
print(classification_report(
    y_test,
    predictions,
    target_names=["FAKE", "REAL"]
))

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("model.pkl created.")
print("vectorizer.pkl created.")
