import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)


@st.cache_resource
def train_model():
    data = pd.read_csv("fake_news.csv")

    data = data.dropna(subset=["text", "label"])

    data["label"] = data["label"].map({
        "REAL": 1,
        "FAKE": 0
    })

    data = data.dropna(subset=["label"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(1, 2)
    )

    text_features = vectorizer.fit_transform(data["text"])

    model = LogisticRegression(max_iter=1000)
    model.fit(text_features, data["label"])

    return model, vectorizer


try:
    model, vectorizer = train_model()

except Exception as error:
    st.error("The application could not load the dataset.")
    st.write(error)
    st.stop()


st.title("Fake News Detection System")

st.write(
    "Enter a news article below and check whether the model "
    "classifies it as real or fake."
)

st.divider()

news_text = st.text_area(
    "News Article",
    height=240,
    placeholder="Paste the news article here..."
)

if st.button("Analyze News", use_container_width=True):

    if not news_text.strip():
        st.warning("Please enter a news article first.")

    elif len(news_text.split()) < 5:
        st.warning("Please enter a little more text for a better prediction.")

    else:
        text_vector = vectorizer.transform([news_text])

        prediction = model.predict(text_vector)[0]

        probabilities = model.predict_proba(text_vector)[0]

        confidence = max(probabilities) * 100

        st.divider()

        if prediction == 1:
            st.subheader("Prediction: REAL")
            st.write(f"Confidence: {confidence:.2f}%")

            if confidence >= 80:
                st.success("The model has high confidence in this prediction.")
            else:
                st.info("The model has moderate confidence in this prediction.")

        else:
            st.subheader("Prediction: FAKE")
            st.write(f"Confidence: {confidence:.2f}%")

            if confidence >= 80:
                st.error("The model has high confidence in this prediction.")
            else:
                st.warning("The model has moderate confidence in this prediction.")

        st.caption(
            "This prediction is based on patterns learned from the training "
            "dataset and is not a replacement for professional fact-checking."
        )
