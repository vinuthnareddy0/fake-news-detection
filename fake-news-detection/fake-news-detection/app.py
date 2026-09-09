import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", layout="centered")

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    st.error("Model files are missing. Run train_model.py first.")
    st.stop()

st.title("Fake News Detection System")
st.write("Enter a news article below and check how the trained model classifies it.")

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
            st.success("The model classified this article as real.")
        else:
            st.subheader("Prediction: FAKE")
            st.write(f"Confidence: {confidence:.2f}%")
            st.error("The model classified this article as fake.")

        st.caption(
            "This is an ML-based prediction and should not be treated as a definitive fact check."
        )
