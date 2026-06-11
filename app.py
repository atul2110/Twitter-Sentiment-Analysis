import streamlit as st
import joblib

# This caches the models so they only load ONE time
@st.cache_resource
def load_my_models():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Call the function
model, vectorizer = load_my_models()

st.title("Twitter Sentiment Analysis")
tweet = st.text_area(
    "Enter a Tweet",
    placeholder="Type something..."
)

if st.button("Analyze"):

    tweet = tweet.lower()

    tweet_vector = vectorizer.transform([tweet])

    prediction = model.predict(tweet_vector)[0]

    if prediction == 1:
        st.success("😊 Positive Sentiment")
    else:
        st.error("😔 Negative Sentiment")