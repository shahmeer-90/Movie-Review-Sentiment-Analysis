import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

st.set_page_config(
    page_title="Movie Review AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and our AI model "
    "will predict whether it is Positive or Negative."
)

MODEL_REPO = "shahMeer2002/movie-review-distilbert"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    return classifier


try:

    classifier = load_model()

    st.success("AI Model Loaded Successfully!")

    review = st.text_area(
        "Write your movie review:",
        placeholder="This movie was amazing..."
    )

    if st.button("Analyze Review"):

        if review.strip():

            result = classifier(
                review,
                truncation=True,
                max_length=256
            )[0]

            label = result["label"]
            confidence = result["score"] * 100

            st.subheader("Prediction Result")

            if label == "POSITIVE":

                st.success("😊 Positive Review")

            else:

                st.error("😞 Negative Review")

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.warning("Please enter a review first.")

except Exception as e:

    st.error("Unable to load model.")
    st.exception(e)
    