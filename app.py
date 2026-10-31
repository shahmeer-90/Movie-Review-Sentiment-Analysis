
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO
    )

    model.eval()

    return tokenizer, model


try:

    tokenizer, model = load_model()

    st.success("AI Model Loaded Successfully!")

    review = st.text_area(
        "Write your movie review:",
        placeholder="This movie was amazing..."
    )

    if st.button("Analyze Review"):

        if review.strip():

            inputs = tokenizer(
                review,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            # Remove token_type_ids for DistilBERT
            inputs.pop("token_type_ids", None)

            with torch.no_grad():

                outputs = model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                confidence = (
                    probabilities[0][predicted_class].item() * 100
                )

            label = model.config.id2label.get(
                predicted_class,
                str(predicted_class)
            ).upper()

            # Handle standard binary sentiment labels
            if label in ["POSITIVE", "LABEL_1", "1"]:
                st.subheader("Prediction Result")
                st.success("😊 Positive Review")

            elif label in ["NEGATIVE", "LABEL_0", "0"]:
                st.subheader("Prediction Result")
                st.error("😞 Negative Review")

            else:
                st.subheader("Prediction Result")
                st.write(f"Model Prediction: {label}")

            st.write(f"Confidence: {confidence:.2f}%")

        else:

            st.warning("Please enter a review first.")

except Exception as e:

    st.error("Unable to load model.")
    st.exception(e)
