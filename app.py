
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download

import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# NLTK RESOURCES
# ============================================================

@st.cache_resource
def download_nltk_resources():

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

download_nltk_resources()


# ============================================================
# LOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="Poushali0108/bilstm-fake-news-model",
        filename="bilstm_fake_news_model.keras",
        repo_type="model"
    )

    model = tf.keras.models.load_model(model_path)

    return model


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_resource
def load_config():

    with open("config.pkl", "rb") as file:
        config = pickle.load(file)

    return config


model = load_model()
tokenizer = load_tokenizer()
config = load_config()

MAX_LEN = config["MAX_LEN"]


# ============================================================
# TEXT PREPROCESSING
# ============================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# PREDICTION
# ============================================================

def predict_news(news_text):

    cleaned_text = clean_text(news_text)

    if not cleaned_text.strip():
        return None, None, None

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability = float(
        model.predict(
            padded_sequence,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:

        result = "TRUE NEWS"
        confidence = probability * 100

    else:

        result = "FAKE NEWS"
        confidence = (1 - probability) * 100

    return result, confidence, probability


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📰 Fake News Detection")

    st.write(
        "A deep learning-based Fake News Detection "
        "system using a Bidirectional LSTM model."
    )

    st.divider()

    st.subheader("Model")

    st.write("Bidirectional LSTM (Bi-LSTM)")

    st.write("Maximum Sequence Length:", MAX_LEN)

    st.write("Classification Threshold: 0.5")

    st.divider()

    st.subheader("Model Selection")

    st.write(
        "Multiple models were evaluated and compared. "
        "The Bi-LSTM model was selected as the final "
        "model based on its performance."
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("📰 Fake News Detection System")

st.markdown(
    """
    ### Detect whether a news article is likely to be Fake or True

    Enter the complete news article below and the trained
    Bi-LSTM model will provide the predicted class along
    with its confidence probability.
    """
)

st.divider()


# ============================================================
# INPUT
# ============================================================

st.subheader("🔎 Enter News Article")

news_text = st.text_area(
    "Paste your news article here:",
    height=300,
    placeholder="Enter or paste the news article..."
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔍 Check News",
    use_container_width=True
):

    if not news_text.strip():

        st.warning(
            "Please enter a news article first."
        )

    else:

        with st.spinner("Analyzing the news article..."):

            result, confidence, probability = predict_news(
                news_text
            )

        st.divider()

        st.subheader("📊 Prediction Result")

        if result == "TRUE NEWS":

            st.success(
                "✅ The model predicts this article as TRUE NEWS."
            )

        else:

            st.error(
                "⚠️ The model predicts this article as FAKE NEWS."
            )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                result
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        st.write("### Confidence")

        st.progress(
            min(int(confidence), 100)
        )


        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Probability of True News:** "
                f"{probability:.4f}"
            )

        with col2:

            st.write(
                f"**Probability of Fake News:** "
                f"{1 - probability:.4f}"
            )


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.subheader("ℹ️ About the Project")

st.write(
    """
    This project performs comparative evaluation of
    multiple machine learning and deep learning models
    for Fake News Detection.

    After comparing the performance of different models,
    the Bidirectional LSTM (Bi-LSTM) model was selected
    as the most suitable model.

    The news article is preprocessed, tokenized and
    converted into a padded sequence before being passed
    to the trained Bi-LSTM model.

    The model produces a probability which is used to
    classify the article as Fake News or True News.
    """
)

st.warning(
    "⚠️ This system is a machine-learning based classifier "
    "and should not be considered an independent fact-checking "
    "system."
)
