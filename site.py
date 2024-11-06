import streamlit as st
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

# Function to load the models and tokenizers/vectorizers
def load_models():
    # Load LSTM model and tokenizer
    with open('lstm_tokenizer.pkl', 'rb') as tokenizer_file:
        lstm_tokenizer = pickle.load(tokenizer_file)
    with open('lstm_spam_classifier.pkl', 'rb') as lstm_model_file:
        lstm_model = pickle.load(lstm_model_file)

    # Load Naive Bayes model and vectorizer
    with open('naive_bayes_vectorizer.pkl', 'rb') as vectorizer_file:
        nb_vectorizer = pickle.load(vectorizer_file)
    with open('naive_bayes_spam_classifier.pkl', 'rb') as nb_model_file:
        nb_model = pickle.load(nb_model_file)

    return lstm_model, lstm_tokenizer, nb_model, nb_vectorizer

# Function to preprocess and predict using LSTM model
def predict_lstm(text, lstm_tokenizer, lstm_model):
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=200, padding='post')
    prediction = lstm_model.predict(padded)
    return "Spam" if prediction[0] > 0.5 else "Not Spam"

# Function to preprocess and predict using Naive Bayes model
def predict_naive_bayes(text, nb_vectorizer, nb_model):
    vectorized_text = nb_vectorizer.transform([text])
    prediction = nb_model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Load the models and tokenizers
lstm_model, lstm_tokenizer, nb_model, nb_vectorizer = load_models()

# Streamlit app layout
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

# Title
st.title("Spam Classifier")

# Dark theme
st.markdown("""
    <style>
        body {
            background-color: #2E2E2E;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Introduction
st.markdown("This app allows you to classify messages as **Spam** or **Not Spam** using two different models. Choose the model you want to use!")

# Input Textbox for user message
text = st.text_area("Enter your message here", "")

# Buttons for choosing model
model_choice = st.radio("Choose which classifier to use:", ("LSTM Classifier", "Naive Bayes Classifier"))

# Prediction button
if st.button("Classify"):
    if model_choice == "LSTM Classifier":
        result = predict_lstm(text, lstm_tokenizer, lstm_model)
        st.write(f"**Prediction using LSTM**: {result}")
    elif model_choice == "Naive Bayes Classifier":
        result = predict_naive_bayes(text, nb_vectorizer, nb_model)
        st.write(f"**Prediction using Naive Bayes**: {result}")

# Footer
st.markdown("Created with ❤️ by Elsayed", unsafe_allow_html=True)
