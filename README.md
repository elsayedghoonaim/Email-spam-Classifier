# Email Spam Classification

This project aims to build a machine learning model to classify emails as either "ham" (legitimate) or "spam". We explore two different approaches: a Naive Bayes classifier and a Long Short-Term Memory (LSTM) neural network.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Naive Bayes Model](#naive-bayes-model)
5. [LSTM Model](#lstm-model)
6. [Model Evaluation](#model-evaluation)
7. [Usage](#usage)
8. [License](#license)

## Introduction
Email spam has become a significant problem in recent years, with a large portion of email traffic being unwanted or malicious. Effective spam detection is crucial for maintaining a secure and productive email ecosystem. In this project, we develop and compare the performance of two spam detection models: a Naive Bayes classifier and an LSTM neural network.

## Dataset
The dataset used in this project is the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains 5,572 SMS messages labeled as either "ham" (legitimate) or "spam".

## Preprocessing
The text data is preprocessed by converting all characters to lowercase, removing HTML tags, removing non-alphanumeric characters, and tokenizing the messages. The Naive Bayes model uses a CountVectorizer to convert the text into a numerical format, while the LSTM model uses a Tokenizer to convert the text into sequences of integer indices.

## Naive Bayes Model
The Naive Bayes classifier achieves an accuracy of **99%** on the test set. The classification report shows:

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       448
           1       0.97      0.97      0.97        68

    accuracy                           0.99       516
   macro avg       0.98      0.98      0.98       516
weighted avg       0.99      0.99      0.99       516
```

## LSTM Model
The LSTM model, after 10 epochs of training, achieves a test set accuracy of **99.18%** and a test loss of **0.0482**.

## Model Evaluation
The performance of both models is evaluated using accuracy, precision, recall, and F1-score. Additionally, confusion matrices are plotted to visualize the classification results.

## Web App
In addition to the traditional models, a **simple web application** has been created using **Streamlit**. This app allows users to input a message and choose between the two classifiers (Naive Bayes or LSTM) for spam detection. The app provides an interactive and user-friendly interface, where users can:

- Choose between two models: **Naive Bayes** or **LSTM**.
- Enter a message to classify as either "Spam" or "Not Spam".
- Receive the classification result for the entered text.

### Running the Web App
To run the Streamlit app, use the following command:

```bash
streamlit run app.py
```

The app will open in your browser, where you can easily enter a message, select the classifier, and get the result.

### Web App Features:
- **Interactive Input**: Users can type a message directly into the text box.
- **Model Selection**: Users can select the classifier (Naive Bayes or LSTM) they wish to use for classification.
- **Dark Theme**: The app uses a simple dark theme for a modern and appealing look.

The web app provides a quick way to interact with the models without needing to write any code or run notebooks.

## Usage
To use the spam detection models, simply run the Jupyter Notebook or Python script. The notebook includes code for training the models, making predictions on new emails, and visualizing the results.

## License
This project is licensed under the [MIT License](LICENSE).
