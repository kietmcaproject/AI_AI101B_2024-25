from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import scipy.special

# Load tokenizer and model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Labels
labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output.logits[0].numpy()
    probabilities = scipy.special.softmax(scores)

    sentiment_index = int(np.argmax(probabilities))
    sentiment = labels[sentiment_index]
    confidence = float(probabilities[sentiment_index])

    # Smart handling: if confidence < 0.5 or close values, set as NEUTRAL
    if abs(probabilities[0] - probabilities[2]) < 0.2 and probabilities[1] > 0.25:
        sentiment = 'NEUTRAL'
        confidence = float(probabilities[1])

    return sentiment, confidence

