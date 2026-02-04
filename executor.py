
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

def run_vader(texts):
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]

def run_textblob(texts):
    return [TextBlob(t).sentiment.polarity for t in texts]

def run_huggingface(texts):
    classifier = pipeline("sentiment-analysis")
    return classifier(texts)

def execute(tool, data):
    if tool == "vader":
        return run_vader(data)
    elif tool == "textblob":
        return run_textblob(data)
    elif tool == "huggingface":
        return run_huggingface(data)
