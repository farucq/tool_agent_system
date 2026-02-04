
import os
import warnings
from contextlib import redirect_stdout, redirect_stderr
import io

# Silence all HF + system logs
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore")

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error()

def silent(func, *args):
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        return func(*args)

def run_vader(texts):
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]

def run_textblob(texts):
    return [TextBlob(t).sentiment.polarity for t in texts]

def run_huggingface(texts, batch=32):
    clf = silent(pipeline, "sentiment-analysis")
    results = []
    for i in range(0, len(texts), batch):
        results.extend(silent(clf, texts[i:i+batch]))
    return results

def execute(tool, data):
    return {
        "vader": run_vader,
        "textblob": run_textblob,
        "huggingface": run_huggingface
    }[tool](data)
