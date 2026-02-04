
from tool_registry import TOOLS
from decision_engine import select_best_tool
from executor import execute, run_vader, run_textblob, run_huggingface
from logger import log_decision
import time
import warnings
import os
from transformers import logging as hf_logging
from contextlib import redirect_stdout, redirect_stderr
import io

# Silence warnings & HF logs
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

class ToolChoosingAgent:

    def silent_run(self, func, texts):
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            return func(texts)

    def runtime(self, func, texts):
        start = time.time()
        self.silent_run(func, texts)
        return round(time.time() - start, 3)

    def accuracy(self, func, texts, labels):
        preds = self.silent_run(func, texts)
        correct = 0

        for pred, label in zip(preds, labels):
            if isinstance(pred, dict):
                sentiment = 1 if pred['label'] == 'POSITIVE' else -1
            else:
                sentiment = 1 if pred > 0 else -1

            if sentiment == label:
                correct += 1

        return round((correct / len(labels)) * 100, 2)

    def run_task(self, task, data):

        labeled_data = [
            ("This product is amazing!", 1),
            ("Very bad experience", -1),
            ("I love this service", 1),
            ("Worst purchase ever", -1)
        ]

        texts = [x[0] for x in labeled_data]
        labels = [x[1] for x in labeled_data]

        best_tool, scores = select_best_tool(TOOLS)

        log_decision(f"Task: {task}")
        log_decision(f"Scores: {scores}")
        log_decision(f"Chosen Tool: {best_tool}")

        print("\n================ TOOL EVALUATION REPORT ================\n")

        print(f"VADER        : {scores['vader']:.2f}")
        print(f"TextBlob     : {scores['textblob']:.2f}")
        print(f"HuggingFace  : {scores['huggingface']:.2f}")

        print("\n-------------------------------------------------------\n")
        print(f"Selected Tool : {best_tool.upper()}")
        print("Justification : Highest weighted performance score")

        print("\n================ PERFORMANCE METRICS ===================\n")
        print("Tool         Runtime (s)     Accuracy (%)")
        print("-----------------------------------------")

        print(f"VADER            {self.runtime(run_vader, texts):<6}         {self.accuracy(run_vader, texts, labels)}")
        print(f"TextBlob         {self.runtime(run_textblob, texts):<6}         {self.accuracy(run_textblob, texts, labels)}")
        print(f"HuggingFace      {self.runtime(run_huggingface, texts):<6}         {self.accuracy(run_huggingface, texts, labels)}")

        print("\n================ EXECUTION RESULTS =====================\n")

        results = execute(best_tool, data)

        for i, score in enumerate(results, 1):
            sentiment = "Positive" if score > 0 else "Negative"
            print(f"Review {i} : {score:>7}   ({sentiment})")

        return results
