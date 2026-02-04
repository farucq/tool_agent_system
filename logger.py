
from datetime import datetime

def log_decision(text):
    with open("tool_agent/logs/decisions.log", "a") as f:
        f.write(f"{datetime.now()} | {text}\n")
