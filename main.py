
from agent import ToolChoosingAgent

def load(file):
    with open(file) as f:
        return f.read().splitlines()

if __name__=="__main__":
    reviews=load("tool_agent/sample_5000_reviews.txt")
    agent=ToolChoosingAgent()
    agent.run_task("Sentiment Analysis on 5,000 Reviews", reviews)
