
from agent import ToolChoosingAgent

def load_reviews(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

if __name__ == "__main__":

    reviews = load_reviews("tool_agent/data/sample_5000_reviews.txt")

    print(f"\nLoaded {len(reviews)} user reviews for analysis\n")

    agent = ToolChoosingAgent()
    agent.run_task("Sentiment Analysis on 5,000 Reviews", reviews)
