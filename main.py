
from agent import ToolChoosingAgent

if __name__ == "__main__":
    data = [
        "This product is amazing!",
        "Very bad experience",
        "I love this service",
        "Worst purchase ever"
    ]

    agent = ToolChoosingAgent()
    output = agent.run_task("Sentiment Analysis", data)

    print(output)
