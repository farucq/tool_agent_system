
import yaml

def load_weights():
    with open("tool_agent/config.yaml") as f:
        return yaml.safe_load(f)["constraints"]

def score_tool(tool_meta, weights):
    return (
        tool_meta["speed"] * weights["speed_weight"]
        + tool_meta["accuracy"] * weights["accuracy_weight"]
        + tool_meta["resource"] * weights["resource_weight"]
    )

def select_best_tool(tools):
    weights = load_weights()
    scores = {}

    for name, meta in tools.items():
        scores[name] = score_tool(meta, weights)

    best_tool = max(scores, key=scores.get)
    return best_tool, scores
