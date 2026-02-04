
import yaml

def load_weights():
    with open("tool_agent/config.yaml") as f:
        return yaml.safe_load(f)["constraints"]

def score_tool(meta, w):
    return meta["speed"]*w["speed_weight"] + meta["accuracy"]*w["accuracy_weight"] + meta["resource"]*w["resource_weight"]

def select_best_tool(tools):
    w = load_weights()
    scores = {k: score_tool(v, w) for k,v in tools.items()}
    return max(scores, key=scores.get), scores
