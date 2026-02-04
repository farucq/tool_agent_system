
from tool_registry import TOOLS
from decision_engine import select_best_tool
from executor import execute, run_vader, run_textblob, run_huggingface
from logger import log_decision
import time

class ToolChoosingAgent:

    def runtime(self, fn, data):
        t=time.time(); fn(data); return round(time.time()-t,3)

    def accuracy(self, fn, data, labels):
        preds=fn(data)
        correct=0
        for p,l in zip(preds,labels):
            s=1 if (p['label']=='POSITIVE' if isinstance(p,dict) else p>0) else -1
            if s==l: correct+=1
        return round(correct/len(labels)*100,2)

    def run_task(self, task, data):

        eval_data=[("This product is amazing!",1),("Very bad experience",-1),
                   ("I love this service",1),("Worst purchase ever",-1)]

        texts=[x[0] for x in eval_data]
        labels=[x[1] for x in eval_data]

        best,scores=select_best_tool(TOOLS)

        log_decision(f"Task: {task}")
        log_decision(f"Scores: {scores}")
        log_decision(f"Chosen Tool: {best}")

        print("\n================ TOOL EVALUATION REPORT ================\n")
        for k in scores:
            print(f"{k.capitalize():12}: {scores[k]:.2f}")

        print("\nSelected Tool :", best.upper())
        print("\n================ PERFORMANCE METRICS ===================\n")
        print("Tool        Runtime(s)   Accuracy(%)")

        print(f"VADER        {self.runtime(run_vader,texts)}        {self.accuracy(run_vader,texts,labels)}")
        print(f"TextBlob     {self.runtime(run_textblob,texts)}        {self.accuracy(run_textblob,texts,labels)}")
        print(f"HuggingFace  {self.runtime(run_huggingface,texts)}        {self.accuracy(run_huggingface,texts,labels)}")

        print("\n================ EXECUTION RESULTS =====================\n")

        results=execute(best,data)
        for i,r in enumerate(results[:4],1):
            print(f"Review {i} : {r}")

        return results
