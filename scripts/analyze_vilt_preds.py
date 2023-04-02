import json
import numpy as np
from collections import defaultdict

def compute_all_annotators_score(pred_file: str="./experiments/frozen_vilt2/predict_logs.json", 
                                 ref_file: str="data/VQA/val.json"):
    preds = json.load(open(pred_file))
    annot = json.load(open(ref_file))
    ref_ans = [[ans['answer'] for ans in d['answers']] for d in annot]
    preds = [rec["pred"] for rec in preds["preds"]]
    inst_scores = []
    for pred, refs in zip(preds, ref_ans):
        votes = defaultdict(lambda:0)
        for ref_ans in refs:
            votes[ref_ans] += 1
        inst_scores.append(min(votes.get(pred,0)/3, 1))

    return np.mean(inst_scores)

# main
if __name__ == "__main__":
    acc_score = compute_all_annotators_score()
    print(f"acc_score: {acc_score}")