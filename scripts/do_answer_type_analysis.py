# analyze answer type prediction accuracies of model preds.
# for this the "best answer" references will be used.

import json
from typing import *
from collections import defaultdict
from sklearn.metrics import f1_score
from src.datautils import VizWizVQABestAnsDataset

train_data = VizWizVQABestAnsDataset(
    "./data/VQA/train.json",
    "./data/VQA/train"
)
val_data = VizWizVQABestAnsDataset(
    "./data/VQA/val.json",
    "./data/VQA/val"
)
# build class name to answer type mapping
# for this we are using the "gold" answer types right now, even though we know we can do better.
class_to_ans_type = defaultdict(lambda:set())
for rec in val_data:
    class_to_ans_type[rec['answer']].add(rec['answer_type'])
for rec in train_data:
    class_to_ans_type[rec['answer']].add(rec['answer_type'])
for k,v in class_to_ans_type.items():
    if len(v) > 1: pass # print(k)
    else: class_to_ans_type[k] = list(v)[0]
# {0: 'unanswerable', 1: 'other', 2: 'yes/no', 3: 'number'}
# fix cases where multiple answer types are mapped (40 such cases).
manual_corrections = {
    "right": 1, "nothing": 1, "white": 1,
    "yes": 2, "clive cussler dirk cussler": 1,
    "no": 2, "blue": 1, "8": 3, "0": 3, "soup": 1,
    "coffee": 1, "cranberry": 1, "5": 3, "regular": 1,
    "black": 1, "90": 3, "bed": 1, "20": 3, "green": 1,
    "light": 1, "425": 3, "78": 3, "75": 3, "low": 1,
    "50": 3, "medicine": 1, "10": 3, "1": 3, "3": 3, 
    "campbells": 1, "canned food": 1, "bird": 1,
    "acetaminophen": 1, "lean cuisine": 1, "69": 3,
    "81": 3, "4": 3, "bread": 1, "2854.562": 3, "mahou": 1,
}
for k, v in manual_corrections.items():
    class_to_ans_type[k] = v
class_to_ans_type = dict(class_to_ans_type)

def compute_ans_type_acc(pred_ans_list: List[str]):
    global val_data
    global class_to_ans_type
    tot, matches = 0, 0
    a_type_wise_tot = defaultdict(lambda:0)
    a_type_wise_matches = defaultdict(lambda:0)
    for i, ans in enumerate(pred_ans_list):
        pred_a_type = class_to_ans_type[ans]
        best_a_type = class_to_ans_type[val_data[i]["answer"]]
        match_ = int(pred_a_type == best_a_type)
        tot += 1
        matches += int(match_)
        a_type = val_data.ind_to_a_type[best_a_type]
        a_type_wise_tot[a_type] += 1
        a_type_wise_matches[a_type] += match_

    a_type_wise_tot = dict(a_type_wise_tot)
    a_type_wise_matches = dict(a_type_wise_matches)
    for k,v in a_type_wise_matches.items():
        a_type_wise_matches[k] = round(100*v/a_type_wise_tot[k], 2)

    return round(100*matches/tot, 2), a_type_wise_matches

def print_result(model_name: str, tot: float, a_type_wise: Dict[str, float]):
    print(f"model: {model_name}")
    print(f"tot: {100*tot:.2f}")
    for k, v in a_type_wise.items():
        print(f"{k} {100*v:.2f}")
    print("-"*30)

def compute_ans_type_f_scores(pred_ans_list: List[str]):
    global val_data
    global class_to_ans_type
    trues, preds = [], []
    a_type_wise_trues = {a_type: [] for a_type in val_data.a_type_to_ind}
    a_type_wise_preds = {a_type: [] for a_type in val_data.a_type_to_ind}
    for i, ans in enumerate(pred_ans_list):
        pred_a_type = class_to_ans_type[ans]
        best_a_type = class_to_ans_type[val_data[i]["answer"]]
        trues.append(best_a_type)
        preds.append(pred_a_type)
        a_type = val_data.ind_to_a_type[best_a_type]
        for ind, name in val_data.ind_to_a_type.items():
            a_type_wise_trues[name].append(int(best_a_type == ind))
            a_type_wise_preds[name].append(int(pred_a_type == ind))

    f_score = f1_score(trues, preds, average="weighted")
    a_type_wise_f_scores = {}
    a_type_wise_trues = dict(a_type_wise_trues)
    a_type_wise_preds = dict(a_type_wise_preds)
    # print(a_type_wise_trues["unanswerable"])
    # print(len(a_type_wise_trues["unanswerable"]))
    for a_type in list(val_data.a_type_to_ind):
        a_type_wise_f_scores[a_type] = f1_score(
            a_type_wise_trues[a_type],
            a_type_wise_preds[a_type],
            average="binary",
        )

    return f_score, a_type_wise_f_scores

if __name__ == "__main__":
    for path in [
            "./experiments/clip/pred.json",
            "./experiments/clip_multimodal/pred.json",
            # "./experiments/frozen_git/formatted_pred.json",
            # "./experiments/frozen_git/preds.json",
            "./experiments/frozen_vilt2/predict_logs.json",
            "./experiments/resnet/preds.json",
            "./experiments/skill_aware_clip2/pred.json"
        ]:
        data = json.load(open(path))
        if isinstance(data, list):
            data = [i['answer'] for i in data]
        elif isinstance(data, dict):
            data = [i['pred'] for i in data["preds"]]
        tot, a_type_wise = compute_ans_type_f_scores(data) 
        print_result(path, tot, a_type_wise)
        # print(path, f"{tot}%", a_type_wise)
        # deberta predicts all unanswerable
    data = ["unanswerable" for _ in data]
    tot, a_type_wise = compute_ans_type_f_scores(data)
    print_result("deberta", tot, a_type_wise)
    # print("deberta", f"{tot}%", a_type_wise)