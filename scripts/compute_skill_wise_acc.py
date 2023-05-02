import json
import numpy as np
import pandas as pd
from typing import *
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_all_annotators_score_for_inst(ans: str, refs: List[str]):
    votes = defaultdict(lambda:0)
    for ref_ans in refs: votes[ref_ans] += 1
    
    return min(votes.get(ans,0)/3, 1)
    
# PREDS_PATH: str = "./experiments/frozen_vilt2/formatted_pred.json"
def load_vizwiz_gold_labels(path: str="./data/skill/vizwiz_skill_typ_val.csv"):
    img_to_skill = {}
    data = pd.read_csv(path).to_dict("records")
    for item in data:
        assert item["IMG"] not in img_to_skill
        img_to_skill[item["IMG"]] = {
            "TXT": item["TXT"]>2,
            "OBJ": item["OBJ"]>2,
            "COL": item["COL"]>2,
            "CNT": item["CNT"]>2,
            # "OTH": item["OTH"]>2,
        }
    return img_to_skill

def compute_acc(preds_path: str):
    refs_data = json.load(open("./data/VQA/val.json"))
    scores = []
    for item, refs_rec in zip(json.load(open(preds_path)), refs_data):
        refs = [rec["answer"] for rec in refs_rec["answers"]]
        scores.append(compute_all_annotators_score_for_inst(
            item["answer"], refs
        ))

    return np.mean(scores)

def compute_skill_wise_acc(preds_path: str):
    # skill_to_ind = {
    #     "TXT": 0,
    #     "OBJ": 1,
    #     "COL": 2,
    #     "CNT": 3,
    #     "OTH": 4,
    # }
    img_to_skill = load_vizwiz_gold_labels()
    img_to_refs = {item["image"]: [i["answer"] for i in item["answers"]] for item in json.load(open("./data/VQA/val.json"))}
    skill_wise_vqa_acc = defaultdict(lambda:[])
    for item in json.load(open(preds_path)):
        img = item["image"]
        answer = item["answer"]
        skills = img_to_skill.get(img)
        if skills is None: continue
        for skill, value in skills.items():
            if value == 0: continue
            refs = img_to_refs[img]
            skill_wise_vqa_acc[skill].append(
                compute_all_annotators_score_for_inst(
                    answer, refs,
                )
            )
    skill_wise_vqa_acc = dict(skill_wise_vqa_acc)
    for k,v in skill_wise_vqa_acc.items():
        print(f"{k}: {100*np.mean(v):.2f} ({len(v)})")

# main
if __name__ == "__main__":
    for preds_path in [
        "./experiments/frozen_vilt2/formatted_pred.json",
        "./experiments/frozen_git/formatted_pred.json",
        "./experiments/clip_multimodal/formatted_pred.json",
        "./experiments/skill_aware_clip2/formatted_pred.json",
        "./experiments/skill_unaware_clip/formatted_pred.json",
        "./experiments/oracle_ensemble/formatted_pred.json",
        "./experiments/skill_clip_hf/formatted_pred.json",
        "./experiments/skill_clip_hf_vis_proj/formatted_pred.json",
        "./experiments/skill_clip_hf_vis_proj2/formatted_pred.json",
    ]:
        print("\x1b[34;1m# "+preds_path.split("/")[2].strip()+"\x1b[0m")
        compute_skill_wise_acc(preds_path)
        print("TOT:", round(100*compute_acc(preds_path), 2))
