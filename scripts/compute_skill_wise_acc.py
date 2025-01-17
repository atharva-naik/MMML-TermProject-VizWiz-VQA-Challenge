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
    preds_data = json.load(open(preds_path))
    access_key = "answer"
    if "preds" in preds_data:
        preds_data = preds_data["preds"]
        access_key = "pred"
    for item, refs_rec in zip(preds_data, refs_data):
        refs = [rec["answer"] for rec in refs_rec["answers"]]
        scores.append(compute_all_annotators_score_for_inst(
            item[access_key], refs
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
    val_data = json.load(open("./data/VQA/val.json"))
    img_to_refs = {item["image"]: [i["answer"] for i in item["answers"]] for item in val_data}
    img_list = [item["image"] for item in val_data]
    skill_wise_vqa_acc = defaultdict(lambda:[])
    preds_data = json.load(open(preds_path))
    if "preds" in preds_data:
        preds_data = preds_data["preds"]
    for i, item in enumerate(preds_data):
        try: img = item["image"]
        except KeyError: img = img_list[i]
        try: answer = item["answer"]
        except KeyError: answer = item["pred"]
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
    skill_wise_vqa_acc = {k: v for k,v in sorted(skill_wise_vqa_acc.items(), key=lambda x: x[0], reverse=True)}
    for k,v in skill_wise_vqa_acc.items():
        print(f"{k}: {100*np.mean(v):.2f} ({len(v)})")

# main
if __name__ == "__main__":
    for preds_path in [
        "./result_t5.json",
        "./result_deberta.json",
        "./experiments/resnet/pred.json",
        "./experiments/clip/pred.json",
        "./experiments/zero_shot_git_large/formatted_pred.json",
        "./experiments/clip_multimodal/formatted_pred.json",
        "./experiments/frozen_vilt2/formatted_pred.json",
        "./experiments/frozen_git/formatted_pred.json",
        "./experiments/skill_unaware_clip/formatted_pred.json",
        "./experiments/skill_aware_clip_multitask/formatted_pred.json",
        "./experiments/skill_aware_clip2/formatted_pred.json",
        # "./experiments/skill_aware_clip3/formatted_pred.json",
        "./experiments/skill_aware_clip_nsctxt/formatted_pred.json",
        "./experiments/skill_aware_clip_nobj/formatted_pred.json",
        # "./experiments/oracle_ensemble/formatted_pred.json",
        # "./experiments/skill_clip_hf/formatted_pred.json",
        # "./experiments/skill_clip_hf_vis_proj/formatted_pred.json",
        # "./experiments/skill_clip_hf_vis_proj2/formatted_pred.json",
    ]:
        try: print("\x1b[34;1m# "+preds_path.split("/")[2].strip()+"\x1b[0m")
        except IndexError:
             print("\x1b[34;1m# "+preds_path.split("/")[1].strip()+"\x1b[0m")
        print("TOT:", round(100*compute_acc(preds_path), 2))
        compute_skill_wise_acc(preds_path)