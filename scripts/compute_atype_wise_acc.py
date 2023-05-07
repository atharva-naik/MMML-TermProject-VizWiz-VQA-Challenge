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

def compute_atype_wise_acc(preds_path: str):
    val_data = json.load(open("./data/VQA/val.json"))
    img_to_refs = {item["image"]: [i["answer"] for i in item["answers"]] for item in val_data}
    img_to_atype = {item["image"]: item["answer_type"] for item in val_data}
    img_list = [item["image"] for item in val_data]
    atype_wise_vqa_acc = defaultdict(lambda:[])
    preds_data = json.load(open(preds_path))
    if "preds" in preds_data:
        preds_data = preds_data["preds"]
    for i, item in enumerate(preds_data):
        try: img = item["image"]
        except KeyError: img = img_list[i]
        try: answer = item["answer"]
        except KeyError: answer = item["pred"]
        refs = img_to_refs[img]
        atype = img_to_atype[img]
        inst_score = compute_all_annotators_score_for_inst(answer, refs)
        atype_wise_vqa_acc[atype].append(inst_score)
    atype_wise_vqa_acc = {k: v for k,v in sorted(atype_wise_vqa_acc.items(), key=lambda x: x[0], reverse=True)}
    for k,v in atype_wise_vqa_acc.items():
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
        compute_atype_wise_acc(preds_path)
        print("TOT:", round(100*compute_acc(preds_path), 2))
