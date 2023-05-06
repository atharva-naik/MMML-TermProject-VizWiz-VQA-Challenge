# sample examples for qualitative error analysis:

# sample exaples for qual analysis from ViLT skill classifier:
import os
import copy
import json
import random
import pandas as pd
from typing import *
from collections import defaultdict

random.seed(42)

# vilt_skill_clf_pred = json.load(open("./experiments/vilt_skill_clf/predict_logs.json"))["preds"]
# skill_data = pd.read_csv("./data/skill/vizwiz_skill_typ_val.csv").to_dict("records")
# ctr = 0
# skill_label_wise = defaultdict(lambda: {"false_pos": [], "false_neg": []})
# for i, rec in enumerate(vilt_skill_clf_pred):
#     for label in rec["true"]:
#         if label not in rec["pred"]:
#             skill_label_wise[label]["false_neg"].append(i)
#     for label in rec["pred"]:
#         if label not in rec["true"]:
#             skill_label_wise[label]['false_pos'].append(i)
# skill_label_wise = dict(skill_label_wise)
# df = []
# for skill in skill_label_wise:
#     print(f"{skill}:")
#     FPs = skill_label_wise[skill]["false_pos"]
#     nFP = len(FPs)
#     FNs = skill_label_wise[skill]["false_neg"]
#     nFN = len(FNs)
#     print("FP:", nFP, "FN:", nFN)
#     for i in random.sample(FPs, min(5, nFP)):
#         df.append({
#             "image": skill_data[i]["IMG"],
#             "question": skill_data[i]["QSN"],
#             "error_type": skill + " False Positive",
#         })
#     for id in random.sample(FNs, min(5, nFN)):
#         df.append({
#             "image": skill_data[i]["IMG"],
#             "question": skill_data[i]["QSN"],
#             "error_type": skill + " False Negative",
#         })
# df = pd.DataFrame(df)
# df.to_csv("./vilt_skill_clf_error_anal.csv", index=False)

df_base_vs_fusion = []
df_fusion_vs_skill = []

val_data = json.load(open("./data/VQA/val.json"))
base_clip = json.load(open("./experiments/clip_multimodal/formatted_pred.json"))
skill_clip = json.load(open("./experiments/skill_aware_clip2/formatted_pred.json"))
fusion_clip = json.load(open("./experiments/skill_unaware_clip/formatted_pred.json"))

def compute_all_annotators_score_for_inst(ans: str, refs: List[str]):
    votes = defaultdict(lambda:0)
    for ref_ans in refs: votes[ref_ans] += 1
    
    return min(votes.get(ans,0)/3, 1)

for bi, si, fi, val_refs in zip(base_clip, skill_clip, fusion_clip, val_data):
    bi = bi["answer"]
    si = si["answer"]
    fi = fi["answer"]
    refs = [d["answer"] for d in val_refs["answers"]]
    
    bia = compute_all_annotators_score_for_inst(bi, refs)
    sia = compute_all_annotators_score_for_inst(si, refs)
    fia = compute_all_annotators_score_for_inst(fi, refs)
    inst = {
        "image": val_refs["image"],
        "question": val_refs["question"],
        "refs": refs,
    }

    if bia < fia or bia > fia:
        inst_ = copy.deepcopy(inst)
        inst_["base"] = bi
        inst_["fusion"] = fi
        inst_["win"] = "fusion wins" if bia < fia else "base wins"
        df_base_vs_fusion.append(inst_)

    if sia < fia or sia > fia:
        inst_ = copy.deepcopy(inst)
        inst_["fusion"] = fi
        inst_["skill"] = si
        inst_["win"] = "skill wins" if fia < sia else "fusion wins"
        df_fusion_vs_skill.append(inst_)

df_base_vs_fusion = pd.DataFrame(df_base_vs_fusion)
df_base_vs_fusion.to_csv("./clip_base_vs_fusion_error_anal.csv", index=False)
print(len(df_base_vs_fusion))
print(df_base_vs_fusion["win"].value_counts())

df_fusion_vs_skill = pd.DataFrame(df_fusion_vs_skill)
df_fusion_vs_skill.to_csv("./clip_fusion_vs_skill_error_anal.csv", index=False)
print(len(df_fusion_vs_skill))
print(df_fusion_vs_skill["win"].value_counts())