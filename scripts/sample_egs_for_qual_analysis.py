# sample examples for qualitative error analysis:

# sample exaples for qual analysis from ViLT skill classifier:
import os
import json
import random
import pandas as pd
from collections import defaultdict

random.seed(42)

vilt_skill_clf_pred = json.load(open("./experiments/vilt_skill_clf/predict_logs.json"))["preds"]
skill_data = pd.read_csv("./data/skill/vizwiz_skill_typ_val.csv").to_dict("records")
ctr = 0
skill_label_wise = defaultdict(lambda: {"false_pos": [], "false_neg": []})
for i, rec in enumerate(vilt_skill_clf_pred):
    for label in rec["true"]:
        if label not in rec["pred"]:
            skill_label_wise[label]["false_neg"].append(i)
    for label in rec["pred"]:
        if label not in rec["true"]:
            skill_label_wise[label]['false_pos'].append(i)
skill_label_wise = dict(skill_label_wise)
df = []
for skill in skill_label_wise:
    print(f"{skill}:")
    FPs = skill_label_wise[skill]["false_pos"]
    nFP = len(FPs)
    FNs = skill_label_wise[skill]["false_neg"]
    nFN = len(FNs)
    print("FP:", nFP, "FN:", nFN)
    for i in random.sample(FPs, min(5, nFP)):
        df.append({
            "image": skill_data[i]["IMG"],
            "question": skill_data[i]["QSN"],
            "error_type": skill + " False Positive",
        })
    for id in random.sample(FNs, min(5, nFN)):
        df.append({
            "image": skill_data[i]["IMG"],
            "question": skill_data[i]["QSN"],
            "error_type": skill + " False Negative",
        })
df = pd.DataFrame(df)
df.to_csv("./vilt_skill_clf_error_anal.csv", index=False)