# sample examples for qualitative error analysis:

# sample exaples for qual analysis from ViLT skill classifier:
import os
import json
from collections import defaultdict

vilt_skill_clf2_pred = json.load(open("./experiments/vilt_skill_clf2/predict_logs.json"))["preds"]
ctr = 0
skill_label_wise = defaultdict(lambda: {"false_pos": [], "false_neg": []})
for i, rec in enumerate(vilt_skill_clf2_pred):
    for label in rec["true"]:
        if label not in rec["pred"]:
            skill_label_wise[label]["false_neg"].append(i)
    for label in rec["pred"]:
        if label not in rec["true"]:
            skill_label_wise[label]['false_pos'].append(i)
skill_label_wise = dict(skill_label_wise)
for skill in skill_label_wise:
    print(f"{skill}:")
    print("FP:", len(skill_label_wise[skill]["false_pos"]), "FN:", len(skill_label_wise[skill]["false_neg"]))