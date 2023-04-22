import json
from sklearn.metrics import f1_score

PREDS_PATH: str = "./experiments/vilt_skill_clf2/predict_logs.json"

# main
if __name__ == "__main__":
    skill_to_ind = {
        "TXT": 0,
        "OBJ": 1,
        "COL": 2,
        "CNT": 3,
        "OTH": 4,
    }
    skill_wise_trues = {skill: [] for skill in skill_to_ind}
    skill_wise_preds = {skill: [] for skill in skill_to_ind}
    saved_preds = json.load(open(PREDS_PATH))
    for skill in skill_to_ind:
        for rec in saved_preds['preds']:
            if skill in rec["pred"]:
                skill_wise_preds[skill].append(1)
            else: skill_wise_preds[skill].append(0)
            if skill in rec["true"]:
                skill_wise_trues[skill].append(1)
            else: skill_wise_trues[skill].append(0)
        print(f"{skill} f1_score: {100*f1_score(skill_wise_trues[skill], skill_wise_preds[skill]):.2f} ({sum(skill_wise_trues[skill])}/{len(saved_preds['preds'])} cases)")
    