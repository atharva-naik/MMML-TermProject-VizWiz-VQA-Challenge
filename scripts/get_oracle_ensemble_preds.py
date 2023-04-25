import json
import pandas as pd

# main
if __name__ == "__main__":
    val_data = json.load(open("data/VQA/val.json"))
    model_wise_preds = {}
    for model, path in {
        "SkillCLIP": "./experiments/skill_aware_clip2/formatted_pred.json",
        "ViLT": "./experiments/frozen_vilt2/formatted_pred.json"
    }.items():
        model_wise_preds[model] = json.load(open(path))
    ensemble_preds = []
    a_type_to_model = {
        "yes/no": "ViLT",
        "other": "SkilLCLIP",
        "number": "SkillCLIP",
        "unanswerable": "SkillCLIP",
    }
    for i, rec in enumerate(val_data):
        if rec['answer_type'] in ["yes/no"]:
            ensemble_preds.append(model_wise_preds["ViLT"][i])
        else: ensemble_preds.append(model_wise_preds["SkillCLIP"][i])
    with open("./experiments/oracle_ensemble/formatted_pred.json", "w") as f:
        json.dump(ensemble_preds, f, indent=4)