import json
import pandas as pd

# main
if __name__ == "__main__":
    val_data = json.load(open("data/VQA/val.json"))
    model_wise_preds = {}
    for model, path in {
        "SkillCLIP": "./experiments/skill_aware_clip2/formatted_pred.json",
        "SkillCLIPHf": "./experiments/skill_clip_hf/formatted_pred.json",
        "ViLT": "./experiments/frozen_vilt2/formatted_pred.json"
    }.items():
        model_wise_preds[model] = json.load(open(path))
    ensemble_preds = []
    a_type_to_model = {
        "yes/no": "ViLT",
        "other": "SkillCLIPHf",
        "number": "SkillCLIP",
        "unanswerable": "SkillCLIPHf",
    }
    for i, rec in enumerate(val_data):
        model_name =a_type_to_model[rec["answer_type"]]
        ensemble_preds.append(model_wise_preds[model_name][i])
        # if rec['answer_type'] in ["yes/no"]:
            # ensemble_preds.append(model_wise_preds["ViLT"][i])
        # else: ensemble_preds.append(model_wise_preds["SkillCLIP"][i])
    with open("./experiments/oracle_ensemble/formatted_pred.json", "w") as f:
        json.dump(ensemble_preds, f, indent=4)