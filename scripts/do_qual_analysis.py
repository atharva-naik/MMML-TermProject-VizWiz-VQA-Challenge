# do qualitative analysis of instances grouped by various properties.
import json
import numpy as np
import maptlotlib.pyplot as plt
from collections import defaultdict

def compute_all_annotators_score_per_inst(pred_file: str, #="./experiments/frozen_vilt2/predict_logs.json", 
                                          ref_file: str="data/VQA/val.json"):
    preds = json.load(open(pred_file))
    annot = json.load(open(ref_file))
    ref_ans = [[ans['answer'] for ans in d['answers']] for d in annot]
    if isinstance(preds, dict):
        preds = [rec["pred"] for rec in preds["preds"]]
    elif isinstance(preds, list):
        preds = [rec["answer"] for rec in preds]
    inst_scores = []
    for pred, refs in zip(preds, ref_ans):
        votes = defaultdict(lambda:0)
        for ref_ans in refs:
            votes[ref_ans] += 1
        inst_scores.append(min(votes.get(pred,0)/3, 1))

    return inst_scores

def analyze_perf_by_num_objects(object_preds_file: str="./data/VQA/val_objects_detected/all_detection_labels.json"):
    plt.clf()
    object_preds = json.load(open(object_preds_file))
    object_list = []
    for inst in object_preds:
        object_list.append([])
        for conf, obj_name in zip(inst["scores"], inst["labels"]):
            if conf > 0.7: object_list[-1].append(obj_name)
    num_objects_per_inst = [len(objs) if len(objs) < 5 else 5 for objs in object_list]
    model_preds = {
        "CLIP": "./experiments/clip/pred.json",
        "CLIP Fusion": "./experiments/clip_multimodal/pred.json",
        "GIT": "./experiments/frozen_git/formatted_pred.json",
        "ViLT": "./experiments/frozen_vilt2/predict_logs.json",
        "ResNet": "./experiments/resnet/preds.json",
        "DeBERTa": "results_class.json",
        "FLAN-T5": "results_t5.json",
    }
    y_points = {model_name: [[] for i in range(6)] for model_name in model_preds}
    x_labels = ["0", "1", "2", "3", "4", ">=5"]
    x = range(1, 6+1)
    ind = 0
    diff = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    for model_name, path in model_preds.items():
        inst_scores = compute_all_annotators_score_per_inst(
            pred_file=path,
        )
        for num_obj, score in zip(num_objects_per_inst, inst_scores):
            y_points[model_name][num_obj].append(score)
        for num_obj in range(6):
            y_points[model_name][num_obj] = np.mean(y_points[model_name][num_obj])
        x_points = [x_i+d_i for x_i, d_i in zip(x, diff)]
        plt.bar(x_points, y_points[model_name], label=model_name, width=0.1)
    plt.xlabel("Number of objects")
    plt.ylabel("Accuracy")
    plt.xticks(x, labels=x_labels)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("./plots/perf_by_num_objects.png")

# main
if __name__ == "__main__":
    analyze_perf_by_num_objects("./data/VQA/val_objects_detected/all_detection_labels.json")