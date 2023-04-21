# do qualitative analysis of instances grouped by various properties.
import json
import numpy as np
import matplotlib.pyplot as plt
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
        "FLAN-T5": "result_t5.json",
        "DeBERTa": "result_class.json",
        "ResNet": "./experiments/resnet/preds.json",
        "CLIP": "./experiments/clip/pred.json",
        "CLIP Fusion": "./experiments/clip_multimodal/pred.json",
        "ViLT": "./experiments/frozen_vilt2/predict_logs.json",
        "GIT": "./experiments/frozen_git/formatted_pred.json",
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
        x_points = [x_i+diff[ind] for x_i in x]
        print(x_points)
        plt.bar(x_points, y_points[model_name], label=model_name, width=0.1)
        ind += 1
    plt.xlabel("Number of objects")
    plt.ylabel("VQA Accuracy")
    plt.xticks(x, labels=x_labels)
    plt.title("Variance of VQA accuracy with number of objects")
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.tight_layout()
    plt.savefig("./plots/perf_by_num_objects.png")

def get_query_len_bucket(q_len: int):
    # buckets created:
    # 1-5, 6-10, 11-15, 15-20, 20-25 
    return (q_len-1)//5 

def analyze_perf_by_question_len():
    plt.clf()
    plt.figure(figsize=(12,8), dpi=120)
    ref_file: str="data/VQA/val.json"
    q_lens = [get_query_len_bucket(len(item["question"].split())) for item in json.load(open(ref_file))]
    model_preds = {
        "FLAN-T5": "result_t5.json",
        "DeBERTa": "result_class.json",
        "ResNet": "./experiments/resnet/preds.json",
        "CLIP": "./experiments/clip/pred.json",
        "CLIP Fusion": "./experiments/clip_multimodal/pred.json",
        "ViLT": "./experiments/frozen_vilt2/predict_logs.json",
        "GIT": "./experiments/frozen_git/formatted_pred.json",
    }
    def safe_mean(seq: list):
        if len(seq) == 0: return 0
        else: return np.mean(seq)
    y_points = {model_name: [[] for i in range(10)] for model_name in model_preds}
    x_labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"]
    x = range(1, 10+1)
    ind = 0
    diff = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    for model_name, path in model_preds.items():
        inst_scores = compute_all_annotators_score_per_inst(
            pred_file=path,
        )
        for bucket_index, score in zip(q_lens, inst_scores): 
            y_points[model_name][bucket_index].append(score)
        for bucket in range(10):
            print(x_labels[bucket], len(y_points[model_name][bucket]))
            y_points[model_name][bucket] = safe_mean(y_points[model_name][bucket])
        x_points = [x_i+diff[ind] for x_i in x]
        # print(len(x_points))
        # print(len(y_points[model_name]))
        # print(y_points[model_name])
        plt.bar(x_points, y_points[model_name], label=model_name, width=0.1)
        ind += 1
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xlabel("Question length")
    plt.ylabel("VQA Accuracy")
    plt.xticks(x, labels=x_labels)
    plt.title("Variance of VQA accuracy with question length")
    plt.tight_layout()
    plt.savefig("./plots/perf_by_question_len.png")

# main
if __name__ == "__main__":
    # analyze_perf_by_num_objects("./data/VQA/val_objects_detected/all_detection_labels.json")
    analyze_perf_by_question_len()