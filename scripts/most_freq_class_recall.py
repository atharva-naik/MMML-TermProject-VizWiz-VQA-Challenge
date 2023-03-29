# test the recall at various k's based on class frequencies.
import os
import json
import matplotlib.pyplot as plt
from src.datautils import VizWizVQABestAnsDataset

def plot_recalls(**system_recalls):
    plt.clf()
    y = {}
    for system_name, recall_at_k_dict in system_recalls.items():
        y[" ".join(system_name.split("_")).title()] = list(recall_at_k_dict.values())
        xlabels = list(recall_at_k_dict.keys())
    # print(recall_at_k)
    # y = list(recall_at_k.values())
    x = range(1, len(xlabels)+1)
    for system_name in y:
        plt.plot(
            x, y[system_name], 
            label=system_name,
        )
    plt.xlabel("k")
    plt.ylabel("recall@k")
    plt.title("System recalls vs k")
    plt.xticks(x, labels=xlabels, rotation=45)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./experiments/vizwiz_ans_cand_ranking_recalls.png")

def train_freq_score_recall_analysis(data_path: str="./data/VQA"):
    val_dataset = VizWizVQABestAnsDataset(
        os.path.join(data_path, "val.json"),
        os.path.join(data_path, "val"),
    )  
    train_dataset = VizWizVQABestAnsDataset(
        os.path.join(data_path, "train.json"),
        os.path.join(data_path, "train"),
    )   
    all_classes = json.load(open("experiments/vizwiz_class_vocab.json"))
    class_counts = {ans: 0 for ans in all_classes}
    true_ans = []
    for inst in val_dataset: true_ans.append(inst["answer"])
    for inst in train_dataset: class_counts[inst['answer']] += 1
    count_ord_classes = {k: v for k,v in sorted(class_counts.items(), reverse=True, key=lambda x: x[1])}
    print(list(count_ord_classes.items())[:5])
    count_ord_classes = list(count_ord_classes.keys())
    # true_indices = [all_classes[ans["answer"]] for ans in dataset] 
    # cos_scores = torch.load(cos_scores_path, map_location="cpu")
    recall_at_k = {}
    ks = [1,5,10,15,20,25,30,35,40,45,50,100,200,300,400,500,1000,2000,3000,4000,5000,6000,7000]
    for k in ks:
        recall_at_k[k] = 0
        for i in range(len(true_ans)):
            # print(true_ans[i], count_ord_classes[i])
            if true_ans[i] in count_ord_classes[:k]:
                recall_at_k[k] += 1
        recall_at_k[k] /= len(true_ans)
    plt.clf()
    # print(recall_at_k)
    y = list(recall_at_k.values())
    x = range(len(ks))
    plt.plot(x, y)
    plt.xlabel("k")
    plt.ylabel("recall@k")
    plt.title("Train freq score recalls vs k")
    plt.xticks(x, labels=ks, rotation=45)
    plt.tight_layout()
    plt.savefig("./experiments/train_freq_score_vizwiz_recall.png")
    with open("./experiments/train_freq_score_vizwiz_recall.json", "w") as f:
        json.dump(recall_at_k, f, indent=4)

    return recall_at_k

def oracle_freq_score_recall_analysis(data_path: str="./data/VQA", split: str="val"):
    dataset = VizWizVQABestAnsDataset(
        os.path.join(data_path, f"{split}.json"),
        os.path.join(data_path, split),
    )   
    all_classes = json.load(open("experiments/vizwiz_class_vocab.json"))
    class_counts = {ans: 0 for ans in all_classes}
    true_ans = []
    for inst in dataset: 
        true_ans.append(inst["answer"])
        class_counts[inst['answer']] += 1
    count_ord_classes = {k: v for k,v in sorted(class_counts.items(), reverse=True, key=lambda x: x[1])}
    print(list(count_ord_classes.items())[:5])
    count_ord_classes = list(count_ord_classes.keys())
    # true_indices = [all_classes[ans["answer"]] for ans in dataset] 
    # cos_scores = torch.load(cos_scores_path, map_location="cpu")
    recall_at_k = {}
    ks = [1,5,10,15,20,25,30,35,40,45,50,100,200,300,400,500,1000,2000,3000,4000,5000,6000,7000]
    for k in ks:
        recall_at_k[k] = 0
        for i in range(len(true_ans)):
            # print(true_ans[i], count_ord_classes[i])
            if true_ans[i] in count_ord_classes[:k]: recall_at_k[k] += 1
        recall_at_k[k] /= len(true_ans)
    plt.clf()
    # print(recall_at_k)
    y = list(recall_at_k.values())
    x = range(len(ks))
    plt.plot(x, y)
    plt.xlabel("k")
    plt.ylabel("recall@k")
    plt.title("Freq score recalls vs k")
    plt.xticks(x, labels=ks, rotation=45)
    plt.tight_layout()
    plt.savefig("./experiments/freq_score_vizwiz_recall.png")

    return recall_at_k

if __name__ == "__main__":
    recall_at_k = oracle_freq_score_recall_analysis()
    with open("./experiments/freq_score_vizwiz_recall.json", "w") as f:
        json.dump(recall_at_k, f, indent=4)