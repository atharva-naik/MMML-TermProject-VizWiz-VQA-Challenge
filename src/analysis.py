import os
import re
import json
import spacy
import numpy as np
from typing import *
from colour import Color
import matplotlib.pyplot as plt
from collections import defaultdict
from src.datautils import VizWizVQABestAnsDataset

def check_color(color):
    """check if a string is/describes a color."""
    try:
        # Converting 'deep sky blue' to 'deepskyblue'
        color = color.replace(" ", "")
        Color(color)
        # if everything goes fine then return True
        return True
    except ValueError: # The color code was not found
        return False

def check_number_or_date(ans: str):
    ans = ans.replace(" ", "").replace("-", "").replace(".", "").replace(":", "").replace("/", "").strip()
    try:
        int(ans)
        return True
    except Exception as e: return False

def check_tech(ans: str):
    tech_words = 0 
    tot_words = 0
    words = ans.lower().strip().split()
    if ("computer" in words) or ("laptop" in words) or ("phone" in words) or ("windows 7" in words) or ("windows 8" in words) or ("windows xp" in words):
        return True
    if ans == "computer mouse": return True
    tech_vocab = set([
        "laptop", "windows", "screen", "boot", 
        "booting", 'desktop', "computer", "repair"
        "log", "keyboard", "restart", "restore", 
        "dell", "hp", "java", "program", "lenovo",
        "speaker", "phone", "iphone", "blackberry",
        "window", "browser", "error", "message", "file",
        "booted", "mac", "os", "keyboard", "system", "blank"
        "problems", "problem", "reset", "start", "adapter",
        "power", "hard", "drive", "image", "web", "internet",
        "cell", "monitor", "headphones", "skype", 'installation',
        "photo", "photos", "radio", "macbook", "ipad", "battery",
        "memory", "diagnostics", "diagnostic", "driver", "dialog",
        "security", "disk", "disc", "program", "programming", "app",
        "updates", "cords", "cord", "icons", "cd", "recovery", "password",
    ])
    for word in words:
        if word in tech_vocab:
            tech_words += 1
        tot_words += 1
    if tech_words/tot_words >= 0.5:
        return True
    return False

def analyze_object_detections(det_file: str, split_name: str="val"):
    object_dets = {}
    obj_hist_path = os.path.join(
        "./plots", split_name, 
        "object_detection_histogram.png",
    )
    ctr_dict = defaultdict(lambda:0)
    with open(det_file, "r") as f:
        object_dets = json.load(f)
    for rec in object_dets:
        ctr_dict[sum(rec['selected'])] += 1
    # maxm = np.max(list(ctr_dict.keys())) # 0 
    # minm = np.min(list(ctr_dict.keys())) # 50
    bin_edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]
    bin_counts = np.zeros(len(bin_edges)+1)
    plt.clf()
    plt.title(f"Histogram of number of objects detected for {split_name} images")
    x = range(1, len(bin_counts)+1)
    for rec in object_dets:
        ind = len(bin_edges)
        num_obj = sum(rec['selected'])
        for i, (l, u) in enumerate(bin_edges):
            if l <= num_obj < u:
                ind = i
                break
        bin_counts[ind] += 1
    y = bin_counts
    xlabels = ["0", "1", "2", "3", "4", "5", ">5"]
    w = 0.8
    bar = plt.bar(x, y, color="red", width=w)
    for rect in bar:
        h = rect.get_height()
        plt.text(0.4+rect.get_x(), h, f'{int(h)}', ha='center', va='bottom')
    plt.xticks(x, labels=xlabels, rotation=0)
    plt.xlabel("number of objects")
    plt.ylabel("number of instances")
    plt.tight_layout()
    plt.savefig(obj_hist_path)
    
    return bin_edges, bin_counts

ALL_FOODS = json.load(open("val_foods.json"))
def get_rule_based_a_type(ans: str):
    global ALL_FOODS
    ans = ans.strip()
    if ans in ["yes", "no", "unanswerable", "unsuitable"]:
        return ans
    if ans.startswith("unanswerable"):
        return "unanswerable"
    if ans in ALL_FOODS: return "food"
    if check_number_or_date(ans): return "number"
    if check_tech(ans): return "technology"
    # if ans.lower() in [
    #     "white", "red", "blue", "grey", 
    #     "green", "yellow", "light purple"
    #     "orange", "brown", "black"]: return "color"
    if check_color(ans.lower()): return "color"
    # num_regex = "^[0-9]+-[0-9]+$"
    # if re.match(num_regex, ans.replace(" ","")) is not None:
        # return 'number' # range of numbers type answer
    return 'other'

def analyze_answer_types(annot_path: str, images_dir: str):
    dataset = VizWizVQABestAnsDataset(
        annot_path=annot_path,
        images_dir=images_dir,
    )
    builtin_a_types = defaultdict(lambda:0)
    regex_based_a_types = defaultdict(lambda:0)
    other_ans = set()
    for rec in dataset:
        a_type = dataset.ind_to_a_type[rec["answer_type"]]
        builtin_a_types[a_type] += 1
        regex_based_a_types[get_rule_based_a_type(rec["answer"])] += 1
        if get_rule_based_a_type(rec["answer"]) == 'other':
            other_ans.add(rec["answer"])

    with open("other_ans.set.json", "w") as f:
        json.dump(list(other_ans), f, indent=4)
    print(other_ans)

    builtin_a_types = dict(builtin_a_types)
    regex_based_a_types = dict(regex_based_a_types)
    for key, value in builtin_a_types.items():
        builtin_a_types[key] = 100*value/len(dataset)
        print(f"{key}: {builtin_a_types[key]:.2f}%")
    print("#"*30)
    for key, value in regex_based_a_types.items():
        regex_based_a_types[key] = 100*value/len(dataset)
        print(f"{key}: {regex_based_a_types[key]:.2f}%")

    return builtin_a_types, regex_based_a_types

def plot_piechart_from_dict(categ_dict: Dict[str, int], path: str, title: str):
    """referenced form this demo: https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html

    Plot a pie chart given a dictionary of counts and save it to the given path.
    ### Inputs:
    - categ_dict: dictionary of counts per category
    - path: path where the pie chart is to be saved.
    """
    labels = [k.title() for k in categ_dict.keys()]
    sizes = list(categ_dict.values())
    sizes = [100*v/sum(sizes) for v in sizes]
    explode = [0 for _ in sizes]
    
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode,# labels=labels, 
           # autopct='%1.1f%%', 
           shadow=True, startangle=90)
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    # title the plot.
    ax.set_title(title)
    # save the pie-chart.
    plt.legend(labels, loc="best")
    plt.tight_layout()
    plt.savefig(path)

def analyze_question_types(dataset, split_name: str):
    os.makedirs(os.path.join(
        "./plots", split_name), 
        exist_ok=True,
    )
    wh_question_types = ["what", "which", "is", "who", "how", "where", "why", "when", "other"]
    q_type_dist = {k:0 for k in wh_question_types}
    tot = 0
    for rec in dataset:
        terms = [w.strip().lower().replace("'s","").replace('"s','') for w in rec["question"].split()]
        q_type = "other"
        for q_type_cand in wh_question_types[:-1]:
            if q_type_cand == terms[0]: q_type = q_type_cand; break
        # for q_type_cand in wh_question_types:
            # if q_type_cand == terms[-1]: q_type = q_type_cand; break
        # for q_type_cand in wh_question_types:
            # if q_type_cand == terms[1:-1]: q_type = q_type_cand; break
        q_type_dist[q_type] += 1
        tot += 1
    for q_type in q_type_dist:
        q_type_dist[q_type] /= tot
    q_type_dist = dict(q_type_dist)
    plot_piechart_from_dict(
        q_type_dist, os.path.join("./plots", split_name, "q_type_dist.png"),
        title=f"Distribution of question types for {split_name} data",
    )
    # format for easier comprehension.
    for k,v in q_type_dist.items():
        q_type_dist[k] = round(100*v, 2)

    return q_type_dist

def analyze_split(dataset, nlp, split_name: str):
    os.makedirs(os.path.join(
        "./plots", split_name), 
        exist_ok=True,
    )
    q_lens_path = os.path.join("./plots", split_name)
    # question length histogram.
    q_len_hist_edges = [
        (1,6),(6,11),(11,21),
        (21,31),(31,51),"> 50",
    ]
    q_len_hist_counts = np.zeros(len(q_len_hist_edges))
    for rec in dataset:
        q_len = len(nlp(rec["question"]))
        ind = len(q_len_hist_edges)-1
        for i, (l, u) in enumerate(q_len_hist_edges[:-1]):
            if l <= q_len < u:
                ind = i
                break
        q_len_hist_counts[ind] += 1
    plt.clf()
    plt.title(f"question lengths histogram for {split_name}")
    x = range(len(q_len_hist_edges))
    y = q_len_hist_counts
    xlabels = [f"{l}-{u-1}" for l,u in q_len_hist_edges[:-1]]+q_len_hist_edges[-1]
    bar = plt.bar(x, y)
    plt.xticks(x, labels=xlabels, rotation=45)
    plt.savefig(q_lens_path)

# main
if __name__ == "__main__":
    # os.makedirs("./plots", exist_ok=True)
    # nlp = spacy.load("en_core_web_lg")
    # # analysis of various splits of the data.
    # for split in ["train", "val", "test"]:
    #     dataset = VizWizVQABestAnsDataset(
    #         f"./{split}.json", 
    #         f"./{split}/",
    #     )
    #     analyze_split(
    #         dataset, nlp, 
    #         split_name=split,
    #     )
    analyze_answer_types(annot_path="val.json", images_dir="./val")