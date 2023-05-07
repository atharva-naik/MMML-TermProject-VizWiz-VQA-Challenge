# compute answer similarity using sentence BERT between answer
import os
import json
import numpy as np
from typing import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-roberta-large-v1"
ANS_EMBS_PATH = f"./ans_bertscore/{MODEL_NAME}_ans_embs.npy"
def dump_ref_embs_lookup(model, all_refs: List[List[str]]):
    uniq_answers = set()
    for inst_refs in all_refs:
        for ref in inst_refs: uniq_answers.add(ref)
    uniq_answers = sorted(list(uniq_answers))
    ans_to_emb_lookup = {}
    embs = model.encode(uniq_answers, batch_size=32, show_progress_bar=True)
    for i, ans in enumerate(uniq_answers):
        ans_to_emb_lookup[ans] = embs[i]
    np.save(ANS_EMBS_PATH, ans_to_emb_lookup)

def dump_pred_embs_lookup(model, preds: List[str], pred_embs_path: str):
    uniq_answers = sorted(list(set(preds)))
    pred_to_emb_lookup = {}
    embs = model.encode(uniq_answers, batch_size=32, show_progress_bar=True)
    for i, ans in enumerate(uniq_answers):
        pred_to_emb_lookup[ans] = embs[i]
    np.save(pred_embs_path, pred_to_emb_lookup)

def compute_inst_bertscore(ans_emb: np.ndarray, ref_embs: List[np.ndarray], thresh: float=0.75):
    scores = util.cos_sim(ans_emb, ref_embs)[0]
    consensus_score = scores[np.argpartition(scores, -3)[-3:]].mean()
    return bool(consensus_score >= thresh)
# main
if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    val_refs = [[d["answer"] for d in rec["answers"]] for rec in json.load(open("./data/VQA/val.json"))]
    if not os.path.exists(ANS_EMBS_PATH): dump_ref_embs_lookup(model, val_refs)
    ans_to_emb_lookup = np.load(ANS_EMBS_PATH, allow_pickle=True)[()]
    thresholds = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    y = defaultdict(lambda:[])
    for thresh in thresholds:
        print(f"\n\x1b[34;1m# thresh={thresh}\x1b[0m")
        for evaluating_model in [
            "t5", "deberta", "resnet", "clip",
            "zero_shot_git_large", "clip_multimodal", 
            "frozen_vilt2", "frozen_git", "skill_unaware_clip", 
            "skill_aware_clip_multitask", 
            "skill_aware_clip2", 
            "skill_aware_clip_nsctxt",
            "skill_aware_clip_nobj",
        ]:
            try: 
                preds = [rec["answer"] for rec in json.load(open(f"./experiments/{evaluating_model}/formatted_pred.json"))]
            except FileNotFoundError: 
                try: preds = [rec["pred"] for rec in json.load(open(f"./experiments/{evaluating_model}/pred.json"))["preds"]]
                except FileNotFoundError:
                    preds = [rec["answer"] for rec in json.load(open(f"./result_{evaluating_model}.json"))]
            pred_embs_path = f"./ans_bertscore/{evaluating_model}_pred_embs.npy"
            assert len(val_refs) == len(preds), "length mismatch between predictions and references!"
            if not os.path.exists(pred_embs_path):
                dump_pred_embs_lookup(model, preds, pred_embs_path)
            pred_to_emb_lookup = np.load(pred_embs_path, allow_pickle=True)[()]
            scores = []
            for refs, answer in zip(val_refs, preds):
                ans_emb = pred_to_emb_lookup[answer]
                ref_embs = [] 
                for ref in refs: ref_embs.append(ans_to_emb_lookup[ref])
                ref_embs = np.stack(ref_embs)
                inst_bertscore = compute_inst_bertscore(ans_emb, ref_embs, thresh=thresh)
                scores.append(inst_bertscore)
            y[evaluating_model].append(round(100*np.mean(scores), 3))
            print(evaluating_model, round(100*np.mean(scores), 3))

    plt.clf()
    # plt.figure(figsize=(6,4))
    plt.title("Answer BERTSim Score vs Tolerance Threshold")
    plt.xlabel("threshold")
    plt.ylabel("Answer BERTSim Score")
    marker_map = {
        "t5": "o", 
        "deberta": "o", 
        "resnet": "o", 
        "clip": "o",
        "zero_shot_git_large": "^", 
        "clip_multimodal": "^", 
        "frozen_vilt2": "s",
        "frozen_git": "s",
        "skill_unaware_clip": ".", 
        "skill_aware_clip_multitask": ".", 
        "skill_aware_clip2": ".", 
        "skill_aware_clip_nsctxt": ".",
        "skill_aware_clip_nobj": ".",
    }
    model_name_map = {
        "t5": "FLAN T5", 
        "deberta": "DeBERTa", 
        "resnet": "ResNet", 
        "clip": "CLIP Unimodal",
        "zero_shot_git_large": "GIT zero-shot", 
        "clip_multimodal": "CLIP Multimodal", 
        "frozen_vilt2": "ViLT",
        "frozen_git": "GIT",
        "skill_unaware_clip": "FusionCLIP", 
        "skill_aware_clip_multitask": "SkillCLIP Multitask", 
        "skill_aware_clip2": "SkillCLIP", 
        "skill_aware_clip_nsctxt": "SkillCLIP (w/o scene texts)",
        "skill_aware_clip_nobj":   "SkillCLIP (w/o object tags)",
    }
    k = 1
    for name, ym in y.items(): 
        model_name = model_name_map[name]
        marker_style = marker_map[name]
        plt.plot(range(k*1, k*(len(thresholds)+1), k), ym, label=model_name, marker=marker_style)
    plt.xticks(range(k*1, k*(len(thresholds)+1), k), labels=thresholds)
    plt.legend(loc="upper right", fancybox=True, shadow=True, bbox_to_anchor=(2, 1))
    plt.tight_layout()
    plt.savefig("./ans_bertscore_plot.png")