# compute answer similarity using sentence BERT between answer
import os
import json
import numpy as np
from typing import *
from tqdm import tqdm
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

def compute_inst_bertscore(ans_emb: np.ndarray, ref_embs: List[np.ndarray]):
    scores = util.cos_sim(ans_emb, ref_embs)[0]
    return scores.max()
# return scores[np.argpartition(scores, -3)[-3:]].mean()
# main
if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    val_refs = [[d["answer"] for d in rec["answers"]] for rec in json.load(open("./data/VQA/val.json"))]
    if not os.path.exists(ANS_EMBS_PATH): dump_ref_embs_lookup(model, val_refs)
    ans_to_emb_lookup = np.load(ANS_EMBS_PATH, allow_pickle=True)[()]

    for evaluating_model in [
        "frozen_git",
        "clip", 
        "clip_multimodal",
        "frozen_vilt2",
        "resnet",
        "skill_unaware_clip",
        "skill_aware_clip2",
        # "skill_aware_clip3",
        # "skill_clip_hf",
        # "skill_clip_hf_vis_proj",
        # "skill_clip_hf_vis_proj2",
    ]:
        try: preds = [rec["answer"] for rec in json.load(open(f"./experiments/{evaluating_model}/formatted_pred.json"))]
        except FileNotFoundError: preds = [rec["pred"] for rec in json.load(open(f"./experiments/{evaluating_model}/pred.json"))["preds"]]
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
            inst_bertscore = compute_inst_bertscore(ans_emb, ref_embs)
            scores.append(inst_bertscore)
        print(evaluating_model, round(100*np.mean(scores), 3))