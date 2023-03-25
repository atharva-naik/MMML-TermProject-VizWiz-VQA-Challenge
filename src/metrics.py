# Author: Navaneethan Vaikunthan, Atharva Naik
# common metrics for the task.

import json
# hf metrics library
import evaluate 
import numpy as np
from typing import *
from collections import defaultdict

# classification based setup metrics:
def get_class_preds(outputs, id2label):
    preds = outputs.logits.argmax(dim=-1)
    pred_words = [id2label[pred.cpu().item()] for pred in preds]
    return pred_words

def proxy_accuracy(inputs, generations, tokenizer):
    decoded_preds = tokenizer.batch_decode(generations, skip_special_tokens=True)
    answers = inputs["answer"]
    for i, answer in enumerate(answers):
        print(f"Question: {inputs['question'][i]}")
        print(f"Prediction: {decoded_preds[i]}")
        print(f"Answer: {answer}")

def class_accuracy(inputs, outputs, id2label):

    pred_words = get_class_preds(outputs, id2label)
    print(pred_words)
    print(inputs["answer"])
    # print((pred_words == inputs["answer"]).sum() / len(pred_words))

# generative setting metrics:
# add BLEU and stuff (as a supplement to the standard VQA task metric (accuracy measure))

bleu_scorer = evaluate.load("bleu")
# [{'answer': 'unanswerable', 'answer_confidence': 'yes'}, {'answer': 'unanswerable', 'answer_confidence': 'yes'}, {'answer': 'unanswerable', 'answer_confidence': 'yes'}, {'answer': 'unanswerable', 'answer_confidence': 'yes'}, {'answer': 'unanswerable', 'answer_confidence': 'maybe'}, {'answer': 'unanswerable', 'answer_confidence': 'yes'}, {'answer': 'unanswerable', 'answer_confidence': 'yes'}, {'answer': 'unanswerable', 'answer_confidence': 'no'}, {'answer': 'cannot repair this computer automatically', 'answer_confidence': 'maybe'}, {'answer': 'blank screen', 'answer_confidence': 'yes'}]
def compute_bleu(preds: List[str], ans_list: List[List[str]], strategy: str="best") -> Dict[str, Any]:
    global bleu_scorer
    refs = []
    assert strategy in ["best", "all"], "unsupported strategy"
    for anns in ans_list:
        if strategy == "best":
            vote_dict = defaultdict(lambda:0)
            for ans in anns:
                vote_dict[ans] += 1
            keys = list(vote_dict.keys())
            values = list(vote_dict.values())
            max_ind = np.argmax(values)
            best_ans = keys[max_ind] 
            refs.append([best_ans])
        elif strategy == "all":
            gold = sorted(list(set(anns)))
            refs.append(gold)
    score = bleu_scorer.compute(predictions=preds, references=refs)

    return score

def compute_bleu_from_zero_shot_vilt(preds_file: str):
    data = json.load(open(preds_file))
    preds = [rec["pred"] for rec in data]
    ans_list = [[a['answer'] for a in rec["trues"]] for rec in data]
    # compute BLEU score with 2 different aggregation strategies:
    best_bleu = compute_bleu(preds, ans_list, "best")
    print(f"best_bleu: {best_bleu}")
    all_bleu = compute_bleu(preds, ans_list, "all")
    print(f"all_bleu: {all_bleu}")