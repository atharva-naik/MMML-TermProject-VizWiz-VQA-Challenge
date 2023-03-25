# code to gather fixed vocab of phrases/answers for classification formulation of VQA.
import os
import json
import numpy as np
from typing import *
from collections import defaultdict

def vote_answer(answers: List[dict], is_ansble: bool=True) -> Tuple[str, float, int]:
    answer_votes = defaultdict(lambda:0)
    get_ans_conf = {"yes": 1, "maybe": 0.5, "no": 0}
    tot = 0
    for ans in answers:
        if is_ansble and ans['answer'] == "unanswerable": continue 
        ans_conf = ans['answer_confidence']
        answer_votes[ans['answer']] += get_ans_conf[ans_conf]
        tot += get_ans_conf[ans_conf]
    i = np.argmax(list(answer_votes.values()))
    picked_ans = list(answer_votes.keys())[i]
    # picked_ans_conf = list(answer_votes.values())[i]/tot
    return picked_ans

task_dir = os.path.join("data", "VQA")
split_sizes = {}
vocab = defaultdict(lambda:0)
for split in ['train', 'val']:
    data = json.load(open(os.path.join(task_dir, f"{split}.json")))
    split_sizes[split] = len(data)
    for inst in data:
        answer_vote = defaultdict(lambda:0)
        # for ans in inst["answers"]:
        #     answer_vote[ans["answer"]] += 1
        #     # vocab[ans["answer"]] += 1
        # keys = list(answer_vote.keys())
        # values = list(answer_vote.values())
        # max_ind = np.argmax(values)
        # ans = keys[max_ind]
        ans = vote_answer(inst["answers"], is_ansble=inst["answerable"])
        vocab[ans] += 1
vocab = {k: v for k,v in sorted(vocab.items(), reverse=True, key=lambda x: x[1])}
# print(vocab)
print(len(vocab))

words = list(vocab.keys())
counts = list(vocab.values())
ctr = 0
tot = sum(counts)
MILESTONE = len(vocab)//10
for i in range(len(vocab)):
    ctr += counts[i]
    j = (i+1) // MILESTONE
    if (i+1) % MILESTONE == 0:
        print(j, j*MILESTONE, str(round(100*ctr/tot, 2))+"%")
print(split_sizes)
vocab_list = sorted(list(set(vocab.keys())))
label2id = {k: i for i,k in enumerate(vocab_list)}
print(len(vocab_list))
with open("experiments/vizwiz_class_vocab.json", "w") as f:
    json.dump(label2id, f, indent=4)