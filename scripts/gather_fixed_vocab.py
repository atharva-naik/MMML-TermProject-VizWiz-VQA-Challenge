# code to gather fixed vocab of phrases/answers for classification formulation of VQA.
import os
import json
from collections import defaultdict

task_dir = os.path.join("data", "VQA")
for split in ['train', 'val']:
    data = json.load(open(os.path.join(task_dir, f"{split}.json")))
    vocab = defaultdict(lambda:0)
    for inst in data:
        for ans in inst["answers"]:
            vocab[ans["answer"]] += 1
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