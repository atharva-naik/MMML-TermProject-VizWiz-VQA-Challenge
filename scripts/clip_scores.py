import json
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datautils import VizWizVQABestAnsDataset

"""Test CLIP L-14 in a zero shot way for this task
This version ignores the question text"""
from sentence_transformers import SentenceTransformer, util

# Load CLIP model
model = SentenceTransformer('clip-ViT-L-14')
# technically not needed but I'll do it anyways
if torch.cuda.is_available(): model.cuda()

dataset = VizWizVQABestAnsDataset(
    "./data/VQA/val.json", 
    "./data/VQA/val",
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

all_classes = json.load(open("experiments/vizwiz_class_vocab.json"))
inv_classes = {v: k for k,v in all_classes.items()}
all_answer_texts = list(all_classes.keys())
ans_emb = model.encode(all_answer_texts, show_progress_bar=True, convert_to_tensor=True)

# Encode all images:
img_emb = []
true_indices = []
for batch in tqdm(dataloader):
    true_indices += [all_classes[ans] for ans in batch["answer"]]
    img_emb.append(model.encode([Image.open(img) for img in batch["image"]], convert_to_tensor=True))
img_emb_list = []
for x in img_emb:
    img_emb_list += x.tolist()
img_emb = torch.as_tensor(img_emb_list)
img_emb = img_emb.to(ans_emb.device)
# compute cosine similarities 
cos_scores = util.cos_sim(img_emb, ans_emb)
ans_inds = cos_scores.cpu().max(axis=1).indices.tolist()
# answers = [inv_classes[i.item()] for i in ans_inds]
print((torch.as_tensor(true_indices) == torch.as_tensor(ans_inds)).sum().item())

with open("experiments/CLIP_zero_shot_ans.json") as f:
    json.dump(answers, indent=4, )