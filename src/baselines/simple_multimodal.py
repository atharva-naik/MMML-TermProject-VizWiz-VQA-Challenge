# simple multimodal baselines.
# Author: Atharva Naik
# code to train and predict with simple multimodal baselines.

import os
import json
import torch
import requests
from typing import *
from PIL import Image
from tqdm import tqdm
from src.PythonHelperTools.vqaTools.vqa import VQA
from src.datautils import VizWizVQABestAnsDataset
from transformers import ViltProcessor, ViltForQuestionAnswering

# there's a bug in HF's ViLT code:
# paste the lines below in `transformers/models/vilt/modeling_vilt.py`
#         # print(select.device)
#         # print("patch_index:", patch_index.device)
#         # print("x:", x.device)
#         # print("pos_embed:", pos_embed.device)
#         # print("select:", select.device)

#         # there is an issue with patch_index:
#         # FIX: makes sure the select indices and the patch index are on the same device.
#         patch_index = patch_index.to(select.device)

# modify the best answer dataset class for ViLT

# NOTE: ViLT doesn't have more than 40 position embeddings it seems (max seq len for anything can't be more than 40)

class ViLTVizWizVQABestAnsDataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, a_args: dict={"padding": "max_length", "max_length": 40, "truncation": True}, 
                 q_args: dict={"padding": "max_length", "max_length": 40, "truncation": True},
                 model_path: str="dandelin/vilt-b32-finetuned-vqa", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.a_args = a_args
        self.q_args = q_args
        # intialize processor for ViLT:
        self.processor = ViltProcessor.from_pretrained(model_path)

    def __getitem__(self, i: int):
        item = super(ViLTVizWizVQABestAnsDataset, self).__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"])
        question = item["question"]
        encoding = self.processor(image, question, return_tensors="pt", **self.q_args)
        answer = self.processor.tokenizer(item["answer"], return_tensors="pt", **self.a_args)
        # encode the answer label too.
        encoding["label_input_ids"] = answer["input_ids"]
        encoding["label_attn_mask"] = answer["attention_mask"]
        encoding["label_tok_type_ids"] = answer["token_type_ids"]
        
        return encoding

# sanity checking/testing functions:
def test_vilt_predict(image_path: str, query: str, 
                      processor: ViltProcessor, model,
                      move_to_cuda: bool=False) -> str:
    image = Image.open(image_path)
    # prepare inputs (tokenization basically)
    encoding = processor(
        image, query, return_tensors="pt",
        truncation=True, max_length=40,
        padding="max_length",
    )
    if move_to_cuda:
        # print("moving encoding to cuda")
        for k,v in encoding.items():
            encoding[k] = v.cuda()
    outputs = model(**encoding)
    logits = outputs.logits.detach().cpu()
    idx = logits.argmax(-1).item() # predict answer ids

    return model.config.id2label[idx]
    # print("Predicted answer:", )

def test_vilt(split: str="train", use_cuda: bool=True, break_at: int=5) -> List[str]:
    # initialize ViLT and its processor/tokenizer.
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # load the dataset in memory
    data_path = f"./data/VQA/{split}.json"
    data = VQA(annotation_file=data_path).getAnns()
    preds = []
    move_to_cuda = False
    if use_cuda and torch.cuda.is_available():
        model.cuda()
        move_to_cuda = True
    for i, rec in tqdm(enumerate(data), total=len(data)):
        image_path = os.path.join("./data/VQA", split, rec["image"])
        query = rec["question"]
        pred = test_vilt_predict(image_path, query, processor, 
                                 model, move_to_cuda=move_to_cuda)
        preds.append({"question": query, "trues": rec["answers"], "pred": pred})
        if (i+1) == break_at: break

    return preds

# main
if __name__ == "__main__":
    # sanity checking ViLT (assessing zero-shot performance)
    split = "val"
    zero_shot_preds = test_vilt(split=split, break_at=100000)
    with open(f"experiments/{split}_zero_shot_vilt_vqa2_preds.json", "w") as f:
        json.dump(zero_shot_preds, f, indent=4)