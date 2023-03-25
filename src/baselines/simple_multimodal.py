# simple multimodal baselines.
# Author: Atharva Naik
# code to train and predict with simple multimodal baselines.

import os
import torch
import requests
from typing import *
from PIL import Image
from tqdm import tqdm
from src.PythonHelperTools.vqaTools.vqa import VQA
from transformers import ViltProcessor, ViltForQuestionAnswering

# sanity checking/testing functions:
def test_vilt_predict(image_path: str, query: str, 
                      processor: ViltProcessor, model,
                      move_to_cuda: bool=False) -> str:
    image = Image.open(image_path)
    # prepare inputs (tokenization basically)
    encoding = processor(image, query, return_tensors="pt")
    if move_to_cuda:
        # print("moving encoding to cuda")
        for k,v in encoding.items():
            encoding[k] = v.cuda()
    outputs = model(**encoding)
    logits = outputs.logits.detach().cpu()
    idx = logits.argmax(-1).item() # predict answer ids

    return model.config.id2label[idx]
    # print("Predicted answer:", )

def test_vilt(split: str="train", use_cuda: bool=True) -> List[str]:
    # initialize ViLT and its processor/tokenizer.
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # load the dataset in memory
    data_path = f"./data/VQA/{split}.json"
    data = VQA(annotation_file=data_path).getAnns()
    preds = []
    break_at = 5
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
    preds = test_vilt()
    print(preds)