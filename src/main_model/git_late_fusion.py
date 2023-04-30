# Author: Nava 
import os
import pdb
import json
import random
import argparse
import numpy as np
from typing import *

import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import GitForCausalLM, GitProcessor
from src.datautils import VizWizVQABestAnsDataset

def read_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "": continue
            data.append(json.loads(line))

    return data

class SkillAwareGitVizWizVQADataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, aux_tokens_data: str="", questions_last: bool=False,
                 model_path: str="microsoft/git-base", 
                 label2id: dict={}, train:bool = True,  **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.processor = GitProcessor.from_pretrained(model_path)
        self.label2id = label2id
        assert aux_tokens_data != "", "missing path to skill labels"
        self.aux_tokens_data = read_jsonl(aux_tokens_data)
        self.skill_to_text = {
            "TXT": "Text Recognition",
            "OBJ": "Object Detection",
            "COL": "Color Recognition",
            "CNT": "Counting",
            "OTH": "Other",
        }
        self.questions_last = questions_last
        self.train = train

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        image = Image.open(item["image"])
        question = item["question"]
        aux_tokens = self.aux_tokens_data[i]
        objects = ", ".join(aux_tokens["objects"][0])
        scene_text = ", ".join(aux_tokens["ocr"][0])
        skills = [self.skill_to_text[skill] for skill in aux_tokens["skills"]]
        skills = ", ".join(skills)
        if not self.questions_last:
            text = f"Objects {objects} Scene Text {scene_text} Question {question} Skills Needed: {skills} Answer: {item['answer']}"
        else: 
            text = f"Skills Needed: {skills} Objects {objects} Scene Text {scene_text}  Question {question} Answer: {item['answer']} "
        
        if self.train: 
            encoding = self.processor(images=image, text=text, padding="max_length", return_tensors="pt", 
                                            max_length=200)
            # remove batch dimension

            encoding = {k:v.squeeze() for k,v in encoding.items()}
            return encoding
        else: 
            return image, text
        

def train_git(args):

    # load model
    model = GitForCausalLM.from_pretrained(args.model_path)
    model.to(args.device)
    model.train()

    # load data and dataloader, initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataset = SkillAwareGitVizWizVQADataset(annot_path="./data/VQA/train.json", images_dir="./data/VQA/train",
         aux_tokens_data="./data/VQA/o365/train_aux_info_tokens.jsonl", questions_last=args.questions_last)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = SkillAwareGitVizWizVQADataset(annot_path="./data/VQA/val.json", images_dir="./data/VQA/val",
            aux_tokens_data="./data/VQA/o365/val_aux_info_tokens.jsonl", questions_last=args.questions_last)    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # best val acc, step & epoch
    best_val_loss = 1000
    best_val_step = 0
    best_val_epoch = 0

    # all the experimental folder paths and create the experiments dir.
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    model_save_path = os.path.join("experiments", args.exp_name, "best_model.pt")
    train_log_path = os.path.join("experiments", args.exp_name, "train_log_epoch_{}.jsonl")
    val_log_path = os.path.join("experiments", args.exp_name, "all_val_logs.jsonl")
    config_path = os.path.join(exp_dir, "finetune_config.json")

    # reset the val logs.
    open(val_log_path, "w")
    # save the finetuning config.
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    # train loop
    for epoch in range(args.epochs):
         # reset the train log.
        open(train_log_path.format(epoch), "w") 
        train_bar = tqdm(enumerate(train_loader), 
                         total=len(train_loader), 
                         desc="commencing training")
        train_epoch_losses = []
        for step, batch in train_bar:
            batch = {k:v.to(args.device) for k,v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()



def val_git_acc(val_dataset, model, device, batch_size=2):
    pass

def get_args():
    """get terminal arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="./data/VQA", type=str, help="path to the VQA dataset")
    parser.add_argument("-mp", "--model_path", default="microsoft/git-base-vqav2", type=str, help="name/path of the HF checkpoint")
    parser.add_argument("-t", "--train", action="store_true", help="finetune ViLT model")
    parser.add_argument("-p", "--predict", action="store_true", help="generate predictions for data with labels")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-ls", "--log_steps", default=10, type=int, help="number of steps after which train stats are logged")
    parser.add_argument("-es", "--eval_steps", default=4000, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-exp", "--exp_name", type=str, required=True, help="name of the experiment")
    parser.add_argument("-de", "--device", default="cuda:0", type=str, help="device to be used for training, defaults to cuda:0")
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="batch size to be used for training")
    parser.add_argument("-ckpt", "--checkpoint_path", type=Union[str, None], default=None, help="checkpoint to be loaded")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="learning rate for training")
    parser.add_argument("-q", "--questions_last", action="store_true", help="Questions are last in the input sequence")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    if args.train: train_git(args)
    processor = GitProcessor.from_pretrained("microsoft/git-base-vqav2")
    tokenizer = processor.tokenizer
    answer_id = tokenizer.convert_tokens_to_ids("answer")
    print(answer_id)