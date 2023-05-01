# Author: Nava 
import os
import pdb
import json
import random
import argparse
import numpy as np
from typing import *
import shutil

import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import GitForCausalLM, GitProcessor
from src.datautils import VizWizVQABestAnsDataset

# seed everything
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

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
                 label2id: dict={}, train:bool = True, partial_loss:bool = False, **kwargs):
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
        self.partial_loss = partial_loss
        self.train = train

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        image = Image.open(item["image"])
        img_path = item["image"]
        question = item["question"]
        aux_tokens = self.aux_tokens_data[i]
        if len(aux_tokens["objects"]) > 50: 
            aux_tokens["objects"] = aux_tokens["objects"][:50]
        if len(aux_tokens["ocr"]) > 50:
            aux_tokens["ocr"] = aux_tokens["ocr"][:50]
        objects = ", ".join(aux_tokens["objects"][0])
        scene_text = ", ".join(aux_tokens["ocr"][0])
        skills = [self.skill_to_text[skill] for skill in aux_tokens["skills"]]
        skills = ", ".join(skills)
        if not self.questions_last:
            text = f"Objects: {objects}; Scene Text: {scene_text}; Question: {question}; Skills Needed: {skills}; Answer: {item['answer']}"
        else: 
            text = f"Skills Needed: {skills}; Objects: {objects}; Scene Text: {scene_text};  Question: {question}; Answer: {item['answer']} "
        
        if self.train: 
            encoding = self.processor(images=image, text=text, padding="max_length", return_tensors="pt", 
                                            max_length=200, truncation=True)

            # remove batch dimension

            if self.partial_loss:
                encoding["labels"] = encoding["input_ids"].clone()
                ans_ix = (encoding["labels"].squeeze() == 3437).nonzero().squeeze()
                encoding["labels"][:, :ans_ix] = -100
                encoding["labels"][encoding["input_ids"] == 0] = -100
            else: 
                encoding["labels"] = encoding["input_ids"].clone()
                encoding["labels"][encoding["input_ids"] == 0] = -100
                
            encoding = {k:v.squeeze() for k,v in encoding.items()}
            return encoding
        else: 
            return image, text, img_path
        

def train_git(args):

    # load model
    model = GitForCausalLM.from_pretrained(args.model_path)
    model.to(args.device)
    model.train()
    epochs = args.epochs
    device = args.device

    # load data and dataloader, initialize optimizer
    #TODO: Make sure to switch back the train to the actual train dataset
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = SkillAwareGitVizWizVQADataset(annot_path="./data/VQA/train.json", images_dir="./data/VQA/train",
         aux_tokens_data="./data/VQA/o365/train_aux_info_tokens.jsonl", questions_last=args.questions_last,
         partial_loss=args.partial_loss, model_path=args.model_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset_encoded = SkillAwareGitVizWizVQADataset(annot_path="./data/VQA/val.json", images_dir="./data/VQA/val",
            aux_tokens_data="./data/VQA/o365/val_aux_info_tokens.jsonl", questions_last=args.questions_last,
                     partial_loss=args.partial_loss, model_path=args.model_path)
    val_loader = DataLoader(val_dataset_encoded, batch_size=args.batch_size)

    val_dataset_not_encoded = SkillAwareGitVizWizVQADataset(annot_path="./data/VQA/val.json", images_dir="./data/VQA/val",
            aux_tokens_data="./data/VQA/o365/val_aux_info_tokens.jsonl", questions_last=args.questions_last,
                     partial_loss=args.partial_loss, model_path=args.model_path, train=False)

    # best val acc, step & epoch
    best_val_loss = 1000
    best_val_step = 0
    best_val_epoch = 0

    # all the experimental folder paths and create the experiments dir.
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    model_save_dir = os.path.join("experiments", args.exp_name, "best_model")
    os.makedirs(model_save_dir, exist_ok=True)

    train_log_path = os.path.join("experiments", args.exp_name, "train_log_epoch_{}.jsonl")
    val_log_path = os.path.join("experiments", args.exp_name, "all_val_logs.jsonl")
    config_path = os.path.join(exp_dir, "finetune_config.json")

    # reset the val logs.
    open(val_log_path, "w")
    # save the finetuning config.
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    # train loop
    for epoch in range(epochs):
         # reset the train log.
        open(train_log_path.format(epoch), "w") 
        train_bar = tqdm(enumerate(train_loader), 
                         total=len(train_loader), 
                         desc="commencing training")
        train_epoch_losses = []
        for step, batch in train_bar:
            batch = {k:v.to(args.device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_epoch_losses.append(loss.item())
            logits = outputs.logits.detach().cpu()
            train_bar.set_description(f"T: {epoch+1}/{epochs}: L:{logits.shape} bl: {loss.item():.3f} l: {np.mean(train_epoch_losses):.3f}")
            
            if step % 250 == 0:
                image, text, _ = val_dataset_not_encoded[1]
                text = text.split("Answer: ")[0] + "Answer: "
                pixel_values = val_dataset_encoded.processor(images=image, return_tensors="pt").pixel_values.to(device)
                input_ids = val_dataset_encoded.processor(text=text, return_tensors="pt").input_ids.to(device)
                gen_ids = model.generate(
                    pixel_values=pixel_values, 
                    input_ids=input_ids, 
                    max_new_tokens=100,
                    min_new_tokens=1,
                    num_beams=2,
                    early_stopping=True,)

                input_text = val_dataset_encoded.processor.batch_decode(input_ids, skip_special_tokens=True)[0]
                generated_text = val_dataset_encoded.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                ans_text = generated_text[len(input_text):].strip()
                print("\x1b[34;1mgeneration sanity check (in):\x1b[0m", input_text)
                print("\x1b[34;1mgeneration sanity check (out):\x1b[0m", ans_text)

            # log stats.
            if (step+1) % args.log_steps == 0 or (step+1) == len(train_loader):
                with open(train_log_path.format(epoch), "a") as f:
                    f.write(json.dumps({"step": step, "loss": np.mean(train_epoch_losses)})+"\n")

            # check if evalulation should be done:
            if (step+1) % args.eval_steps == 0 or (step+1) == len(train_loader):
                val_loss = validate_git(
                    model=model, dataloader=val_loader, 
                    eval_step=step, epoch=epoch, args=args,
                )
                val_acc = val_git_acc(val_dataset_not_encoded, model, device)
                # save the best model.
                if val_loss < best_val_loss:
                    best_val_step = step
                    best_val_epoch = epoch
                    best_val_loss = val_loss
                    print("Deleting the previous best model.")
                    shutil.rmtree(model_save_dir)
                    os.makedirs(model_save_dir, exist_ok=True)
                    # save the checkpoint for model/optimizer
                    print("Saving the new best model.")
                    model.save_pretrained(model_save_dir)
                with open(val_log_path, "a") as f:
                    f.write(json.dumps({
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "epoch": epoch, "step": step,
                        "best_val_epoch": best_val_epoch,
                        "best_val_step": best_val_step,
                        "best_val_loss": best_val_loss,
                    })+"\n")

            model.train()



def validate_git(model, dataloader: DataLoader, eval_step: int, 
                 epoch: int, args: argparse.Namespace) -> float:
    val_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="validating")
    val_losses = []
    model.eval()
    for step, batch in val_bar:
        model.zero_grad()
        with torch.no_grad():
            batch = {k:v.to(args.device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits.detach().cpu()

            val_losses.append(loss.item())
            val_bar.set_description(f"V: {epoch+1}/{args.epochs} ({eval_step}) bl: {loss.item():.3f} l: {np.mean(val_losses):.3f}")

    return np.mean(val_losses)

def val_git_acc(val_dataset, model, device, num_examples = 100, batch_size=2):
    val_subset = torch.utils.data.Subset(val_dataset, range(num_examples))
    model.eval()
    val_bar = tqdm(enumerate(val_subset), total=len(val_subset), desc="getting val accuracy")
    preds = []
    answers = []
    for step, batch in val_bar:
        image, full_text, _ = batch
        text = full_text.split("Answer: ")[0] + "Answer: " 
        answer = full_text.split("Answer: ")[1].strip()
        pixel_values = val_dataset.processor(images=image, return_tensors="pt").pixel_values.to(device)
        input_ids = val_dataset.processor(text=text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=100,
                min_new_tokens=1,
                num_beams=2,
                early_stopping=True,
            )
            input_text = val_dataset.processor.batch_decode(input_ids, skip_special_tokens=True)[0]
            generated_text = val_dataset.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            ans_text = generated_text[len(input_text):].strip()
            preds.append(ans_text)
            answers.append(answer)

    acc_vector = [1 if pred == ans else 0 for pred, ans in zip(preds, answers)]
    return np.mean(acc_vector)

def predict_git(args):
    device = args.device
    load_path = os.path.join("experiments", args.exp_name, "best_model")
    result_path = os.path.join("experiments", args.exp_name, "results.json")
    verbose_result_path = os.path.join("experiments", args.exp_name, "verbose_results.json")
    model = GitForCausalLM.from_pretrained(load_path).to(args.device)
    model.eval()
    val_dataset = SkillAwareGitVizWizVQADataset(annot_path="./data/VQA/val.json", images_dir="./data/VQA/val",
            aux_tokens_data="./data/VQA/o365/val_aux_info_tokens.jsonl", questions_last=True,
                     partial_loss=args.partial_loss, model_path=args.model_path, train=False)
    val_bar = tqdm(enumerate(val_dataset), total=len(val_dataset), desc="predicting on val")
    results = []
    verbose_results = []
    for step, batch in val_bar:
        image, full_text, img_path = batch
        text = full_text.split("Answer: ")[0] + "Answer: " 
        question = text.split("Question: ")[1].split("Answer: ")[0].strip()
        answer = full_text.split("Answer: ")[1].strip()
        pixel_values = val_dataset.processor(images=image, return_tensors="pt").pixel_values.to(device)
        input_ids = val_dataset.processor(text=text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=100,
                min_new_tokens=1,
                num_beams=2,
                early_stopping=True,
            )
            input_text = val_dataset.processor.batch_decode(input_ids, skip_special_tokens=True)[0]
            generated_text = val_dataset.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            ans_text = generated_text[len(input_text):].strip()
            result = {"image": img_path, "answer": ans_text}
            verbose_result = {"image": img_path, "question": question, "pred_answer": ans_text, "gt_answer": answer}
            results.append(result)
            verbose_results.append(verbose_result)
        if step == 0:
            break

    with open(result_path, "w") as f: 
        json.dump(results, f)
    with open(verbose_result_path, "w") as f:
        json.dump(verbose_results, f)
    

def get_args():
    """get terminal arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="./data/VQA", type=str, help="path to the VQA dataset")
    parser.add_argument("-mp", "--model_path", default="microsoft/git-base-vqav2", type=str, help="name/path of the HF checkpoint")
    parser.add_argument("-t", "--train", action="store_true", help="finetune ViLT model")
    parser.add_argument("-p", "--predict", action="store_true", help="generate predictions for data with labels")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-ls", "--log_steps", default=20, type=int, help="number of steps after which train stats are logged")
    parser.add_argument("-es", "--eval_steps", default=5000, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-exp", "--exp_name", type=str, required=True, help="name of the experiment")
    parser.add_argument("-de", "--device", default="cuda:0", type=str, help="device to be used for training, defaults to cuda:0")
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="batch size to be used for training")
    parser.add_argument("-ckpt", "--checkpoint_path", type=Union[str, None], default=None, help="checkpoint to be loaded")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="learning rate for training")
    parser.add_argument("-q", "--questions_last", action="store_true", help="Questions are last in the input sequence")
    parser.add_argument("-pl", "--partial_loss", action="store_true", help="Only Take LM Loss on the Answers")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    if args.train: train_git(args)
    if args.predict: predict_git(args)