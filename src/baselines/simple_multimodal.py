# simple multimodal baselines.
# Author: Atharva Naik, Yash Butala
# code to train and predict with simple multimodal baselines.

import os
import clip
import json
import torch
import random
import argparse
# import requests
import numpy as np
from typing import *
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.PythonHelperTools.vqaTools.vqa import VQA
from src.datautils import VizWizVQABestAnsDataset
from transformers import ViltProcessor, ViltForQuestionAnswering

# seed everything
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

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

class ViLTDataLoader(DataLoader):
    def __init__(self, dataset, split: str="train", **kwargs):
        self.split = split
        if split == "train": 
            super().__init__(dataset, shuffle=True, 
                             collate_fn=self.custom_collate_fn, **kwargs)
        else: super().__init__(dataset, shuffle=False, 
                               collate_fn=self.custom_collate_fn, **kwargs)
        self.processor = dataset.processor

    def custom_collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        if self.split != "test":
            labels = [item['labels'] for item in batch]
        # create padded pixel values and corresponding pixel mask
        encoding = self.processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        # create new batch
        batch = {}
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['token_type_ids'] = torch.stack(token_type_ids)
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        if self.split != "test":
            batch['labels'] = torch.stack(labels)
        
        return batch

class FrozenViLTVizWizVQA(nn.Module):
    def __init__(self, model_path: str="dandelin/vilt-b32-finetuned-vqa",
                 device: str="cuda:0", label2id: dict={}, emb_size: int=768):
        super().__init__()
        self.model_path = model_path
        self.model = ViltForQuestionAnswering.from_pretrained(model_path)
        self.num_classes = len(label2id)
        self.emb_size = emb_size
        self.upscale_size = self.num_classes // 2
        self.model.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.emb_size, 
                out_features=self.upscale_size, 
                bias=True
            ),
            nn.LayerNorm(
                (self.upscale_size,), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                in_features=self.upscale_size, 
                out_features=self.num_classes, 
                bias=True
            ),
        )
        # freeze ViLT params:
        for param in self.model.vilt.parameters():
            param.requires_grad = False
        self.label2id = label2id
        self.id2label = {v: k for k,v in self.label2id.items()}
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(device)
        print(f"moving model to device: {device}")

    def parameters(self):
        return self.model.classifier.parameters()

    def train(self):
        self.model.classifier.train()

    def eval(self):
        self.model.classifier.eval()

    def forward(self, **encoding):
        """return outputs and loss"""
        label = encoding.get("labels")
        encoding = {k: v for k,v in encoding.items() if k != "labels"}
        # print(encoding.keys())
        outputs = self.model(**encoding)
        loss = None
        if label is not None: 
            loss = self.loss_fn(outputs.logits, label)
            return outputs, loss
        return outputs

class ViLTVizWizVQABestAnsDataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, 
                 # a_args: dict={"padding": "max_length", "max_length": 40, "truncation": True}, 
                 #  q_args: dict={"padding": "max_length", "max_length": 40, "truncation": True},
                 model_path: str="dandelin/vilt-b32-finetuned-vqa", 
                 split: str="train", label2id: dict={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        self.model_path = model_path
        # self.q_args = q_args
        self.label2id = label2id
        # intialize processor for ViLT:
        self.processor = ViltProcessor.from_pretrained(model_path)

    def __getitem__(self, i: int):
        item = super(ViLTVizWizVQABestAnsDataset, self).__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"])
        question = item["question"]
        encoding = self.processor(image, question, padding="max_length", 
                                  truncation=True, return_tensors="pt")
        for key in encoding: encoding[key] = encoding[key][0]
        if self.split != "test":
            encoding["labels"] = torch.as_tensor(self.label2id.get(item["answer"]))
        # answer = self.processor.tokenizer(item["answer"], return_tensors="pt", **self.a_args)
        # encode the answer label too.
        # encoding["label_input_ids"] = answer["input_ids"]
        # encoding["label_attn_mask"] = answer["attention_mask"]
        # encoding["label_tok_type_ids"] = answer["token_type_ids"]
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

def save_current_model(model, optimizer, epoch: int, step: int, save_path: str):
    """save the current model and optimizer state"""
    ckpt_dict = {}
    ckpt_dict["model_state_dict"] = model.state_dict()
    ckpt_dict["optim_state_dict"] = optimizer.state_dict()
    ckpt_dict["epoch"] = epoch
    ckpt_dict["step"] = step
    torch.save(ckpt_dict, save_path)

def validate_vilt(model, dataloader: DataLoader, eval_step: int, 
                  epoch: int, args: argparse.Namespace, 
                  output_preds: bool=False) -> Union[Tuple[float, float], Tuple[List[dict], float, float]]:
    val_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="validating")
    tot, matches, val_losses = 0, 0, []
    val_preds = []
    model.eval()
    for step, batch in val_bar:
        model.zero_grad()
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].to(args.device)
            outputs, loss = model(**batch)
            preds = outputs.logits.argmax(axis=-1).detach().cpu()
            labels = batch["labels"].detach().cpu()
            batch_matches = (preds == labels).sum().item() 
            batch_tot = len(preds)
            matches += batch_matches
            tot += batch_tot
            acc = matches/tot
            if output_preds:
                for pred, label in zip(preds.tolist(), labels.tolist()):
                    val_preds.append({
                        "pred": model.id2label[pred],
                        "true": model.id2label[label],
                    })
            val_losses.append(loss.item())
            val_bar.set_description(f"V: {epoch+1}/{args.epochs} ({eval_step}): a: {100*acc:.2f} ba: {(100*batch_matches/batch_tot):.2f} bl: {loss.item():.3f} l: {np.mean(val_losses):.3f}")
    if output_preds: 
        return val_preds, np.mean(val_losses), matches/tot
    else: return np.mean(val_losses), matches/tot

def testing_vilt(model, dataloader: DataLoader, args: argparse.Namespace, test_ids=[]) -> Union[Tuple[float, float], Tuple[List[dict], float, float]]:
    test_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="testing")
    test_preds = []
    all_preds = []
    model.eval()
    for step, batch in test_bar:
        model.zero_grad()
        with torch.no_grad():
            for key in batch: batch[key] = batch[key].to(args.device)
            outputs = model(**batch)
            preds = outputs.logits.argmax(axis=-1).detach().cpu()
            for pred in preds.tolist(): all_preds.append(model.id2label[pred])
    for pred, image_id, in zip(all_preds, test_ids): 
        test_preds.append({
            "image": image_id,
            "answer": pred,
        })
    
    return test_preds

def test_unseen_vilt(args):
    data_dir = args.data_dir # default: ./data/VQA
    model_path = args.model_path # model_path: dandelin/vilt-b32-finetuned-vqa
    device = args.device
    test_images = os.path.join(data_dir, "test")
    test_annot = os.path.join(data_dir, "test.json") 
    test_ids = [i["image"] for i in json.load(open(test_annot))]
    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    # declare model.
    model = FrozenViLTVizWizVQA(
        model_path=model_path, 
        label2id=label2id, 
        device=device,
    )
    # all the experimental folder paths and create the experiments dir.
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # restore from checkpoint.
    if args.checkpoint_path is None:
        # load the `best_model.pt` checkpoint.
        checkpoint_path = os.path.join(exp_dir, "best_model.pt")
    else: checkpoint_path = args.checkpoint_path
    ckpt_dict = torch.load(
        checkpoint_path, 
        map_location=model.device
    )
    model.load_state_dict(ckpt_dict["model_state_dict"])
    # optimizer.load_state_dict(ckpt_dict["optim_state_dict"])
    batch_size = args.batch_size
    data = ViLTVizWizVQABestAnsDataset(
        test_annot, test_images, label2id=label2id,
        model_path=model_path, split="test",
    )
    data_loader = ViLTDataLoader(dataset=data, batch_size=batch_size, split="test")
    test_log_path = os.path.join("experiments", args.exp_name, "test_logs.json")
    
    model.eval()
    test_preds = testing_vilt(
        model=model, dataloader=data_loader, 
        args=args, test_ids=test_ids,
    )
    with open(test_log_path, "w") as f:
        json.dump(test_preds, f, indent=4)

def predict_vilt(args):
    data_dir = args.data_dir # default: ./data/VQA
    model_path = args.model_path # model_path: dandelin/vilt-b32-finetuned-vqa
    device = args.device
    val_images = os.path.join(data_dir, "val")
    val_annot = os.path.join(data_dir, "val.json") 
    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    # declare model.
    model = FrozenViLTVizWizVQA(
        model_path=model_path, 
        label2id=label2id, 
        device=device,
    )
    # all the experimental folder paths and create the experiments dir.
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # restore from checkpoint.
    if args.checkpoint_path is None:
        # load the `best_model.pt` checkpoint.
        checkpoint_path = os.path.join(exp_dir, "best_model.pt")
    else: checkpoint_path = args.checkpoint_path
    ckpt_dict = torch.load(
        checkpoint_path, 
        map_location=model.device
    )
    model.load_state_dict(ckpt_dict["model_state_dict"])
    # optimizer.load_state_dict(ckpt_dict["optim_state_dict"])
    batch_size = args.batch_size
    data = ViLTVizWizVQABestAnsDataset(
        val_annot, val_images,
        label2id=label2id,
        model_path=model_path, 
    )
    data_loader = ViLTDataLoader(dataset=data, batch_size=batch_size, split="val")
    predict_log_path = os.path.join("experiments", args.exp_name, "predict_logs.json")
    
    model.eval()
    val_preds, val_loss, val_acc = validate_vilt(
        model=model, dataloader=data_loader, 
        eval_step=0, epoch=0, args=args,
        output_preds=True,
    )
    with open(predict_log_path, "a") as f:
        json.dump({
            "predict_acc": val_acc, 
            "predict_loss": val_loss,
            "preds": val_preds,
        }, f)

def finetune_vilt(args):
    data_dir = args.data_dir # default: ./data/VQA
    model_path = args.model_path # model_path: dandelin/vilt-b32-finetuned-vqa
    device = args.device
    train_images = os.path.join(data_dir, "train") 
    val_images = os.path.join(data_dir, "val")
    train_annot = os.path.join(data_dir, "train.json") 
    val_annot = os.path.join(data_dir, "val.json") 

    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    # declare model.
    model = FrozenViLTVizWizVQA(
        model_path=model_path, 
        label2id=label2id, 
        device=device,
    )
    # create the optimizer.
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # restore from checkpoint.
    if args.checkpoint_path is not None:
        ckpt_dict = torch.load(
            args.checkpoint_path, 
            map_location=model.device
        )
        model.load_state_dict(ckpt_dict["model_state_dict"])
        optimizer.load_state_dict(ckpt_dict["optim_state_dict"])

    batch_size = args.batch_size
    train_data = ViLTVizWizVQABestAnsDataset(
        train_annot, train_images,
        label2id=label2id,
        model_path=model_path, 
    )
    val_data = ViLTVizWizVQABestAnsDataset(
        val_annot, val_images,
        label2id=label2id,
        model_path=model_path, 
    )
    train_loader = ViLTDataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        split="train",
    )
    val_loader = ViLTDataLoader(
        dataset=val_data, 
        batch_size=batch_size, 
        split="val",
    )
    epochs = args.epochs
    
    # best val acc, step & epoch
    best_val_acc = 0
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

    # train-eval loop
    for epoch in range(epochs):
        # reset the train log.
        open(train_log_path.format(epoch), "w") 
        train_bar = tqdm(enumerate(train_loader), 
                         total=len(train_loader), 
                         desc="commencing training")
        tot, matches, train_epoch_losses = 0, 0, []
        for step, batch in train_bar:
            model.train()
            model.zero_grad()
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs, loss = model(**batch)

            # backward and optimizer step
            loss.backward()
            optimizer.step()

            preds = outputs.logits.argmax(axis=-1).detach().cpu()
            labels = batch["labels"].detach().cpu()
            batch_matches = (preds == labels).sum().item() 
            batch_tot = len(preds)
            matches += batch_matches
            tot += batch_tot
            acc = matches/tot
            train_epoch_losses.append(loss.item())
            train_bar.set_description(f"T: {epoch+1}/{epochs}: a: {100*acc:.2f} ba: {(100*batch_matches/batch_tot):.2f} bl: {loss.item():.3f} l: {np.mean(train_epoch_losses):.3f}")

            # log stats.
            if (step+1) % args.log_steps == 0 or (step+1) == len(train_loader):
                with open(train_log_path.format(epoch), "a") as f:
                    f.write(json.dumps({"train_acc": acc, "step": step, "loss": np.mean(train_epoch_losses)})+"\n")

            # check if evalulation should be done:
            if (step+1) % args.eval_steps == 0 or (step+1) == len(train_loader):
                val_loss, val_acc = validate_vilt(
                    model=model, dataloader=val_loader, 
                    eval_step=step, epoch=epoch, args=args,
                )
                # save the best model.
                if val_acc > best_val_acc:
                    best_val_step = step
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    model_save_path
                    # save the checkpoint for model/optimizer
                    save_current_model(model=model, optimizer=optimizer, 
                                       save_path=model_save_path,
                                       epoch=best_val_epoch, 
                                       step=best_val_step)
                with open(val_log_path, "a") as f:
                    f.write(json.dumps({
                        "val_acc": val_acc, "val_loss": val_loss,
                        "epoch": epoch, "step": step,
                        "best_val_epoch": best_val_epoch,
                        "best_val_step": best_val_step,
                        "best_val_acc": best_val_acc,
                    })+"\n")
          
def get_args():
    """get terminal arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="./data/VQA", type=str, help="path to the VQA dataset")
    parser.add_argument("-mp", "--model_path", default="dandelin/vilt-b32-finetuned-vqa", type=str, help="name/path of the HF checkpoint")
    parser.add_argument("-t", "--train", action="store_true", help="finetune ViLT model")
    parser.add_argument("-p", "--predict", action="store_true", help="generate predictions for data with labels")
    parser.add_argument("-tu", "--test_unseen", action="store_true", help="generate predictions for data without labels")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-ls", "--log_steps", default=10, type=int, help="number of steps after which train stats are logged")
    parser.add_argument("-es", "--eval_steps", default=100, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-exp", "--exp_name", type=str, required=True, help="name of the experiment")
    parser.add_argument("-de", "--device", default="cuda:0", type=str, help="device to be used for training, defaults to cuda:0")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size to be used for training")
    parser.add_argument("-ckpt", "--checkpoint_path", type=Union[str, None], default=None, help="checkpoint to be loaded")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="learning rate for training")
    args = parser.parse_args()

    return args

class CLIPTVizWizVQABestAnsDataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, model_path: str="ViT-L/14", label2id: dict={}, preprocess, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.label2id = label2id
        self.preprocess = preprocess

    def __getitem__(self, i: int):
        item = super(CLIPTVizWizVQABestAnsDataset, self).__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"]).convert("RGB")
        question = item["question"]
        encoding = {}
        encoding["question"] = self.preprocess(question)
        encoding["image_features"] = self.preprocess(image)
        encoding["labels"] = torch.as_tensor(self.label2id.get(item["answer"]))
        
        return encoding
    
class FrozenCLIPVizWizVQAMultimodal(nn.Module):
    def __init__(self, model_path: str="ViT-L/14",
                 device: str="cuda:0", label2id: dict={}, 
                 lang_emb_size: int=768, image_emb_size: int=768):
        super().__init__()
        self.model_path = model_path
        self.model, self.preprocess = clip.load(args.model_path)
        self.num_classes = len(label2id)
        self.emb_size = lang_emb_size+image_emb_size
        self.upscale_size = self.num_classes // 2
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.emb_size, 
                out_features=2048, 
                bias=True
            ),
            nn.LayerNorm(
                (2048), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                in_features=2048, 
                out_features=len(label2id), 
                bias=True
            ),
        )
        # freeze CLIP params:
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(device)
        print(f"moving model to device: {device}")

    def parameters(self):
        return self.classifier.parameters()

    def train(self):
        self.classifier.train()

    def eval(self):
        self.classifier.eval()

    def forward(self, **encoding):
        """return outputs and loss"""
        outputs = self.model.encode_image(encoding["image_features"]).float()
        outputs = self.model.encode_text(encoding["question"]).float()
        logits = self.classifier(outputs)
        labels = encoding["labels"]

        loss = self.loss_fn(logits, labels)
        return logits, loss

# main
if __name__ == "__main__":
    # sanity checking ViLT (assessing zero-shot performance)
    # split = "val"
    # zero_shot_preds = test_vilt(split=split, break_at=100000)
    # with open(f"experiments/{split}_zero_shot_vilt_vqa2_preds.json", "w") as f:
    #     json.dump(zero_shot_preds, f, indent=4)
    args = get_args()
    if args.train: 
        pass
        # finetune_vilt(args)
    elif args.predict: predict_vilt(args) 
    elif args.test_unseen: test_unseen_vilt(args)
    # python -m src.baselines.simple_multimodal -t -d 'cuda:0'
    # python -m src.baselines.simple_multimodal -t -de 'cuda:0' -exp frozen_vilt2
    # python -m src.baselines.simple_multimodal -p -de 'cuda:0' -exp frozen_vilt2
    # python -m src.baselines.simple_multimodal -tu -de 'cuda:0' -exp frozen_vilt2