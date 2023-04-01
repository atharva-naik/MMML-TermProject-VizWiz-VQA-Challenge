# Author: Yash Butala

import clip
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from src.datautils import VizWizVQABestAnsDataset
from tqdm import tqdm
import json
import pdb
import os
from torchvision import transforms

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
        encoding["question"] = question
        encoding["image_features"] = self.preprocess(image).to("cuda:0")
        encoding["labels"] = torch.as_tensor(self.label2id.get(item["answer"])).to("cuda:0")
        return encoding
    
class FrozenCLIPVizWizVQA(nn.Module):
    def __init__(self, model_path: str="ViT-L/14",
                 device: str="cuda:0", label2id: dict={}, emb_size: int=768):
        super().__init__()
        self.model_path = model_path
        self.model, self.preprocess = clip.load(args.model_path)
        self.num_classes = len(label2id)
        self.emb_size = emb_size
        self.upscale_size = self.num_classes // 2
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=768, 
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
        logits = self.classifier(outputs)
        labels = encoding["labels"]

        loss = self.loss_fn(logits, labels)
        return logits, loss
    

class FrozenResNetVizWizVQA(nn.Module):
    def __init__(self, model_path: str="resnet152",
                 device: str="cuda:0", label2id: dict={}, emb_size: int=768):
        super().__init__()
        self.model_path = model_path
        self.model =  torch.hub.load('pytorch/vision:v0.10.0', model_path, pretrained=True)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.num_classes = len(label2id)
        self.emb_size = emb_size
        self.upscale_size = self.num_classes // 2
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=1000, 
                out_features=2000, 
                bias=True
            ),
            nn.LayerNorm(
                (2000), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                in_features=2000, 
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
        outputs = self.model(encoding["image_features"]).float()
        logits = self.classifier(outputs)
        labels = encoding["labels"]

        loss = self.loss_fn(logits, labels)
        return logits, loss
    
    
def train_clip(args):
    
    model_save_path = os.path.join("experiments", args.exp_name, "best_model.pt")
    val_log_path = os.path.join("experiments", args.exp_name, "logs.jsonl")
    if not os.path.isdir(os.path.join("experiments", args.exp_name)):
        os.mkdir(os.path.join("experiments", args.exp_name))
        
    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    model = None
    if args.base_model == "clip":
        model =  FrozenCLIPVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, 
            device=args.device,
        )
    else:
        model = FrozenResNetVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, 
            device=args.device,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
       
    train_dataset = CLIPTVizWizVQABestAnsDataset(annot_path="data/VQA/train.json", images_dir="../data/train", label2id=label2id, preprocess=model.preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = CLIPTVizWizVQABestAnsDataset(annot_path="data/VQA/val.json", images_dir="../data/val", label2id=label2id, preprocess=model.preprocess)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    

    best_val_acc = 0
    for epoch in range(args.epochs):
        train_bar = tqdm(enumerate(train_dataloader), 
                        total=len(train_dataloader), 
                        desc="Training in progress...")
        tot, matches, train_epoch_losses = 0, 0, []
        for i, data in train_bar:
            model.train()
            model.zero_grad()
            outputs, loss = model(**data)
            loss.backward()
            optimizer.step()
            
            preds = outputs.argmax(axis=-1).detach().cpu()
            labels = data["labels"].detach().cpu()

            batch_matches = (preds == labels).sum().item() 
            batch_tot = len(preds)
            matches += batch_matches
            tot += batch_tot
            acc = matches/tot
            bacc = batch_matches/batch_tot
            train_epoch_losses.append(loss.item())
            train_bar.set_description(f"T: {epoch+1}/{args.epochs}: a: {100*acc:.2f} ba: {(100*bacc):.2f} bl: {loss.item():.3f} l: {np.mean(train_epoch_losses):.3f}")

            
            if (i+1) % args.eval_steps == 0 or (i+1) == len(train_dataloader):
                val_loss, val_acc = validate_clip( model=model, dataloader=val_dataloader,eval_step=i, epoch=epoch, args=args)
                # save the best model.
                print(f"val acc: {val_acc}")
                if val_acc > best_val_acc:
                    best_val_step = i
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    # save the checkpoint for model/optimizer
                    print(f"saving model.. acc={val_acc}")
                    save_current_model(model=model, optimizer=optimizer, 
                        save_path=model_save_path,
                        epoch=best_val_epoch, 
                        step=best_val_step)
                    with open(val_log_path, "a") as f:
                        f.write(json.dumps({
                            "val_acc": val_acc, "val_loss": val_loss,
                            "epoch": epoch, "step": i,
                            "best_val_epoch": best_val_epoch,
                            "best_val_step": best_val_step,
                            "best_val_acc": best_val_acc,
                        })+"\n")
                
                        
        
def validate_clip(model, dataloader, eval_step, epoch, args):
    val_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating...")
    tot, matches, val_losses = 0, 0, []
    model.eval()
    for i, data in  val_bar:
        model.zero_grad()
        with torch.no_grad():
            outputs, loss = model(**data)

            preds = outputs.argmax(axis=-1).detach().cpu()
            labels = data["labels"].detach().cpu()
            
            batch_matches = (preds == labels).sum().item() 
            batch_tot = len(preds)
            matches += batch_matches
            tot += batch_tot
            acc = matches/tot
            bacc = batch_matches/batch_tot
            val_losses.append(loss.item())
            val_bar.set_description(f"T: {epoch+1}/{args.epochs}: a: {100*acc:.2f} ba: {(100*bacc):.2f} bl: {loss.item():.3f} l: {np.mean(val_losses):.3f}")

    return np.mean(val_losses), matches/tot


def predict_clip(args, move_to_cuda: bool=False) -> str:

    model_path = os.path.join("experiments", args.exp_name, "best_model.pt")
        
    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    id2label = {v: k for k,v in label2id.items()}

    model = None
    if args.base_model == "clip":
        model =  FrozenCLIPVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, 
            device=args.device,
        )
    else:
        model = FrozenResNetVizWizVQA(
            model_path="resnet152", 
            label2id=label2id, 
            device=args.device,
        )
    
    ckpt_dict = torch.load(
        model_path, 
        map_location=args.device
    )
    model.load_state_dict(ckpt_dict["model_state_dict"])
    model.eval()
    
    results = {}
    results["accuracy"] = 0
    results["preds"] = []
    val_dataset = CLIPTVizWizVQABestAnsDataset(annot_path="data/VQA/val.json", images_dir="../data/val", label2id=label2id, preprocess=model.preprocess)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    val_bar = tqdm(enumerate(val_dataloader), 
                        total=len(val_dataloader), 
                        desc="Hang on, Running predictions...")
    tot, matches = 0, 0
    for i, data in val_bar:

        model.zero_grad()
        with torch.no_grad():
            outputs, loss = model(**data)

        preds = outputs.argmax(axis=-1).detach().cpu()
        labels = data["labels"].detach().cpu()

        matches += (preds == labels).sum().item() 
        tot += len(preds)
        
        for pred, label in zip(preds ,labels):
            results["preds"].append({"pred": id2label[pred.item()], "true": id2label[label.item()]})

    results["accuracy"] = matches/tot
    if args.pred_file == None:
        args.pred_file = os.path.join("experiments", args.exp_name, "pred.json")
    with open(args.pred_file, "w") as fn:
        fn.write(json.dumps(results, indent=4))


    # print("Predicted answer:", )
    
def save_current_model(model, optimizer, epoch: int, step: int, save_path: str):
    """save the current model and optimizer state"""
    ckpt_dict = {}
    ckpt_dict["model_state_dict"] = model.state_dict()
    ckpt_dict["optim_state_dict"] = optimizer.state_dict()
    ckpt_dict["epoch"] = epoch
    ckpt_dict["step"] = step
    torch.save(ckpt_dict, save_path)
    
    
def get_args():
    """get terminal arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="../data/", type=str, help="path to the VQA dataset")
    parser.add_argument("-bm", "--base_model", default="clip", type=str, help="name/path of the CLIP checkpoint")
    parser.add_argument("-mp", "--model_path", default="ViT-L/14", type=str, help="name/path of the CLIP model")
    parser.add_argument("-t", "--train", action="store_true", help="finetune CLIP model")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-ls", "--log_steps", default=10, type=int, help="number of steps after which train stats are logged")
    parser.add_argument("-es", "--eval_steps", default=100, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-exp", "--exp_name", type=str, help="name of the experiment")
    parser.add_argument("-de", "--device", default="cuda:0", type=str, help="device to be used for training, defaults to cuda:0")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size to be used for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("-pfn", "--pred_file", type=str, help="file to store predictions")
    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    args = get_args()
    # train_clip(args)
    predict_clip(args, move_to_cuda=True)
    

            

            



