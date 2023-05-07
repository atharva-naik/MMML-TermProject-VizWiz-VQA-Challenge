# Author: Yash Butala
import clip
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import itertools
from src.datautils import VizWizVQABestAnsDataset
from tqdm import tqdm
import json
import pdb
import os
import sys
from torchvision import transforms
import math
from torch.utils.data import RandomSampler

from src.PythonHelperTools.vqaTools.vqa import VQA
from src.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import default_collate


def read_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "": continue
            data.append(json.loads(line))

    return data

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
        encoding["question"] = clip.tokenize(question).squeeze()
        encoding["image_features"] = self.preprocess(image)
        encoding["labels"] = torch.as_tensor(self.label2id.get(item["answer"]))
        return encoding


# skill aware CLIP for VizWiz VQA.
class SkillAwareCLIPVizWizVQABestAnsDataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, aux_tokens_data: str="",
                 model_path: str="ViT-L/14", label2id: dict={}, 
                 preprocess,flag: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.label2id = label2id
        assert aux_tokens_data != "", "missing path to skill labels"
        self.aux_tokens_data = read_jsonl(aux_tokens_data)
        self.preprocess = preprocess
        self.skill_to_ind = {
            "TXT": 0,
            "OBJ": 1,
            "COL": 2,
            "CNT": 3,
            "OTH": 4,
        }
        self.flag = flag
    def __getitem__(self, i: int):
        item = super(SkillAwareCLIPVizWizVQABestAnsDataset, self).__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"]).convert("RGB")
        question = item["question"]
        skills = item["skills"]
        # pdb.set_trace()
        aux_tokens = self.aux_tokens_data[i]
        objects = " ".join(aux_tokens["objects"][0])
        scene_text = " ".join(aux_tokens["ocr"][0])
        encoding = {}
        encoding["question"] = clip.tokenize(question, truncate=True).squeeze()
        encoding["flag"] = self.flag
        # make aux_tokens["skills"] into single best one
        if item.get("main_skill") != None:
            encoding["main_skill"] = torch.as_tensor(int(item["main_skill"]))
        else:
            encoding["main_skill"] = torch.as_tensor(self.skill_to_ind[aux_tokens["skills"][0]])
        # encoding["skills"] = torch.as_tensor(self.skill_to_ind[aux_tokens["skills"][0]])
        encoding["objects"] = clip.tokenize(objects, truncate=True).squeeze()
        encoding["scene_text"] = clip.tokenize(scene_text, truncate=True).squeeze()
        encoding["image_features"] = self.preprocess(image)
        encoding["labels"] = torch.as_tensor(self.label2id.get(item["answer"]))

        return encoding
 
class FrozenCLIPVizWizVQA(nn.Module):
    def __init__(self, model_path: str="ViT-L/14",
                 device: str="cuda:0", label2id: dict={}, image_emb_size: int=768, lang_emb_size: int = 768 ):
        super().__init__()
        self.model_path = model_path
        self.model, self.preprocess = clip.load(args.model_path)
        self.num_classes = len(label2id)
        self.emb_size = lang_emb_size*3+image_emb_size
        self.upscale_size1 = self.num_classes // 2
        self.upscale_size2 = round(self.upscale_size1 * 0.7)
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.emb_size, 
                out_features=self.upscale_size1, 
                bias=True
            ),
            nn.LayerNorm(
                (self.upscale_size1), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                in_features=self.upscale_size1, 
                out_features=self.upscale_size2, 
                bias=True
            ),
            nn.LayerNorm(
                (self.upscale_size2), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
        )
        self.skill_output = nn.Sequential(
            nn.Linear(
                in_features=self.upscale_size2, 
                out_features=5, 
                bias=True
            ),
        )
        self.vqa_output = nn.Sequential(
            nn.Linear(
                in_features=self.upscale_size2, 
                out_features=len(label2id), 
                bias=True
            )
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
        self.skill_output.train()
        self.vqa_output.train()

    def eval(self):
        self.classifier.eval()
        self.skill_output.eval()
        self.vqa_output.eval()

    def forward(self, task: str, **encoding):
        """return outputs and loss"""
        image_outputs = self.model.encode_image(encoding["image_features"].to(self.device)).float()
        question_outputs = self.model.encode_text(encoding["question"].to(self.device)).float()
        text_outputs = self.model.encode_text(encoding["scene_text"].to(self.device)).float()
        object_outputs = self.model.encode_text(encoding["objects"].to(self.device)).float()
        multitask_logits = self.classifier(torch.cat([image_outputs, question_outputs, text_outputs, object_outputs ], -1))
        logits = None
        loss = None
        
        if task == "skill":
            logits = self.skill_output(multitask_logits)
            labels = encoding["main_skill"].to(self.device)
            loss = self.loss_fn(logits, labels)
        elif task  == "vqa":
            logits = self.vqa_output(multitask_logits)
            labels = encoding["labels"].to(self.device)
            loss = self.loss_fn(logits, labels)
        else:
            print(encoding["flag"])
            print("Error tasks should be either skill or vqa")
            sys.exit()

        return logits, loss
    
def multitask_collate_fn(batch):
    flag_to_data = {}
    for data in batch:
        flag = data['flag']
        if flag not in flag_to_data:
            flag_to_data[flag] = []
        flag_to_data[flag].append(data)
    
    batches = []
    for flag, data_list in flag_to_data.items():
        # Collate the data samples in each batch as usual
        collated_data = default_collate(data_list)
        batches.append(collated_data)
    
    return batches

    
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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
       
    train_dataset_vqa = SkillAwareCLIPVizWizVQABestAnsDataset(annot_path="data/VQA/train.json", images_dir="../data/train", label2id=label2id, preprocess=model.preprocess, aux_tokens_data="./data/VQA/train_aux_info_tokens.jsonl", flag = "vqa", skill_annot_path='./data/skill/vizwiz_skill_typ_train.csv')
    train_dataloader_vqa = DataLoader(train_dataset_vqa, batch_size=args.batch_size, shuffle=True)
    
    train_dataset_skill = SkillAwareCLIPVizWizVQABestAnsDataset(annot_path="data/VQA/train.json", images_dir="../data/train", label2id=label2id, preprocess=model.preprocess, aux_tokens_data="./data/VQA/train_aux_info_tokens.jsonl", flag = "skill", skill_annot_path='./data/skill/vizwiz_skill_typ_train.csv')
    train_dataloader_skill = DataLoader(train_dataset_skill, batch_size=args.batch_size, shuffle=True)
    
    combined_dataset = ConcatDataset([train_dataset_vqa, train_dataset_skill])
    combined_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=multitask_collate_fn)
    
    val_dataset = SkillAwareCLIPVizWizVQABestAnsDataset(annot_path="data/VQA/val.json", images_dir="../data/val", label2id=label2id, preprocess=model.preprocess, aux_tokens_data="./data/VQA/val_aux_info_tokens.jsonl",flag="vqa")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    

    best_val_acc = 0
    for epoch in range(args.epochs):
        train_bar = tqdm(enumerate(combined_dataloader), 
                        total=len(combined_dataloader), 
                        desc="Training in progress...")
        tot, matches, train_epoch_losses = 0, 0, []
        for i, data in train_bar:
            if data[0]['flag'][0]=="vqa":
                tmp = data[0]
                data[0] = data[1]
                data[1] = tmp
            # now data[0] is skill and data[1] is vqa
            
            model.train()
            model.zero_grad()
            outputs, loss = model("skill", **data[0])
            loss.backward()
            optimizer.step()

            model.train()
            model.zero_grad()
            outputs, loss = model("vqa", **data[1])
            loss.backward()
            optimizer.step()
                 
            preds = outputs.argmax(axis=-1).detach().cpu()
            labels = data[1]["labels"].detach().cpu()

            batch_matches = (preds == labels).sum().item() 
            batch_tot = len(preds)
            matches += batch_matches
            tot += batch_tot
            acc = matches/tot
            bacc = batch_matches/batch_tot
            train_epoch_losses.append(loss.item())
            train_bar.set_description(f"T: {epoch+1}/{args.epochs}: a: {100*acc:.2f} ba: {(100*bacc):.2f} bl: {loss.item():.3f} l: {np.mean(train_epoch_losses):.3f}")

            
            if (i+1) % args.eval_steps == 0 or (i+1) == len(combined_dataloader):
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
            outputs, loss = model("vqa", **data)

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

    model = FrozenCLIPVizWizVQA(
        model_path=args.model_path, 
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
    
def get_eval_scores(annFile, resFile): 
    gold_json = json.load(open(annFile))
    pred_json = json.load(open(resFile))
    formatted_pred_json = []
    for gold_exp, pred_exp in zip(gold_json,pred_json["preds"]):
        formatted_exp = {"image": gold_exp["image"], "answer": pred_exp["pred"]}
        formatted_pred_json.append(formatted_exp)
    formatted_file = f"experiments/{args.exp_name}/formatted_pred.json"
    f = json.dump(formatted_pred_json, open(formatted_file, "w"), indent=4)
    vqa = VQA(annFile)
    vqaRes = VQA(formatted_file)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    # evaluate VQA results
    vqaEval.evaluate() 
    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")

    print("Caption metrics are :")
    for k, v in list(vqaEval.caption_metric.items()):
        print("%s: %.2f"%(k,v))
        
 
 
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
    train_clip(args)
    predict_clip(args, move_to_cuda=True)
    
    get_eval_scores("data/VQA/val.json", 
                f'experiments/{args.exp_name}/pred.json')

            

