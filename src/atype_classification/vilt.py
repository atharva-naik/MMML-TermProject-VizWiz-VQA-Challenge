# ViLT base answer type classifier.
# code to train and predict with simple multimodal baselines.

import os
import json
import torch
import random
import argparse
import numpy as np
from typing import *
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from src.datautils import VizWizVQABestAnsDataset
from src.PythonHelperTools.vqaTools.vqa import VQA
from transformers import ViltProcessor, ViltForQuestionAnswering
# from src.baselines.simple_multimodal import ViLTVizWizVQABestAnsDataset
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
        if split == "train": 
            super().__init__(dataset, shuffle=True, 
                             collate_fn=self.custom_collate_fn, **kwargs)
        else: super().__init__(dataset, shuffle=False, 
                               collate_fn=self.custom_collate_fn, **kwargs)
        self.processor = dataset.processor
        self.a_type_to_ind = dataset.a_type_to_ind
        self.ind_to_a_type = dataset.ind_to_a_type

    def custom_collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        has_labels = False
        if "labels" in batch[0]: 
            has_labels = True
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
        if has_labels: batch['labels'] = torch.stack(labels)
        
        return batch

class ViLTVizWizATypeClassifier(nn.Module):
    def __init__(self, model_path: str="dandelin/vilt-b32-finetuned-vqa",
                 device: str="cuda:0", num_classes: int=4, emb_size: int=768):
        super().__init__()
        self.model_path = model_path
        self.model = ViltForQuestionAnswering.from_pretrained(model_path)
        
        self.emb_size = emb_size
        self.num_classes = num_classes
        self.upscale_size = 1000 # self.num_classes // 2
        self.model.classifier = nn.Sequential(
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
                out_features=num_classes, 
                bias=True
            ),
        )
        # nn.Sequential(
        #     nn.Linear(
        #         in_features=self.emb_size, 
        #         out_features=self.upscale_size, 
        #         bias=True
        #     ),
        #     nn.LayerNorm(
        #         (self.upscale_size,), eps=1e-05, 
        #         elementwise_affine=True
        #     ),
        #     nn.GELU(approximate="none"),
        #     nn.Linear(
        #         in_features=self.upscale_size, 
        #         out_features=self.num_classes, 
        #         bias=True
        #     ),
        #     nn.Sigmoid(),
        # )
        # freeze ViLT params:
        # for param in self.model.vilt.parameters():
            # param.requires_grad = False
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(device)
        print(f"moving model to device: {device}")

    # def parameters(self):
    #     return self.model.classifier.parameters()
    
    # def train(self):
    #     self.model.classifier.train()

    # def eval(self):
    #     self.model.classifier.eval()

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

class ViLTVizWizATypeDataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, model_path: str="dandelin/vilt-b32-finetuned-vqa", **kwargs):
        super().__init__(*args, filter_out_no_skill_instances=False, **kwargs)
        self.model_path = model_path
        # intialize processor for ViLT:
        self.processor = ViltProcessor.from_pretrained(model_path)

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"])
        question = item["question"]
        encoding = self.processor(image, question, padding="max_length", 
                                  truncation=True, return_tensors="pt")
        for key in encoding: encoding[key] = encoding[key][0]
        encoding["labels"] = torch.as_tensor(item["answer_type"])

        return encoding

class ViLTVizWizATypeUnseenData(VizWizVQABestAnsDataset):
    def __init__(self, *args, model_path: str="dandelin/vilt-b32-finetuned-vqa", **kwargs):
        super().__init__(*args, filter_out_no_skill_instances=False, **kwargs)
        self.model_path = model_path
        # intialize processor for ViLT:
        self.processor = ViltProcessor.from_pretrained(model_path)

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"])
        question = item["question"]
        encoding = self.processor(image, question, padding="max_length", 
                                  truncation=True, return_tensors="pt")
        for key in encoding: encoding[key] = encoding[key][0]

        return encoding
# # sanity checking/testing functions:
# def test_vilt_predict(image_path: str, query: str, 
#                       processor: ViltProcessor, model,
#                       move_to_cuda: bool=False) -> str:
#     image = Image.open(image_path)
#     # prepare inputs (tokenization basically)
#     encoding = processor(
#         image, query, return_tensors="pt",
#         truncation=True, max_length=40,
#         padding="max_length",
#     )
#     if move_to_cuda:
#         # print("moving encoding to cuda")
#         for k,v in encoding.items():
#             encoding[k] = v.cuda()
#     outputs = model(**encoding)
#     logits = outputs.logits.detach().cpu()
#     binidx = (logits>0.5).item().float() # predict binary inputs.

#     return binidx
# def test_vilt(split: str="train", use_cuda: bool=True, break_at: int=5) -> List[str]:
#     # initialize ViLT and its processor/tokenizer.
#     processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#     model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#     # load the dataset in memory
#     data_path = f"./data/VQA/{split}.json"
#     data = VQA(annotation_file=data_path).getAnns()
#     preds = []
#     move_to_cuda = False
#     if use_cuda and torch.cuda.is_available():
#         model.cuda()
#         move_to_cuda = True
#     for i, rec in tqdm(enumerate(data), total=len(data)):
#         image_path = os.path.join("./data/VQA", split, rec["image"])
#         query = rec["question"]
#         pred = test_vilt_predict(image_path, query, processor, 
#                                  model, move_to_cuda=move_to_cuda)
#         # for pidx, tidx in enumerate(pred, rec["answers"]):
#         preds.append({"question": query, "trues": rec["answers"], "pred": pred})
#         if (i+1) == break_at: break

#     return preds
def save_current_model(model, optimizer, epoch: int, step: int, save_path: str):
    """save the current model and optimizer state"""
    ckpt_dict = {}
    ckpt_dict["model_state_dict"] = model.state_dict()
    ckpt_dict["optim_state_dict"] = optimizer.state_dict()
    ckpt_dict["epoch"] = epoch
    ckpt_dict["step"] = step
    torch.save(ckpt_dict, save_path)

def validate_vilt(model, dataloader: DataLoader, eval_step: int, 
                  epoch: int, args: argparse.Namespace) -> Tuple[List[dict], float, float, float]:
    val_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="validating")
    val_losses = []
    val_preds = []
    tot, matches = 0, 0
    model.eval()
    for step, batch in val_bar:
        model.zero_grad()
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].to(args.device)
            outputs, loss = model(**batch)
            preds = outputs.logits.detach().argmax(dim=-1).cpu()
            labels = batch["labels"].detach().cpu()
            batch_matches = (preds == labels).sum().item() 
            batch_tot = len(preds)
            matches += batch_matches
            tot += batch_tot
            acc = matches/tot
            # if output_preds:
            for pred, label in zip(preds.tolist(), labels.tolist()):
                val_preds.append({
                    "pred": dataloader.ind_to_a_type[pred],
                    "true": dataloader.ind_to_a_type[label],
                })
            val_losses.append(loss.item())
            val_bar.set_description(f"V: {epoch+1}/{args.epochs} ({eval_step}): a: {100*acc:.2f} ba: {(100*batch_matches/batch_tot):.2f} bl: {loss.item():.3f} l: {np.mean(val_losses):.3f}")
    # if output_preds: 
    all_trues = [d["true"] for d in val_preds]
    all_preds = [d["pred"] for d in val_preds]
    val_F = f1_score(all_trues, all_preds, average="macro") # calculate F1 score.
    
    return val_preds, val_F, np.mean(val_losses), matches/tot
    # else: return np.mean(val_losses), matches/tot

def validate_vilt_without_labels(model, dataloader: DataLoader) -> Union[Tuple[float, float], Tuple[List[dict], float, float]]:
    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="validating")
    split_preds = []
    model.eval()
    id = 0
    for step, batch in bar:
        model.zero_grad()
        with torch.no_grad():
            for key in batch: batch[key] = batch[key].to(args.device)
            outputs = model(**batch)
            preds = outputs.logits.detach().argmax(dim=-1).cpu()
            for pred in preds.tolist():
                split_preds.append({
                    "id": id,
                    "pred": dataloader.ind_to_a_type[pred], # [dataloader.ind_to_skill[i] for i,lbl in enumerate(pred) if lbl==1]
                })
                id += 1
            bar.set_description(f"predicting on unseen data")
    
    return split_preds

def predict_unseen_vilt(args):
    data_dir = args.data_dir # default: ./data/VQA
    model_path = args.model_path # model_path: dandelin/vilt-b32-finetuned-vqa
    device = args.device
    split: str = "test"
    split_images = os.path.join(data_dir, split)
    split_annot = os.path.join(data_dir, f"{split}.json")
    # declare model.
    model = ViLTVizWizATypeClassifier(
        model_path=model_path, 
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
    batch_size = args.batch_size
    data = ViLTVizWizATypeUnseenData(
        split_annot, split_images, 
        model_path=model_path,
    )
    data_loader = ViLTDataLoader(dataset=data, batch_size=batch_size, split="val")
    predict_log_path = os.path.join("experiments", args.exp_name, f"{split}_unseen_preds.json")
    
    model.eval()
    preds = validate_vilt_without_labels(
        model=model, 
        dataloader=data_loader, 
    )
    with open(predict_log_path, "a") as f:
        json.dump(preds, f)

def predict_vilt(args):
    data_dir = args.data_dir # default: ./data/VQA
    model_path = args.model_path # model_path: dandelin/vilt-b32-finetuned-vqa
    device = args.device
    split: str = "val"
    split_images = os.path.join(data_dir, split)
    split_annot = os.path.join(data_dir, f"{split}.json")
    # declare model.
    model = ViLTVizWizATypeClassifier(
        model_path=model_path, 
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
    data = ViLTVizWizATypeDataset(
        split_annot, split_images, 
        model_path=model_path,
    )
    data_loader = ViLTDataLoader(dataset=data, batch_size=batch_size, split="val")
    predict_log_path = os.path.join("experiments", args.exp_name, f"predict_logs.json")
    
    model.eval()
    val_preds, val_F, val_loss, val_acc = validate_vilt(
        model=model, dataloader=data_loader, 
        eval_step=0, epoch=0, args=args,
    )
    with open(predict_log_path, "w") as f:
        json.dump({
            "predict_acc": val_acc,
            "predict_F": val_F, 
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

    # declare model.
    model = ViLTVizWizATypeClassifier(
        model_path=model_path, device=device,
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
    train_data = ViLTVizWizATypeDataset(
        train_annot, train_images, model_path=model_path,
    )
    val_data = ViLTVizWizATypeDataset(
        val_annot, val_images, model_path=model_path, 
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
    best_val_F = 0
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
            preds = outputs.logits.detach().argmax(dim=-1).cpu()
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
                _, val_F, val_loss, val_acc = validate_vilt(
                    model=model, dataloader=val_loader, 
                    eval_step=step, epoch=epoch, args=args,
                )
                # save the best model.
                if val_F > best_val_F:
                    best_val_step = step
                    best_val_F = val_F
                    best_val_epoch = epoch
                    model_save_path
                    # save the checkpoint for model/optimizer
                    print(f"\x1b[32;1msaving best model with F-score: {best_val_F}\x1b[0m")
                    save_current_model(model=model, optimizer=optimizer, 
                                       save_path=model_save_path,
                                       epoch=best_val_epoch, 
                                       step=best_val_step)
                with open(val_log_path, "a") as f:
                    f.write(json.dumps({
                        "val_acc": val_acc,
                        "val_F": val_F, "val_loss": val_loss,
                        "epoch": epoch, "step": step,
                        "best_val_epoch": best_val_epoch,
                        "best_val_step": best_val_step,
                        "best_val_F": best_val_F,
                    })+"\n")
          
def get_args():
    """get terminal arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="./data/VQA", type=str, help="path to the VQA dataset")
    parser.add_argument("-mp", "--model_path", default="dandelin/vilt-b32-finetuned-vqa", type=str, help="name/path of the HF checkpoint")
    parser.add_argument("-t", "--train", action="store_true", help="finetune ViLT model")
    parser.add_argument("-p", "--predict", action="store_true", help="generate predictions for data with labels")
    parser.add_argument("-pu", "--predict_unseen", action="store_true", help="predict over unseeen data without labels")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-ls", "--log_steps", default=10, type=int, help="number of steps after which train stats are logged")
    parser.add_argument("-es", "--eval_steps", default=100, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-exp", "--exp_name", type=str, required=True, help="name of the experiment")
    parser.add_argument("-de", "--device", default="cuda:0", type=str, help="device to be used for training, defaults to cuda:0")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size to be used for training")
    parser.add_argument("-ckpt", "--checkpoint_path", type=Union[str, None], default=None, help="checkpoint to be loaded")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="learning rate for training")
    args = parser.parse_args()

    return args

# main
if __name__ == "__main__":
    # sanity checking ViLT (assessing zero-shot performance)
    # split = "val"
    # zero_shot_preds = test_vilt(split=split, break_at=100000)
    # with open(f"experiments/{split}_zero_shot_vilt_vqa2_preds.json", "w") as f:
    #     json.dump(zero_shot_preds, f, indent=4)
    args = get_args() 
    if args.train: finetune_vilt(args)
    elif args.predict: predict_vilt(args) 
    elif args.predict_unseen: predict_unseen_vilt(args)
    # python -m src.atype_classification.vilt -t -de 'cuda:0' -exp vilt_atype_clf
    # python -m src.atype_classification.vilt -p -de 'cuda:0' -exp vilt_atype_clf
    # python -m src.atype_classification.vilt -pu -de 'cuda:0' -exp vilt_atype_clf
    # python -m src.atype_classification.vilt -t -de 'cuda:0' -exp vilt_atype_clf2
    # python -m src.atype_classification.vilt -pu -de 'cuda:0' -exp vilt_atype_clf2