import os
import pdb
import clip
import json
import torch
import random
import argparse
import numpy as np
from typing import *
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from src.datautils import VizWizVQABestAnsDataset
from src.PythonHelperTools.vqaTools.vqa import VQA
from src.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


# seed stuff
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
# def correct_word(word: str, checker: SpellChecker):
#     if len(checker.unknown([word])) == 1:
#         corrected_word = checker.correction(word)
#         if corrected_word is None: return word
#         else: return corrected_word
#     return word

# huggingface implementation of CLIP and best answer datase
class SkillAwareHfCLIPVizWizVQADataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, aux_tokens_data: str="",
                 model_path: str="openai/clip-vit-large-patch14", 
                 label2id: dict={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.label2id = label2id
        assert aux_tokens_data != "", "missing path to skill labels"
        self.aux_tokens_data = read_jsonl(aux_tokens_data)
        self.skill_to_ind = {
            "TXT": 0,
            "OBJ": 1,
            "COL": 2,
            "CNT": 3,
            "OTH": 4,
        }

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"])
        question = item["question"]
        aux_tokens = self.aux_tokens_data[i]
        objects = " ".join(aux_tokens["objects"][0])
        scene_text = " ".join(aux_tokens["ocr"][0])
        encoding = {}
        encoding["question"] = question
        skills = [self.skill_to_ind[skill] for skill in aux_tokens["skills"]]
        skills = skills+[5]*(6-len(skills))
        encoding["image"] = image 
        encoding["skills"] = skills
        encoding["objects"] = objects
        encoding["scene_text"] = scene_text
        encoding["labels"] = self.label2id.get(item["answer"])

        return encoding
    
    def collate_fn(self, batch: List[dict]):
        batch_enc = {}
        for k in ["question", "objects", "scene_text"]:
            batch_enc[k] = self.processor(
                text=[rec[k] for rec in batch], 
                truncation=True, padding="longest", 
                return_tensors="pt",
            )
        batch_enc["skills"] = torch.as_tensor([rec["skills"] for rec in batch])
        batch_enc["pixel_values"] = self.processor(
            images=[rec["image"] for rec in batch], 
            return_tensors="pt", padding="longest", 
        )["pixel_values"]
        batch_enc["labels"] = torch.as_tensor([rec["labels"] for rec in batch])

        return batch_enc

# skill aware CLIP for VizWiz VQA.
class SkillAwareCLIPVizWizVQABestAnsDataset(VizWizVQABestAnsDataset):
    def __init__(self, *args, aux_tokens_data: str="", split="train",
                 model_path: str="ViT-L/14", label2id: dict={}, 
                 preprocess, **kwargs):
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
        self.split = split

    # def __getitem__(self, i: int):
    #     item = super(SkillAwareCLIPVizWizVQABestAnsDataset, self).__getitem__(i)
    #     # encode using ViLT processor:
    #     image = Image.open(item["image"]).convert("RGB")
    #     question = item["question"]
    #     aux_tokens = self.aux_tokens_data[i]
    #     skills = " ".join(aux_tokens["skills"])
    #     obj_tags = " ".join(aux_tokens["objects"][0])
    #     scene_text = " ".join(aux_tokens["ocr"][0])
    #     encoding = {}
    #     fused_tokens = skills + " " + question + " " + obj_tags #+ " " + scene_text
    #     encoding["question"] = clip.tokenize(fused_tokens, truncate=True).squeeze()
    #     encoding["image_features"] = self.preprocess(image)
    #     encoding["labels"] = torch.as_tensor(self.label2id.get(item["answer"]))
    #     return encoding

    def __getitem__(self, i: int):
        item = super(SkillAwareCLIPVizWizVQABestAnsDataset, self).__getitem__(i)
        # encode using ViLT processor:
        image = Image.open(item["image"]).convert("RGB")
        question = item["question"]
        aux_tokens = self.aux_tokens_data[i]
        objects = " ".join(aux_tokens["objects"][0])
        scene_text = " ".join(aux_tokens["ocr"][0])
        encoding = {}
        encoding["question"] = clip.tokenize(question, truncate=True).squeeze()
        skills = [self.skill_to_ind[skill] for skill in aux_tokens["skills"]]
        skills = skills+[5]*(6-len(skills))
        encoding["skills"] = torch.as_tensor(skills)
        encoding["objects"] = clip.tokenize(objects, truncate=True).squeeze()
        encoding["scene_text"] = clip.tokenize(scene_text, truncate=True).squeeze()
        encoding["image_features"] = self.preprocess(image)
        if self.split != "test":
            encoding["labels"] = torch.as_tensor(self.label2id.get(item["answer"]))

        return encoding

# skill aware CLIP with attention block to fuse modalities.
class SkillAwareAttnHfCLIP(nn.Module):
    def __init__(self, model_path: str="openai/clip-vit-large-patch14", device: str="cuda:0", 
                 label2id: dict={}, emb_size: int=768, no_skill_mode: bool=False):
        super().__init__()
        self.model_path = model_path
        self.model = CLIPModel.from_pretrained(model_path)
        self.num_classes = len(label2id)
        self.skill_to_ind = {
            "TXT": 0,
            "OBJ": 1,
            "COL": 2,
            "CNT": 3,
            "OTH": 4,
        }
        self.no_skill_mode = no_skill_mode
        self.skill_embs = nn.Embedding(
            len(self.skill_to_ind)+1, 
            emb_size, padding_idx=5,
        )
        self.skill_attn_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=4, 
            batch_first=True,
        )
        self.upscale_size = self.num_classes // 2
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=emb_size, 
                out_features=self.upscale_size, 
                bias=True
            ),
            nn.LayerNorm(
                (self.upscale_size), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                in_features=self.upscale_size, 
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
        return list(self.classifier.parameters())+list(self.skill_embs.parameters())+list(self.skill_attn_layer.parameters())#+list(self.model.visual_projection.parameters())+list(self.model.text_projection.parameters())

    def train(self):
        self.classifier.train()
        self.skill_embs.train()
        self.skill_attn_layer.train()
        # self.model.visual_projection.train()
        # self.model.text_projection.train()

    def eval(self):
        self.classifier.eval()
        self.skill_embs.eval()
        self.skill_attn_layer.eval()
        # self.model.visual_projection.eval()
        # self.model.text_projection.eval()

    def forward(self, **encoding):
        """return outputs and loss"""
        image_seq = self.model.visual_projection(self.model.vision_model(
            pixel_values=encoding["pixel_values"].to(self.device)
        ).last_hidden_state)
        question_seq = self.model.text_projection(self.model.text_model(
            input_ids=encoding["question"]["input_ids"].to(self.device),
            attention_mask=encoding["question"]["attention_mask"].to(self.device),
        ).last_hidden_state)
        objects_seq = self.model.text_projection(self.model.text_model(
            input_ids=encoding["objects"]["input_ids"].to(self.device),
            attention_mask=encoding["objects"]["attention_mask"].to(self.device),
        ).last_hidden_state)
        scene_seq = self.model.text_projection(self.model.text_model(
            input_ids=encoding["scene_text"]["input_ids"].to(self.device),
            attention_mask=encoding["scene_text"]["attention_mask"].to(self.device),
        ).last_hidden_state)
        # print("image:", image_emb.shape)
        # print("question:", question_emb.shape)
        if not self.no_skill_mode:
            skill_seq = self.skill_embs(encoding["skills"].to(self.device))
            skill_attn_mask = (encoding["skills"] == 5).int()
            fusion_seq = torch.cat([
                skill_seq, image_seq, question_seq,
                objects_seq, scene_seq], axis=1
            )
            # print("question:", encoding["question"]["attention_mask"].device)
            # print("skill mask:", skill_attn_mask.device)
            # print("objects:", encoding["objects"]["attention_mask"].device)
            # print("scene text:", encoding["scene_text"]["attention_mask"].device)
            fusion_mask = torch.cat([
                skill_attn_mask, torch.ones(image_seq.shape[0], image_seq.shape[1]),
                encoding["question"]["attention_mask"],
                encoding["objects"]["attention_mask"],
                encoding["scene_text"]["attention_mask"],
            ], axis=1)
        else:
            fusion_seq = torch.cat([
                image_seq, question_seq,
                objects_seq, scene_seq], axis=1
            )
            fusion_mask = torch.cat([
                torch.ones(image_seq.shape[0], image_seq.shape[1]),
                encoding["question"]["attention_mask"],
                encoding["objects"]["attention_mask"],
                encoding["scene_text"]["attention_mask"],
            ], axis=1)
        fusion_mask = ((1-fusion_mask)*(-10000.0)).unsqueeze(dim=-1)
        fusion_mask = fusion_mask.to(self.device)
        fusion_seq += fusion_mask
        fusion_seq = self.skill_attn_layer(fusion_seq)
        # print(f"fusion seq: {fusion_seq.shape}")
        # print(f"fusion mask: {fusion_mask.shape}")
        fusion_seq = fusion_seq.mean(axis=1)
        logits = self.classifier(fusion_seq)
        labels = encoding["labels"].to(self.device)
        loss = self.loss_fn(logits, labels)

        return logits, loss

class SkillAwareHfCLIPVizWizVQA(nn.Module):
    def __init__(self, model_path: str="openai/clip-vit-large-patch14", device: str="cuda:0", 
                 label2id: dict={}, image_emb_size: int=768, 
                 lang_emb_size: int=768, skill_emb_size: int=200,
                 no_skill_mode: bool=False):
        super().__init__()
        self.model_path = model_path
        self.model = CLIPModel.from_pretrained(model_path)
        self.num_classes = len(label2id)
        self.skill_to_ind = {
            "TXT": 0,
            "OBJ": 1,
            "COL": 2,
            "CNT": 3,
            "OTH": 4,
        }
        self.no_skill_mode = no_skill_mode
        if no_skill_mode:
            self.emb_size = 3*lang_emb_size+image_emb_size
        else:
            self.skill_embs = nn.Embedding(
                len(self.skill_to_ind)+1, 
                skill_emb_size, padding_idx=5,
            ) # 5 "skills": TXT, COL, CNT, OBJ, OTH
            self.emb_size = 3*lang_emb_size+image_emb_size+skill_emb_size
        self.upscale_size = self.num_classes // 2
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.emb_size, 
                out_features=self.upscale_size, 
                bias=True
            ),
            nn.LayerNorm(
                (self.upscale_size), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                in_features=self.upscale_size, 
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
        return list(self.classifier.parameters())+list(self.skill_embs.parameters())

    def train(self):
        self.classifier.train()
        self.skill_embs.train()

    def eval(self):
        self.classifier.eval()
        self.skill_embs.eval()

    def forward(self, **encoding):
        """return outputs and loss"""
        # vision_model to be used for last hidden states
        # text_model to be used for last hidden states
        # visual_projection to be applied for each vector in the sequence
        # textual_projection to be applied for each vector in the sequence
        image_emb = self.model.visual_projection(self.model.vision_model(
            pixel_values=encoding["pixel_values"].to(self.device)
        ).pooler_output)
        question_emb = self.model.text_projection(self.model.text_model(
            input_ids=encoding["question"]["input_ids"].to(self.device),
            attention_mask=encoding["question"]["attention_mask"].to(self.device),
        ).pooler_output)
        objects_emb = self.model.text_projection(self.model.text_model(
            input_ids=encoding["objects"]["input_ids"].to(self.device),
            attention_mask=encoding["objects"]["attention_mask"].to(self.device),
        ).pooler_output)
        scene_text_emb = self.model.text_projection(self.model.text_model(
            input_ids=encoding["scene_text"]["input_ids"].to(self.device),
            attention_mask=encoding["scene_text"]["attention_mask"].to(self.device),
        ).pooler_output)
        # print("image:", image_emb.shape)
        # print("question:", question_emb.shape)
        if not self.no_skill_mode:
            skill_emb = self.skill_embs(encoding["skills"].to(self.device)).mean(axis=1)
            # pdb.set_trace()
            logits = self.classifier(torch.cat([image_emb,question_emb,skill_emb,objects_emb,scene_text_emb], -1))
        else: logits = self.classifier(torch.cat([image_emb,question_emb,objects_emb,scene_text_emb], -1))
        labels = encoding["labels"].to(self.device)
        loss = self.loss_fn(logits, labels)

        return logits, loss

class SkillAwareCLIPVizWizVQA(nn.Module):
    def __init__(self, model_path: str="ViT-L/14", device: str="cuda:0", 
                 label2id: dict={}, image_emb_size: int=768, 
                 lang_emb_size: int=768, skill_emb_size: int=200,
                 no_skill_mode: bool=False, no_obj_mode: bool=False,
                 no_scene_text_mode: bool=False, norm_features: bool=False):
        super().__init__()
        self.model_path = model_path
        self.norm_features = norm_features
        self.model, self.preprocess = clip.load(args.model_path)
        self.num_classes = len(label2id)
        self.skill_to_ind = {
            "TXT": 0,
            "OBJ": 1,
            "COL": 2,
            "CNT": 3,
            "OTH": 4,
        }
        self.no_obj_mode = no_obj_mode
        self.no_skill_mode = no_skill_mode
        self.no_scene_text_mode = no_scene_text_mode
        if no_skill_mode:
            print("no skill mode")
            self.emb_size = 3*lang_emb_size+image_emb_size
        else:
            self.skill_embs = nn.Embedding(
                len(self.skill_to_ind)+1, 
                skill_emb_size, padding_idx=5,
            ) # 5 "skills": TXT, COL, CNT, OBJ, OTH
            if no_scene_text_mode and no_obj_mode:
                print("skill embs only mode")
                self.emb_size = lang_emb_size+image_emb_size+skill_emb_size
                print(f"emb_size: {self.emb_size}")
            elif no_scene_text_mode or no_obj_mode:
                if no_obj_mode: print("no object mode")
                if no_scene_text_mode: print("no scene text mode")
                self.emb_size = 2*lang_emb_size+image_emb_size+skill_emb_size
                print(f"emb_size: {self.emb_size}")
            else: self.emb_size = 3*lang_emb_size+image_emb_size+skill_emb_size
        self.upscale_size = self.num_classes // 2
        if norm_features:
            print("doing post-fusion feature normalization!")
            self.fusion_norm = nn.LayerNorm(
                self.emb_size, eps=1e-05, 
                elementwise_affine=True,
            )
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.emb_size, 
                out_features=self.upscale_size, 
                bias=True
            ),
            nn.LayerNorm(
                (self.upscale_size), eps=1e-05, 
                elementwise_affine=True
            ),
            nn.GELU(approximate="none"),
            nn.Linear(
                in_features=self.upscale_size, 
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
        return list(self.classifier.parameters())
        # return list(self.classifier.parameters())+list(self.skill_embs.parameters())
    def train(self):
        self.classifier.train()
        # self.skill_embs.train()
    def eval(self):
        self.classifier.eval()
        # self.skill_embs.eval()
    def forward(self, **encoding):
        """return outputs and loss"""
        image_emb = self.model.encode_image(encoding["image_features"].to(self.device)).float()
        question_emb = self.model.encode_text(encoding["question"].to(self.device)).float()
        modalities = [image_emb, question_emb]
        if not self.no_skill_mode:
            skill_emb = self.skill_embs(encoding["skills"].to(self.device)).mean(axis=1)
            modalities.append(skill_emb)
        if not self.no_obj_mode:
            objects_emb = self.model.encode_text(encoding["objects"].to(self.device)).float()
            modalities.append(objects_emb)
        if not self.no_scene_text_mode:
            scene_text_emb = self.model.encode_text(encoding["scene_text"].to(self.device)).float()
            modalities.append(scene_text_emb)
        fused_features = torch.cat(modalities, -1)
        if self.norm_features: 
            # print(fused_features)
            # print("normalizing features")
            fused_features = self.fusion_norm(fused_features)
            # print(fused_features)
        logits = self.classifier(fused_features)
        # logits = self.classifier(torch.cat([image_emb,question_emb,skill_emb,objects_emb,scene_text_emb], -1))
        # else: logits = self.classifier(torch.cat([image_emb,question_emb,objects_emb,scene_text_emb], -1))
        if "labels" in encoding:
            labels = encoding["labels"].to(self.device)
            loss = self.loss_fn(logits, labels)
            
            return logits, loss
        else: return logits
    
# training loop
def train_clip(args):
    model_save_path = os.path.join("experiments", args.exp_name, "best_model.pt")
    val_log_path = os.path.join("experiments", args.exp_name, "logs.jsonl")
    if not os.path.isdir(os.path.join("experiments", args.exp_name)):
        os.mkdir(os.path.join("experiments", args.exp_name))
        
    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    model = None
    # if args.base_model == "clip":
    if not(args.use_hf_clip):
        model = SkillAwareCLIPVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, device=args.device,
            no_skill_mode=args.no_skill,
            no_obj_mode=args.no_obj,
            no_scene_text_mode=args.no_scene_text,
            norm_features=args.norm_features,
        )
        train_dataset = SkillAwareCLIPVizWizVQABestAnsDataset(
            annot_path="./data/VQA/train.json", images_dir="./data/VQA/train", 
            label2id=label2id, preprocess=model.preprocess,
            aux_tokens_data="./data/VQA/train_aux_info_tokens.jsonl",
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        val_dataset = SkillAwareCLIPVizWizVQABestAnsDataset(
            annot_path="data/VQA/val.json", images_dir="./data/VQA/val", 
            label2id=label2id, preprocess=model.preprocess,
            aux_tokens_data="./data/VQA/val_aux_info_tokens.jsonl",
        )
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        if not args.use_skill_attention:
            model = SkillAwareHfCLIPVizWizVQA(
                model_path=args.model_path, 
                label2id=label2id, device=args.device,
                no_skill_mode=args.no_skill,
            )
        else:
            model = SkillAwareAttnHfCLIP(
                model_path=args.model_path, emb_size=768,
                label2id=label2id, device=args.device,
                no_skill_mode=args.no_skill,
            )
        train_dataset = SkillAwareHfCLIPVizWizVQADataset(
            annot_path="./data/VQA/train.json", images_dir="./data/VQA/train", 
            label2id=label2id, aux_tokens_data="./data/VQA/train_aux_info_tokens.jsonl",
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=True, collate_fn=train_dataset.collate_fn,
        )
        val_dataset = SkillAwareHfCLIPVizWizVQADataset(
            annot_path="data/VQA/val.json", images_dir="./data/VQA/val", 
            label2id=label2id, aux_tokens_data="./data/VQA/val_aux_info_tokens.jsonl",
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, 
            shuffle=True, collate_fn=val_dataset.collate_fn,
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_val_acc = 0
    # best_val_step = -1
    # best_val_epoch = -1
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
                print("starting validation\n\nvalidate_clip()")
                val_loss, val_acc = validate_clip(model=model, dataloader=val_dataloader,eval_step=i, epoch=epoch, args=args)
                # save the best model.
                print(f"step: {i}")
                print(f"epoch: {epoch}")
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
                print(f"best_val_epoch: {best_val_epoch}")
                print(f"best_val_step: {best_val_step}")
                print(f"best_val_acc: {best_val_acc}")
                    
# validation loop
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

def predict_unseen_clip(args) -> str:
    model_path = os.path.join("experiments", args.exp_name, "best_model.pt")
    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    id2label = {v: k for k,v in label2id.items()}
    
    if not(args.use_hf_clip):
        model = SkillAwareCLIPVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, 
            device=args.device,
            no_skill_mode=args.no_skill,
            no_obj_mode=args.no_obj,
            no_scene_text_mode=args.no_scene_text,
            norm_features=args.norm_features,
        )
    else:
        model = SkillAwareHfCLIPVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, 
            device=args.device,
            no_skill_mode=args.no_skill,
        )

    ckpt_dict = torch.load(
        model_path, 
        map_location=args.device
    )
    model.load_state_dict(ckpt_dict["model_state_dict"])
    model.eval()
    
    test_ids = [i["image"] for i in json.load(open("./data/VQA/test.json"))]
    if not(args.use_hf_clip):
        test_dataset = SkillAwareCLIPVizWizVQABestAnsDataset(
            annot_path="data/VQA/test.json", images_dir="./data/VQA/test", 
            label2id=label2id, preprocess=model.preprocess, split="test",
            aux_tokens_data="./data/VQA/test_aux_info_tokens.jsonl",
        )
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        test_dataset = SkillAwareHfCLIPVizWizVQADataset(
            annot_path="data/VQA/test.json", images_dir="./data/VQA/test", 
            label2id=label2id, aux_tokens_data="./data/VQA/test_aux_info_tokens.jsonl",
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, 
            shuffle=False, collate_fn=test_dataset.collate_fn,
        )
    test_bar = tqdm(enumerate(test_dataloader), 
                    total=len(test_dataloader), 
                    desc="Hang on, Running predictions...")
    all_preds = []
    for i, data in test_bar:
        model.zero_grad()
        with torch.no_grad(): outputs = model(**data)
        preds = outputs.argmax(axis=-1).detach().cpu()
        for pred in preds:
            all_preds.append(id2label[pred.item()])
    formatted_preds = []
    for id, pred in zip(test_ids, all_preds):
        formatted_preds.append({
            "image": id,
            "answer": pred
        })

    if args.pred_file == None:
        args.pred_file = os.path.join("experiments", args.exp_name, "test_logs.json")
    with open(args.pred_file, "w") as fn:
        fn.write(json.dumps(formatted_preds, indent=4))

def predict_clip(args) -> str:
    model_path = os.path.join("experiments", args.exp_name, "best_model.pt")
    label2id = json.load(open("experiments/vizwiz_class_vocab.json"))
    id2label = {v: k for k,v in label2id.items()}
    
    if not(args.use_hf_clip):
        model = SkillAwareCLIPVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, 
            device=args.device,
            no_skill_mode=args.no_skill,
            no_obj_mode=args.no_obj,
            no_scene_text_mode=args.no_scene_text,
            norm_features=args.norm_features,
        )
    else:
        model = SkillAwareHfCLIPVizWizVQA(
            model_path=args.model_path, 
            label2id=label2id, 
            device=args.device,
            no_skill_mode=args.no_skill,
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
    if not(args.use_hf_clip):
        val_dataset = SkillAwareCLIPVizWizVQABestAnsDataset(
            annot_path="data/VQA/val.json", images_dir="./data/VQA/val", 
            label2id=label2id, preprocess=model.preprocess,
            aux_tokens_data="./data/VQA/val_aux_info_tokens.jsonl",
        )
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_dataset = SkillAwareHfCLIPVizWizVQADataset(
            annot_path="data/VQA/val.json", images_dir="./data/VQA/val", 
            label2id=label2id, aux_tokens_data="./data/VQA/val_aux_info_tokens.jsonl",
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, 
            shuffle=False, collate_fn=val_dataset.collate_fn,
        )
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
    parser.add_argument("-d", "--data_dir", default="./data/VQA", type=str, help="path to the VQA dataset")
    parser.add_argument("-hf", "--use_hf_clip", action="store_true", help="use huggingface CLIP implementation")
    parser.add_argument("-bm", "--base_model", default="clip", type=str, help="name/path of the CLIP checkpoint")
    parser.add_argument("-mp", "--model_path", default="ViT-L/14", type=str, help="name/path of the CLIP model")
    parser.add_argument("-t", "--train", action="store_true", help="finetune CLIP model")
    parser.add_argument("-p", "--predict", action="store_true", help="predict using CLIP model")
    parser.add_argument("-ges", "--get_eval_scores", action="store_true", help='compute metrics from dumped predictions')
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-ls", "--log_steps", default=10, type=int, help="number of steps after which train stats are logged")
    parser.add_argument("-es", "--eval_steps", default=100, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-exp", "--exp_name", type=str, help="name of the experiment", required=True)
    parser.add_argument("-de", "--device", default="cuda:0", type=str, help="device to be used for training, defaults to cuda:0")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size to be used for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("-pfn", "--pred_file", type=str, help="file to store predictions")
    parser.add_argument("-ska", "--use_skill_attention", action="store_true", help="use self attention fusion")
    parser.add_argument("-nsk", "--no_skill", action="store_true", help="no skill embedding mode")
    parser.add_argument("-nobj", "--no_obj", action="store_true", help="no object tags")
    parser.add_argument("-nf", "--norm_features", action="store_true", help="apply layer norm on concatenated features")
    parser.add_argument("-nsctxt", "--no_scene_text", action="store_true", help="no scene text")
    parser.add_argument("-pu", "--predict_unseen", action="store_true", help="predict on unseen data without labels")
    args = parser.parse_args()
    if args.use_hf_clip: args.model_path = "openai/clip-vit-large-patch14"

    return args

    
if __name__ == "__main__":
    args = get_args()
    if args.train: train_clip(args)
    elif args.predict: predict_clip(args)
    elif args.predict_unseen:
        predict_unseen_clip(args)
    if args.predict or args.get_eval_scores:
        # compute evaluation metrics.
        get_eval_scores(
            "data/VQA/val.json", 
            f'experiments/{args.exp_name}/pred.json'
        )
    # python -m src.main_model.clip_late_fusion -t -de "cuda:0" -exp skill_unaware_clip
    # python -m src.main_model.clip_late_fusion -t -de "cuda:0" -exp skill_clip_hf_vis_proj2 -hf
    # python -m src.main_model.clip_late_fusion -t -de "cuda:0" -exp skill_clip_hf_vis_proj2 -hf -ska
    # python -m src.main_model.clip_late_fusion -t -de "cuda:0" -exp skill_aware_clip_nobj -nobj
    # python -m src.main_model.clip_late_fusion -t -de "cuda:0" -exp skill_aware_clip_nsctxt -nsctxt