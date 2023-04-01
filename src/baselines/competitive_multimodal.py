# competitive multimodal baselines.
# Authors: Atharva Naik, Yash Butala

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
import matplotlib.pyplot as plt
from src.metrics import proxy_accuracy
from torch.utils.data import DataLoader
from src.datautils import VizWizVQABestAnsDataset
from transformers import AutoProcessor, AutoModelForCausalLM, GitForCausalLM

# seed everything
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)

def test_GIT_zero_shot():
    dataset = VizWizVQABestAnsDataset(
        "./data/VQA/val.json",
        "./data/VQA/val",
    )
    processor = AutoProcessor.from_pretrained("microsoft/git-large-vqav2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vqav2")
    model.cuda()

    preds = []
    for item in tqdm(dataset):
        image_path = item["image"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.cuda()
        question = item["question"] # "what does the front of the bus say at the top?"
        answer = item["answer"]

        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

        generated_ids = model.generate(
            pixel_values=pixel_values, 
            input_ids=input_ids, 
            max_length=50
        )
        gen_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred = gen_text[len(question):].strip()
        pred = "unaswerable" if pred == "" else pred
        preds.append({'true': answer, "pred": pred})
        print(pred , "|", answer)
    preds = {"preds": preds}
    with open("./experiments/git-large-zero-shot-preds.json", "w") as f:
        json.dump(preds, f, indent=4)

def clip_score_recall_analysis(cos_scores_path: str, 
                               data_path: str="./data/VQA", 
                               split: str="val"):
    dataset = VizWizVQABestAnsDataset(
        os.path.join(data_path, f"{split}.json"),
        os.path.join(data_path, split),
    )   
    all_classes = json.load(open("experiments/vizwiz_class_vocab.json"))
    true_indices = [all_classes[ans["answer"]] for ans in dataset] 
    cos_scores = torch.load(cos_scores_path, map_location="cpu")
    recall_at_k = {}
    ks = [1,5,10,15,20,25,30,35,40,45,50,100,200,300,400,500,1000,2000,3000,4000,5000,6000,7000]
    for k in ks:
        topk_inds = torch.topk(cos_scores, k=k).indices.tolist()
        recall_at_k[k] = 0
        for i in range(len(true_indices)):
            if true_indices[i] in topk_inds[i]:
                recall_at_k[k] += 1
        recall_at_k[k] /= len(true_indices)
    plt.clf()
    # print(recall_at_k)
    y = list(recall_at_k.values())
    x = range(len(ks))
    plt.plot(x, y)
    plt.xlabel("k")
    plt.ylabel("recall@k")
    plt.title("CLIP score recalls vs k")
    plt.xticks(x, labels=ks, rotation=45)
    plt.tight_layout()
    plt.savefig("./experiments/clip_cos_scores_vizwiz_recall.png")

    return recall_at_k

def test_CLIP_zero_shot():
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
    dataloader = DataLoader(dataset)
    all_classes = json.load(open("experiments/vizwiz_class_vocab.json"))
    true_indices = []
    all_answer_texts = list(all_classes.keys())
    ans_emb = model.encode(all_answer_texts, show_progress_bar=True, convert_to_tensor=True)
    # Encode all images:
    img_emb = []
    for batch in tqdm(dataset):
        true_indices += [all_classes[ans] for ans in batch["answer"]]
        img_emb.append(model.encode([Image.open(img) for img in batch["image"]], convert_to_tensor=True))
    img_emb = torch.stack(img_emb)
    # compute cosine similarities 
    cos_scores = util.cos_sim(img_emb, ans_emb)
    # print(cos_scores)
    return cos_scores, true_indices

class CLIPEnsemble(nn.Module):
    """CLIP Ensemble model based on this
    arxiv paper: https://arxiv.org/pdf/2206.05281.pdf"""
    def __init__(self, num_classes: int, answer_types: int):
        super().__init__()
        self.num_classes = num_classes

class GITDataLoader(DataLoader):
    def __init__(self, dataset, split: str="train", 
                 model_path: str="microsoft/git-large-vqav2", 
                 **kwargs):
        if split == "train": 
            super().__init__(dataset, shuffle=True, 
                             collate_fn=self.custom_collate_fn, **kwargs)
        else: super().__init__(dataset, shuffle=False, 
                               collate_fn=self.custom_collate_fn, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def custom_collate_fn(self, batch):
        images = [Image.open(item["image"]) for item in batch]
        input_ids = []
        labels = []
        pad_len = 0

        for item in batch:
            q_iids = self.processor(text=item["question"], add_special_tokens=False).input_ids
            a_iids = self.processor(text=item["answer"], add_special_tokens=False).input_ids
            iid = [101]+(q_iids+a_iids)[:510]+[102]
            label = [101]+([-100 for _ in q_iids]+a_iids)[:510]+[102]
            input_ids.append(iid)
            labels.append(label)
            pad_len = max(pad_len, len(iid))
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        for iid, label in zip(input_ids, labels):
            attention_mask.append([0]*len(iid)+[1]*(pad_len-len(iid))) # padding on the left
            padded_input_ids.append([0]*(pad_len-len(iid))+iid) # padding on the left
            padded_labels.append([-100]*(pad_len-len(iid))+label) # padding on the left
        pixel_values = self.processor(
            images=images, return_tensors="pt", 
            padding="longest", truncation=True,
        ).pixel_values
        enc_batch = {}
        enc_batch["pixel_values"] = pixel_values
        enc_batch["input_ids"] = torch.as_tensor(padded_input_ids) 
        enc_batch["attention_mask"] = torch.as_tensor(attention_mask)
        enc_batch["labels"] = torch.as_tensor(padded_labels)
        
        return enc_batch

class FrozenGITVizWizVQA(nn.Module):
    def __init__(self, model_path: str="microsoft/git-large-vqav2",
                 device: str="cuda:0", freeze_layers: List[str]=[
                    "embeddings", "image_encoder", "encoder"
                 ]):
        super().__init__()
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # GIT layers to be frozen:
        for layer in freeze_layers:
            for p in getattr(self.model.git, layer).parameters():
                p.requires_grad = True
        # hard coded right now:
        # git.visual_projection is trainable.
        # the final output linear projection layer (over the vocab) is trainable.
        self.device = device
        self.to(device)
        print(f"moving model to device: {device}")

    def parameters(self):
        return list(self.model.git.visual_projection.parameters())+list(self.model.output.parameters())

    def train(self):
        self.model.git.visual_projection.train()
        self.model.output.train()

    def eval(self):
        self.model.git.visual_projection.eval()
        self.model.output.eval()

    def forward(self, **encoding):
        """return outputs and loss"""
        return self.model(**encoding)

def test_git_large():
    dataset = VizWizVQABestAnsDataset(
        "./data/VQA/val.json",
        "./data/VQA/val"
    )
    processor = AutoProcessor.from_pretrained("microsoft/git-large-vqav2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vqav2")
    image = Image.open(dataset[0]["image"])
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    question = dataset[0]["question"]
    answer = dataset[0]["answer"]
    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    print(f"image: {image}")
    print(f"question: {question}")
    print(f"GT answer: {answer}")
    print("answer:", processor.batch_decode(generated_ids, skip_special_tokens=True))

def save_current_model(model, optimizer, epoch: int, step: int, save_path: str):
    """save the current model and optimizer state"""
    ckpt_dict = {}
    ckpt_dict["model_state_dict"] = model.state_dict()
    ckpt_dict["optim_state_dict"] = optimizer.state_dict()
    ckpt_dict["epoch"] = epoch
    ckpt_dict["step"] = step
    torch.save(ckpt_dict, save_path)

def validate_git(model, dataloader: DataLoader, eval_step: int, 
                 epoch: int, args: argparse.Namespace) -> Union[Tuple[float, float], Tuple[List[dict], float, float]]:
    val_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="validating")
    tot, matches, val_losses = 0, 0, []
    val_preds = []
    model.eval()
    for step, batch in val_bar:
        model.zero_grad()
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].to(args.device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            # if output_preds:
            #     for pred, label in zip(preds.tolist(), labels.tolist()):
            #         val_preds.append({
            #             "pred": model.id2label[pred],
            #             "true": model.id2label[label],
            #         })
            val_losses.append(loss.item())
            val_bar.set_description(f"V: {epoch+1}/{args.epochs} ({eval_step}) bl: {loss.item():.3f} l: {np.mean(val_losses):.3f}")
    # if output_preds: 
    #     return val_preds, np.mean(val_losses), matches/tot
    else: return np.mean(val_losses)

def predict_git(args):
    data_dir = args.data_dir
    model_path = args.model_path
    device = args.device
    val_images = os.path.join(data_dir, "val")
    val_annot = os.path.join(data_dir, "val.json") 
    # declare model.
    model = FrozenGITVizWizVQA(
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
    # batch_size = args.batch_size
    # data = VizWizVQABestAnsDataset(val_annot, val_images)
    # data_loader = GITDataLoader(dataset=data, batch_size=batch_size, split="val")
    # predict_log_path = os.path.join("experiments", args.exp_name, "predict_logs.json")
    
    # model.eval()
    # val_preds, val_loss, val_acc = validate_git(
    #     model=model, dataloader=data_loader, 
    #     eval_step=0, epoch=0, args=args,
    #     output_preds=True,
    # )
    # with open(predict_log_path, "a") as f:
    #     json.dump({
    #         "predict_acc": val_acc, 
    #         "predict_loss": val_loss,
    #         "preds": val_preds,
    #     }, f)

def finetune_git(args):
    data_dir = args.data_dir
    model_path = args.model_path
    device = args.device
    train_images = os.path.join(data_dir, "train") 
    val_images = os.path.join(data_dir, "val")
    train_annot = os.path.join(data_dir, "train.json") 
    val_annot = os.path.join(data_dir, "val.json") 

    # declare model.
    model = FrozenGITVizWizVQA(
        model_path=model_path, 
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
    train_data = VizWizVQABestAnsDataset(train_annot, train_images)
    val_data = VizWizVQABestAnsDataset(val_annot, val_images)
    train_loader = GITDataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        split="train",
    )
    val_loader = GITDataLoader(
        dataset=val_data, 
        batch_size=batch_size, 
        split="val",
    )
    epochs = args.epochs
    
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

    # train-eval loop
    for epoch in range(epochs):
        # reset the train log.
        open(train_log_path.format(epoch), "w") 
        train_bar = tqdm(enumerate(train_loader), 
                         total=len(train_loader), 
                         desc="commencing training")
        train_epoch_losses = []
        for step, batch in train_bar:
            model.train()
            model.zero_grad()
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            loss = outputs.loss

            # backward and optimizer step
            loss.backward()
            optimizer.step()

            logits = outputs.logits.detach().cpu()
            train_epoch_losses.append(loss.item())
            train_bar.set_description(f"T: {epoch+1}/{epochs}: L:{logits.shape} bl: {loss.item():.3f} l: {np.mean(train_epoch_losses):.3f}")

            if step%100 == 0:
                question = val_data[0]["question"]
                image = Image.open(val_data[0]["image"])
                pixel_values = train_loader.processor(images=image, return_tensors="pt").pixel_values.to(device)
                input_ids = train_loader.processor(text=question, return_tensors="pt").input_ids.to(device)
                gen_ids = model.model.generate(
                    pixel_values=pixel_values, 
                    input_ids=input_ids, 
                    max_length=100,
                )
                input_text = train_loader.processor.batch_decode(input_ids, skip_special_tokens=True)[0]
                generated_text = train_loader.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
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
                # save the best model.
                if val_loss < best_val_loss:
                    best_val_step = step
                    best_val_epoch = epoch
                    best_val_loss = val_loss
                    model_save_path
                    # save the checkpoint for model/optimizer
                    save_current_model(model=model, optimizer=optimizer, 
                                       save_path=model_save_path,
                                       epoch=best_val_epoch, 
                                       step=best_val_step)
                with open(val_log_path, "a") as f:
                    f.write(json.dumps({
                        "val_loss": val_loss,
                        "epoch": epoch, "step": step,
                        "best_val_epoch": best_val_epoch,
                        "best_val_step": best_val_step,
                        "best_val_acc": best_val_loss,
                    })+"\n")

def get_args():
    """get terminal arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="./data/VQA", type=str, help="path to the VQA dataset")
    parser.add_argument("-mp", "--model_path", default="microsoft/git-large-vqav2", type=str, help="name/path of the HF checkpoint")
    parser.add_argument("-t", "--train", action="store_true", help="finetune ViLT model")
    parser.add_argument("-p", "--predict", action="store_true", help="generate predictions for data with labels")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-ls", "--log_steps", default=10, type=int, help="number of steps after which train stats are logged")
    parser.add_argument("-es", "--eval_steps", default=2000, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-exp", "--exp_name", type=str, required=True, help="name of the experiment")
    parser.add_argument("-de", "--device", default="cuda:0", type=str, help="device to be used for training, defaults to cuda:0")
    parser.add_argument("-bs", "--batch_size", type=int, default=2, help="batch size to be used for training")
    parser.add_argument("-ckpt", "--checkpoint_path", type=Union[str, None], default=None, help="checkpoint to be loaded")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="learning rate for training")
    args = parser.parse_args()

    return args

# main
if __name__ == "__main__":
    args = get_args()
    if args.train: finetune_git(args)
    elif args.predict: predict_git(args) 
    # python -m src.baselines.competitive_multimodal -t -de 'cuda:0' -exp frozen_git