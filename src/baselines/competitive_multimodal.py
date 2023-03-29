# competitive multimodal baselines.
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.datautils import VizWizVQABestAnsDataset
from transformers import AutoProcessor, GitVisionModel, AutoModelForCausalLM

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
    ks = [1,5,10,15,20,25,30,35,40,45,50,100,200,300,400,500,1000,2000,3000,4000,5000]
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
    input_ids = processor(text=question, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    print(f"image: {image}")
    print(f"question: {question}")
    print("answer:", processor.batch_decode(generated_ids, skip_special_tokens=True))