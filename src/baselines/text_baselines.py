# Author: Navaneethan Vaikunthan

import argparse
from collections import Counter
import torch 
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, AutoTokenizer, get_scheduler
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import wandb

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datautils import VizWizVQABestAnsDataset
from PythonHelperTools.vqaTools.vqa import VQA

def preprocessing_func_seq2seq(examples, tokenizer, model):
    model_inputs = tokenizer(examples["question"], padding=True, 
                             truncation=True, max_length=512, return_tensors="pt")
    labels = tokenizer(text_target=examples["answer"], max_length=128, 
                       truncation=True, padding=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    model_inputs["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
    return model_inputs

def preprocessing_func_class(examples, tokenizer):
    model_inputs = tokenizer(examples["question"], padding=True, 
                             truncation=True, max_length=512, return_tensors="pt")
    model_inputs["labels"] = examples["class_label"]
    return model_inputs


def train_unifiedqa_base(args):
    # Intitialize Model and Tokenizer
    torch.manual_seed(42)
    model_ckpt = args.model
    device = args.device

    model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model.to(device)
    # Create Data Loaders
    batch_size, num_epochs = int(args.batch_sz), int(args.num_epochs)

    train_dataset = VizWizVQABestAnsDataset(annot_path="data/VQA/train.json", images_dir="data/VQA/train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = VizWizVQABestAnsDataset(annot_path="data/VQA/val.json", images_dir="data/VQA/val")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_steps = num_epochs * len(train_dataloader)

    # Intialize Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4)
    lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=int(train_steps/20), num_training_steps=train_steps
)

    # Setup WandB Tracking
    """run = wandb.init(
        project="test-unifiedqa",
    )
    wandb.config = {
        "epochs": num_epochs, 
        "learning_rate": 5e-4, 
        "batch_size": batch_size   
    }"""

    for i in tqdm(range(num_epochs)):
        model.train()
        for i, train_batch in tqdm((train_dataloader)):
            tokenized_batch = preprocessing_func_seq2seq(train_batch, tokenizer, model)
            tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
            outputs = model(**tokenized_batch)
            train_loss = outputs.loss
            train_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            #wandb.log({"training_loss": train_loss})

            if i > 30: 
                break

        model.eval()
        device = "cpu" if device == "mps" else device
        model.to(device)
        eval_loss = []
        with torch.no_grad():
            for i, eval_batch in tqdm((val_dataloader)):
                tokenized_batch = preprocessing_func_seq2seq(eval_batch, tokenizer, model)
                tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
                outputs = model(**tokenized_batch)
                generations = model.generate(tokenized_batch["input_ids"], 
                                            attention_mask=tokenized_batch["attention_mask"], 
                                            num_beams=4, max_new_tokens=5, early_stopping=True)
                eval_loss.append(outputs.loss.cpu())
                proxy_accuracy(eval_batch, generations, tokenizer)

                if i > 5:
                    break
        print(eval_loss)
    return model

        #wandb.log({"eval_loss": sum(eval_loss) / len(eval_loss)})
                            
def proxy_accuracy(inputs, generations, tokenizer):
    decoded_preds = tokenizer.batch_decode(generations, skip_special_tokens=True)
    answers = inputs["answer"]
    for i, answer in enumerate(answers):
        print(f"Question: {inputs['question'][i]}")
        print(f"Prediction: {decoded_preds[i]}")
        print(f"Answer: {answer}")

def class_accuracy(inputs, outputs, id2label):

    pred_words = get_class_preds(outputs, id2label)
    print(pred_words)
    print(inputs["answer"])
    # print((pred_words == inputs["answer"]).sum() / len(pred_words))

def get_class_preds(outputs, id2label):
    preds = outputs.logits.argmax(dim=-1)
    pred_words = [id2label[pred.cpu().item()] for pred in preds]
    return pred_words

def train_text_classifier(args):
    # Get DataLoaders and Extract Relevant Label Information
    batch_size, num_epochs = int(args.batch_sz), int(args.num_epochs)

    train_dataset = VizWizVQABestAnsDataset(annot_path="data/VQA/train.json", images_dir="data/VQA/train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_steps = num_epochs * len(train_dataloader)

    val_dataset = VizWizVQABestAnsDataset(annot_path="data/VQA/val.json", images_dir="data/VQA/val")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Intitialize Model and Tokenizer
    torch.manual_seed(42)
    model_ckpt = args.model
    device = args.device

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(train_dataset.id2label), 
                                                               id2label=train_dataset.id2label, label2id=train_dataset.label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model.to(device)   
    
    # Intialize Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-4)
    lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=train_steps
)
    weights = train_dataset.class_weights.to(device)
    print(weights)

    ce_loss = torch.nn.CrossEntropyLoss(weight=weights, reduction="mean")
    for i in tqdm(range(num_epochs)):
        model.train()
        for i, train_batch in tqdm(enumerate(train_dataloader)):
            tokenized_batch = preprocessing_func_class(train_batch, tokenizer)
            tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
            outputs = model(**tokenized_batch)
            logits = outputs.logits
            train_loss = ce_loss(logits, tokenized_batch["labels"]) 
            train_loss.backward()
            print(train_loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if i > 30: break

        model.eval()
        device = "cpu" if device == "mps" else device
        model.to(device)
        eval_loss = []
        with torch.no_grad():
            for i, val_batch in tqdm(enumerate(val_dataloader)):
                tokenized_batch = preprocessing_func_class(val_batch, tokenizer)
                tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
                print(tokenized_batch["labels"])
                print(device)
                outputs = model(**tokenized_batch)
                eval_loss.append(outputs.loss.cpu())
                class_accuracy(val_batch, outputs, val_dataset.id2label)
                if i > 35: break

        return model
        
def predict_text_only(args, model=None):
    # Get DataLoaders and Extract Relevant Label Information
    test_dataset = VizWizVQABestAnsDataset(annot_path="data/VQA/val.json", images_dir="data/VQA/val")
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    num_classes = int(args.num_classes)
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_classes,
                                                                id2label=test_dataset.id2label, 
                                                                label2id=test_dataset.label2id)    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    device = "cpu" if args.device == "mps" else args.device
    model.to(device)
    results = []
    with torch.no_grad():
        for i, test_batch in tqdm(enumerate(test_dataloader)):
            tokenized_batch = preprocessing_func_class(test_batch, tokenizer)
            tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
            outputs = model(**tokenized_batch)
            image = [img for img in test_batch["image"]]
            answer = get_class_preds(outputs, test_dataset.id2label)
            result = [{"image": img, "answer": ans} for img, ans in zip(image, answer)]
            results.extend(result)
            if i > 10:
                break
    
def get_class_dicts(num_classes=6540):
    train_annot_path = "data/VQA/train.json"
    val_annot_path = "data/VQA/val.json"
    vqa_train = VQA(train_annot_path)
    train_objs = vqa_train.getAnns()
    all_answers = []
    for item in train_objs:
        answer_text = [ans_dict['answer'] for ans_dict in item["answers"]]
        all_answers.extend(answer_text)
    vqa_val = VQA(val_annot_path)
    val_objs = vqa_val.getAnns()

    for item in val_objs:
        answer_text = [ans_dict['answer'] for ans_dict in item["answers"]]
        all_answers.extend(answer_text)

    answer_freq = Counter(all_answers)
    common_answers = answer_freq.most_common(num_classes)
    label_to_id = {}
    id_to_label = {}

    for i, item in enumerate(common_answers):
        label_to_id[item[0]] = i
        id_to_label[i] = item[0]

    return label_to_id, id_to_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model Checkpoint", 
                        default="google/flan-t5-base", required=False)
    parser.add_argument('--batch_sz', type=str, help="Batch Size", default="4", required=False)
    parser.add_argument('--num_epochs', type=str, help="Number of Epochs", default="1", required=False)
    parser.add_argument('--device', type=str, help="Hardware Accelerator", default="cuda", required=True)
    parser.add_argument('--baseline_type', type=str, help="Which Text Baseline", default="QA", required=False)
    parser.add_argument('--num_classes', type=str, help="Number of Classes", default="6540", required=False)


    arguments = parser.parse_args()
    # get_class_dicts()

    
    if arguments.baseline_type == 'QA':
       model = train_unifiedqa_base(arguments)
    else: 
       model = train_text_classifier(arguments)
    predict_text_only(arguments, model)
    
    

