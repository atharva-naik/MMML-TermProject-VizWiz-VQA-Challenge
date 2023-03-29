import json
from tqdm import tqdm
from src.datautils import VizWizVQABestAnsDataset

all_classes = json.load(open("./experiments/vizwiz_class_vocab.json"))
train = VizWizVQABestAnsDataset("./data/VQA/train.json", "./data/VQA/train")
val = VizWizVQABestAnsDataset("./data/VQA/val.json", "./data/VQA/val")

val_class_ids = set()
train_class_ids = set()
for inst in tqdm(train):
    train_class_ids.add(all_classes[inst["answer"]])
for inst in tqdm(val):
    val_class_ids.add(all_classes[inst["answer"]])

common_class_inst_ctr = 0
diff_class_inst_ctr = 0
common_class_ids = val_class_ids.intersection(train_class_ids)
for inst in tqdm(val):
    if all_classes[inst["answer"]] in common_class_ids:
        common_class_inst_ctr += 1
    else: diff_class_inst_ctr += 1

print(f"common classes: {len(common_class_ids)}")
print(f"instances in val with common classes: {common_class_inst_ctr} ({(100*common_class_inst_ctr/len(val)):.2f}%)")
print(f"instances in val with diff classes: {diff_class_inst_ctr} ({(100*diff_class_inst_ctr/len(val)):.2f}%)")
print(f"classes seen in train only: {len(train_class_ids.difference(val_class_ids))}")
print(f"classes seen in val only: {len(val_class_ids.difference(train_class_ids))}")