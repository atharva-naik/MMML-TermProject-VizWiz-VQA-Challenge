import os
import json
import easyocr
from typing import *
from tqdm import tqdm

def get_scene_texts(dataset, split_name: str):
    reader = easyocr.Reader(['en'])
    path = f"{split_name}_scene_text_extracted.jsonl"
    open(path, "w")
    for rec in tqdm(dataset):
        result = reader.readtext(rec['image'])
        with open(path, "a") as f:
            f.write(json.dumps({
                "image": rec['image'],
                "scene_text": [i[1] for i in result],
                "scene conf": [i[2] for i in result],
                # "scene boxes": [i[0] for i in result],
            })+"\n")

# main
if __name__ == "__main__":
    from src.datautils import VizWizVQATestDataset
    trainset = VizWizVQATestDataset("./train.json", "./train/")
    testset = VizWizVQATestDataset("./test.json", "./test/")
    valset = VizWizVQATestDataset("./val.json", "./val/")
    get_scene_texts(trainset, "train")
    get_scene_texts(testset, "test")
    get_scene_texts(valset, "val")