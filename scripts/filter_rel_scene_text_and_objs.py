# script to use spacy's parsing of the questions in the data to filter 
# out relevant scene text and object tag tokens wherever possible.

import os
import json
import spacy
import fuzzywuzzy
from tqdm import tqdm
from fuzzywuzzy import fuzz

def read_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "": continue
            data.append(json.loads(line))

    return data

# main
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    # do_scene_text: bool=False
    # do_obj_tags: bool=True
    for split in ["train", "val"]:
        scene_text = read_jsonl(f"./data/VQA/{split}_scene_text_extracted_spell_correction.jsonl")
        obj_tags = read_jsonl(f"./data/VQA/o365/object_detections_{split}.jsonl")
        data = json.load(open(f"./data/VQA/{split}.json"))
        skill_data = json.load(open(f"./experiments/vilt_skill_clf/{split}_unseen_preds.json"))
        write_path = f"./data/VQA/o365/{split}_aux_info_tokens.jsonl"
        open(write_path, "w")
        for ind, (scene_rec, object_rec, rec, skills) in tqdm(enumerate(zip(scene_text, obj_tags, data, skill_data)), total=len(data)):
            skills = skills["pred"]
            compacted_scene_text = []
            compacted_scene_conf = []
            compacted_obj_tags = []
            compacted_obj_conf = []
            for score, term in zip(scene_rec["scene conf"], scene_rec["scene_text"]):
                fuzz_score = fuzz.token_set_ratio(term, rec["question"])
                if fuzz_score >= 75:
                    compacted_scene_text.append(term)
                    compacted_scene_conf.append(score)
            for score, term in zip(object_rec["scores"], object_rec["labels"]):
                fuzz_score = fuzz.token_set_ratio(term, rec["question"])
                if fuzz_score >= 75:
                    compacted_obj_tags.append(term)
                    compacted_obj_conf.append(score)
            if len(compacted_scene_text) == 0:
                compacted_scene_text = scene_rec["scene_text"]
                compacted_scene_conf = scene_rec["scene conf"] 
            if len(compacted_obj_tags) == 0:
                compacted_obj_tags = object_rec["labels"]
                compacted_obj_conf = object_rec["scores"]
            # print("skills: ", ", ".join(skills)) 
            # print("scene text:", " ".join(compacted_scene_text))
            # print("object tags:", ", ".join(compacted_obj_tags))
            write_rec = {
                "skills": skills,
                "ocr": [compacted_scene_text, compacted_scene_conf],
                "objects": [compacted_obj_tags, compacted_obj_conf],
            }
            # if ind == 5: break
            with open(write_path, "a") as f:
                f.write(json.dumps(write_rec)+"\n")