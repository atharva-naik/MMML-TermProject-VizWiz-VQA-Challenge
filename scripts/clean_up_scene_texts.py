import os
import sys
import json
import spellchecker
from tqdm import tqdm
from spellchecker import SpellChecker

def read_jsonl(path: str):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "": continue
            data.append(json.loads(line))

    return data

def correct_word(word: str, checker: SpellChecker):
    if len(checker.unknown([word])) == 1:
        corrected_word = checker.correction(word)
        if corrected_word is None: return word
        else: return corrected_word
    return word

# main
if __name__ == "__main__":
    checker = SpellChecker()
    for split in [sys.argv[1]]: #["train"]: # ["val", "test"]: #["train"]
        path = f"data/VQA/{split}_scene_text_extracted.jsonl"
        spell_corrected_path = f"data/VQA/{split}_scene_text_extracted_spell_correction.jsonl"
        data = read_jsonl(path)
        open(spell_corrected_path, "w")
        for rec in tqdm(data, desc=split): 
            rec["scene_text"] = [correct_word(w.lower(), checker) for w in rec["scene_text"]]
            with open(spell_corrected_path, "a") as f:
                f.write(json.dumps(rec)+"\n")