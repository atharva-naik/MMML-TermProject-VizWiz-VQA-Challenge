import json

# main
if __name__ == "__main__":
    atype_preds = [d["pred"] for d in json.load(open("./experiments/vilt_atype_clf2/test_unseen_preds.json"))]
    clip_preds = [d["answer"] for d in json.load(open("./experiments/CLIP_test.json"))]
    vilt_preds = [d["answer"] for d in json.load(open("./experiments/frozen_vilt2/test_logs.json"))]
    ids = [d["image"] for d in json.load(open("./experiments/frozen_vilt2/test_logs.json"))]
    combined_preds = []
    for atype, clip_pred, vilt_pred, id in zip(atype_preds, clip_preds, vilt_preds, ids):
        if atype == "yes/no": 
            pred = vilt_pred
        else: pred = clip_pred
        combined_preds.append({
            "image": id,
            "answer": pred,
        })
    with open("./experiments/CLIP_ViLT_ViLT_atype_test_preds.json", "w") as f:
        json.dump(combined_preds, f, indent=4)
