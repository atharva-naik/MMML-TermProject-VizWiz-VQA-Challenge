import os
import argparse
import json 

import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor, DetaForObjectDetection
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from src.datautils import VizWizVQAObjDetectDataset, post_proc_detections
from src.object_detection.visualize import visualize_detections
from src.object_detection.coco_classes import COCO_CLASSES, COCO_COLORS
from src.analysis import analyze_object_detections

def deta_inference(model_ckpt, image_dir, save_dir=None, 
        device='cuda', results_dir=None, confidence_thresh=.5, 
        batch_sz = 3, split="val"):
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = DetaForObjectDetection.from_pretrained(model_ckpt)

    data = VizWizVQAObjDetectDataset(image_dir)
    model.to(device)
    model.eval()
    dataloader = DataLoader(data, batch_size=batch_sz, shuffle=False)
    classes = model.config.id2label
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if save_dir is None:
        save_dir = os.getcwd()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if results_dir is None:
        results_dir = os.getcwd()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    j = 0

    with open(os.path.join(results_dir, f"object_detections_{split}.json"), "w") as f:
        with torch.no_grad():
            pbar = tqdm(dataloader, 
                    desc="detecting val set images")
            for img_paths in pbar:
                
                images = [Image.open(img_path) for img_path in img_paths]
                img_names = [img_path.split("/")[-1] for img_path in img_paths]
                save_pths = [os.path.join(save_dir, img_name) for img_name in img_names]
                # print(save_pths)
                inputs = image_processor(images, return_tensors="pt").to(device)
                outputs = model(**inputs)
    
                target_sizes = [torch.tensor(image.size[::-1]) for image in images]
                results = image_processor.post_process_object_detection(outputs, 
                    threshold=confidence_thresh, target_sizes=target_sizes)
                # print(results)
                if j < 100: 
                    for i in range(len(images)):
                        visualize_detections(cv2.imread(img_paths[i]), results[i], classes=classes, 
                        COLORS=colors, save_path=save_pths[i], confidence_thresh=confidence_thresh)
               
                for k, result in enumerate(results):
                    print_result = {k: v.detach().cpu().tolist() for k,v in result.items()}
                    print_result["labels"] = [classes[i] for i in print_result["labels"]]
                    print_result["selected"] = [(score > confidence_thresh) for score in print_result["scores"]]
                    print_result["image"] = img_names[k]
                    # save the results.
                    f.write(json.dumps(print_result) + "\n")

                j += 1

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="jozhang97/deta-swin-large-o365")
    parser.add_argument("--image_dir", type=str, default="data/VQA/train")
    parser.add_argument("--save_dir", type=str, default="data/VQA/train_obj_detect/o365")
    parser.add_argument("--results_dir", type=str, default="data/VQA/")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--confidence_thresh", type=float, default=.35)
    parser.add_argument("--batch_sz", type=int, default=3)

    args = parser.parse_args()
    deta_inference(args.model_ckpt, args.image_dir, args.save_dir, device=args.device, 
        results_dir=args.results_dir, confidence_thresh=args.confidence_thresh, batch_sz=args.batch_sz, 
        split=args.split)
    analyze_object_detections(os.path.join(args.results_dir,f"object_detections_{args.split}.json"), args.split)


