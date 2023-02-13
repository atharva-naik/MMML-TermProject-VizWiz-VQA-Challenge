# code to load and analyze the data.
import os
import cv2
import clip
import json
import torch
import pickle
import pathlib
import argparse
import numpy as np
from typing import *
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from collections import defaultdict
from torchvision.models import detection
import src.object_detection.utils as utils
# import src.object_detection.transforms as T
from torch.utils.data import Dataset, DataLoader
from src.PythonHelperTools.vqaTools.vqa import VQA
from src.object_detection.coco_classes import COCO_CLASSES
from src.object_detection.visualize import visualize_detections

def get_transform(train: bool=False):
    """
    apply transforms (and data augmentation for train).
    borrowed from here: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
    """
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train: transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

# only for test set (no answers available)
class VizWizVQATestDataset(Dataset):
    def __init__(self, annot_path: str, images_dir: str):
        super(VizWizVQATestDataset, self).__init__()
        self.annot_path = annot_path
        self.images_dir = images_dir
        self.annot_db = VQA(annotation_file=annot_path)
        self.data = []
        for rec in self.annot_db.getAnns():
            img_path = os.path.join(images_dir, 
                                    rec['image'])
            q = rec['question']
            self.data.append({
                "question": q,
                "image": img_path,
            })
    def __len__(self): return len(self.data) 
    def __getitem__(self, i: int): return self.data[i]

# only best answer considered.
class VizWizVQABestAnsDataset(Dataset):
    def __init__(self, annot_path: str, images_dir: str):
        super(VizWizVQABestAnsDataset, self).__init__()
        self.annot_path = annot_path
        self.images_dir = images_dir
        self.annot_db = VQA(annotation_file=annot_path)
        self.data = []
        self.a_type_to_ind = {
            'unanswerable': 0, 
            'other': 1, 
            'yes/no': 2, 
            'number': 3,
        }
        for rec in self.annot_db.getAnns():
            a_type = rec['answer_type']
            img_path = os.path.join(images_dir, 
                                    rec['image'])
            q = rec['question']
            is_ansble = rec["answerable"]
            answer, answer_confidence, total = self._vote_answer(
                rec['answers'], is_ansble=is_ansble,
            )
            self.data.append({
                "question": q, "is_answerable": is_ansble,
                "answer_type": self.a_type_to_ind[a_type],
                "answer_confidence": answer_confidence,
                "image": img_path, "answer": answer,
                "total": total,
            })
    def __len__(self): return len(self.data) 
    def __getitem__(self, i: int): return self.data[i]         
    def _vote_answer(self, answers: List[str], 
                     is_ansble: bool=True) -> Tuple[str, float, int]:
        answer_votes = defaultdict(lambda:0)
        get_ans_conf = {"yes": 1, "maybe": 0.5, "no": 0}
        tot = 0
        for ans in answers:
            if is_ansble and ans['answer'] == "unanswerable": continue 
            ans_conf = ans['answer_confidence']
            answer_votes[ans['answer']] += get_ans_conf[ans_conf]
            tot += get_ans_conf[ans_conf]
        i = np.argmax(list(answer_votes.values()))
        picked_ans = list(answer_votes.keys())[i]
        picked_ans_conf = list(answer_votes.values())[i]/tot
        
        return picked_ans, picked_ans_conf, tot

# all unique answers considered.
class VizWizVQAUniqueAnsDataset(Dataset):
    def __init__(self, annot_path: str, images_dir: str):
        super(VizWizVQAUniqueAnsDataset, self).__init__()
        self.annot_path = annot_path
        self.images_dir = images_dir
        self.annot_db = VQA(annotation_file=annot_path)
        self.data = []
        self.a_type_to_ind = {
            'unanswerable': 0, 
            'other': 1, 
            'yes/no': 2, 
            'number': 3,
        }
        for rec in self.annot_db.getAnns():
            a_type = rec['answer_type']
            img_path = os.path.join(images_dir, 
                                    rec['image'])
            q = rec['question']
            is_ansble = rec["answerable"]
            uniq_ans, tot = self._get_unique_answers(rec["answers"])
            for ans, conf in uniq_ans.items():
                self.data.append({
                    "question": q, "is_answerable": is_ansble,
                    "answer_type": self.a_type_to_ind[a_type],
                    "answer_confidence": conf, "total": tot,
                    "image": img_path, "answer": ans,
                
                })
    def __len__(self): return len(self.data) 
    def __getitem__(self, i: int): return self.data[i]         
    def _get_unique_answers(self, answers: List[str]) -> Tuple[Dict[str, float], int]:
        uniq_ans = defaultdict(lambda:0)
        get_ans_conf = {"yes": 1, "maybe": 0.5, "no": 0}
        tot = 0
        for ans in answers:
            ans_conf = ans['answer_confidence']
            uniq_ans[ans['answer']] += get_ans_conf[ans_conf]
            tot += get_ans_conf[ans_conf]
        for ans, conf in uniq_ans.items(): uniq_ans[ans] = conf/tot
        
        return uniq_ans, tot

# VizWiz QA dataset for using CLIP.
class VizWizVQAClipDataset(Dataset):
    def __init__(self, annot_path: str, images_dir: str, 
                 image_preprocess_fn=None, 
                 qa_pair_type: str="best", 
                 device: str="cuda"):
        super(VizWizVQAClipDataset, self).__init__()
        self.image_preprocess_fn = image_preprocess_fn
        self.qa_pair_type = qa_pair_type
        self.device = device if torch.cuda.is_available() else "cpu"
        if qa_pair_type == "unique":
            self.vqa_dataset = VizWizVQAUniqueAnsDataset(
                annot_path=annot_path,
                images_dir=images_dir,
            )
        elif qa_pair_type == "best":
            self.vqa_dataset = VizWizVQABestAnsDataset(
                annot_path=annot_path,
                images_dir=images_dir,
            )
    
    def __getitem__(self, i: int):
        data = self.vqa_dataset[i]
        img_path = data["image"]
        # question text tokenized.
        # context image tokenized.
        c_image_tok = self.image_preprocess_fn(
            Image.open(img_path)
        ).unsqueeze(0)[0]#.to(self.device)
        q_text_tok = clip.tokenize(data['question'])[0]#.to(self.device)
        a_text_tok = clip.tokenize(data['answer'])[0]#.to(self.device)
        a_type = torch.as_tensor(data["answer_type"])
        a_conf = torch.as_tensor(data["answer_confidence"])
        is_ansble = torch.as_tensor(data["is_answerable"])
        
        return {
            "context_image": c_image_tok, 
            "question_text": q_text_tok, 
            "answer_text": a_text_tok,
            "answer_confidence": a_conf,
            "is_answerable": is_ansble,
            "answer_type": a_type,
        }

# VizWiz QA dataset for object detection.
# code borrowed from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
# class VizWizVQAObjDetectDataset(Dataset):
#     def __init__(self, images_dir: str, transforms):
#         super(VizWizVQAObjDetectDataset, self).__init__()
#         self.images_dir = images_dir
#         self.transforms = transforms
#         self.imgs = list(sorted(os.listdir(images_dir)))
#         # there are no masks (targets) for this dataset.
#     def __len__(self): return len(self.imgs)
#     def __getitem__(self, idx: int):
#         # load images.
#         img_path = os.path.join(self.images_dir, self.imgs[idx])
#         img = Image.open(img_path).convert("RGB")
#         # note that we haven't converted the mask to RGB,
#         # because each color corresponds to a different instance
#         # with 0 being background
#         # mask = Image.open(mask_path)
#         # # convert the PIL Image into a numpy array
#         # mask = np.array(mask)
#         # # instances are encoded as different colors
#         # obj_ids = np.unique(mask)
#         # # first id is the background, so remove it
#         # obj_ids = obj_ids[1:]
#         # # split the color-encoded mask into a set
#         # # of binary masks
#         # masks = mask == obj_ids[:, None, None]
#         # # get bounding box coordinates for each mask
#         # num_objs = len(obj_ids)
#         # boxes = []
#         # for i in range(num_objs):
#         #     pos = np.where(masks[i])
#         #     xmin = np.min(pos[1])
#         #     xmax = np.max(pos[1])
#         #     ymin = np.min(pos[0])
#         #     ymax = np.max(pos[0])
#         #     boxes.append([xmin, ymin, xmax, ymax])
#         # # convert everything into a torch.Tensor
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # # there is only one class
#         # labels = torch.ones((num_objs,), dtype=torch.int64)
#         # masks = torch.as_tensor(masks, dtype=torch.uint8)
#         # image_id = torch.tensor([idx])
#         # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # # suppose all instances are not crowd
#         # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
#         # target = {}
#         # target["boxes"] = boxes
#         # target["labels"] = labels
#         # target["masks"] = masks
#         # target["image_id"] = image_id
#         # target["area"] = area
#         # target["iscrowd"] = iscrowd
#         if self.transforms is not None:
#             # img, target = self.transforms(img, target)
#             img = self.transforms(img)
#         # return img, target
#         return img, img
class VizWizVQAObjDetectDataset(Dataset):
    def __init__(self, images_dir: str):
        super(VizWizVQAObjDetectDataset, self).__init__()
        self.images_dir = images_dir
        # self.transforms = transforms
        self.imgs = [
            os.path.join(images_dir, path) for \
                path in list(sorted(os.listdir(images_dir))
            )
        ]
        # there are no masks (targets) for this dataset.
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx: int):
        img_path = self.imgs[idx]
        image = cv2.imread(img_path) # print(image)
        orig = image.copy()
        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        # send the input to the device and pass the it through the network to
        # get the detections and predictions
        # image = image.to(DEVICE)
        return image, orig, img_path

def get_save_path(parent: str, path: str):
    pobj = pathlib.Path(path)
    stem = pobj.stem

    return os.path.join(parent, f"{stem}_objects.png")

def post_proc_detections(detections: Dict[str, torch.Tensor], 
                         confidence_thresh: float=0.7):
    for k,v in detections.items():
        detections[k] = v.cpu().tolist()
    detections.update({
        "orig_path": img_path,
        "det_save_path": det_save_path,
    })
    detections["labels"] = [COCO_CLASSES[i] for i in detections["labels"]]
    detections["selected"] = [(score > confidence_thresh) for score in detections["scores"]]

    return detections

# main
if __name__ == "__main__":
    import torchvision

    # classes of objects to be detected.
    classes = COCO_CLASSES

    # object detector model.
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn() # (weights="DEFAULT")
    model = detection.fasterrcnn_resnet50_fpn(
        pretrained=True, progress=True,
        pretrained_backbone=True,
        num_classes=len(classes),
        weights='DEFAULT',
        weights_backbone='DEFAULT',
    )

    # move model to device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # load validation object detection dataset and dataloader.
    transforms = get_transform(train=False)
    val_obj_detect_dataset = VizWizVQAObjDetectDataset(
        images_dir='./val/',
        # transforms=transforms,
    )
    # val_obj_detect_dataloader = DataLoader(
    #     val_obj_detect_dataset, batch_size=20,
    #     shuffle=False, # num_workers=2,
    #     # collate_fn=utils.collate_fn,
    # )

    # freeze the model for inference.
    model.eval()
    
    pbar = tqdm(val_obj_detect_dataset, 
                desc="encoding val set images")
    # directory for saving object detections.
    DETECTION_SAVE_PATH = "val_objects_detected"
    os.makedirs(DETECTION_SAVE_PATH, exist_ok=True)
    detected_objects = []
    for image, orig, img_path in pbar:
        # images = list(image.to(device) for image in batch[0])
        # origs = list(orig for orig in batch[1])
        image = image.to(device)
        with torch.no_grad():
            det_save_path = get_save_path(
                DETECTION_SAVE_PATH, 
                img_path,
            )
            detections = model(image)[0]
            # print(detections.keys())
            # print(det_save_path)
            visualize_detections(
                orig, detections=detections,
                save_path=det_save_path,
            )
            detected_objects.append(post_proc_detections(detections))
            # break
    detections_json_path = os.path.join(
        DETECTION_SAVE_PATH, 
        "all_detection_labels.json",
    )
    with open(detections_json_path, "w") as f:
        json.dump(detected_objects, f, indent=4)