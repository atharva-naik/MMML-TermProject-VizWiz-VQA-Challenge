import cv2
from typing import *
from object_detection.coco_classes import COCO_CLASSES, COCO_COLORS

def visualize_detections(orig, detections: Dict[str, Any], save_path: str, 
                         confidence_thresh: float=0.7, 
                         classes: List[str]=COCO_CLASSES,
                         COLORS: List[str]=COCO_COLORS):
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confidence_thresh:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            # print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # show the output image
    cv2.imwrite(save_path, orig)