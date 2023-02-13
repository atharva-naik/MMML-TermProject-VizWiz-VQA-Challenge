import os
import numpy as np

COCO_CLASSES_PATH = os.path.join(
    os.getcwd(), "./src/object_detection/coco_classes.lst"
)
COCO_CLASSES = [class_.strip() for class_ in open(COCO_CLASSES_PATH, 'r').readlines()]
COCO_COLORS = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))