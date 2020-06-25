from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import os

# Using pre-trained parameters: eval mode immediately
# ResNet50 with Feature Pyramid Network backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Faster RCNN was pre-trained on MS-COCO
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    """

    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Coordinates
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class, pred_score


def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    """
      object_detection_api
        parameters:
          - img_path - path of the input image
          - threshold - threshold value for prediction score
          - rect_th - thickness of bounding box
          - text_size - size of the class label text
          - text_th - thickness of the text
        method:
          - prediction is obtained from get_prediction method
          - for each prediction, bounding box is drawn and text is written with opencv
          - the final image is displayed
    """

    # Returns bounding boxes of each object instance; and the predicted class each belongs to
    boxes, pred_cls, pred_score = get_prediction(img_path, threshold)
    img = plt.imread(img_path)  # Plt seems to be more robust than cv2.imread
    # img = cv2.imread(img_path, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        print(boxes[i][0], "\t", boxes[i][1])
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])), color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i] + ": %.3f" % (pred_score[i]), (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    # plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    head, tail = os.path.split(img_path)
    plt.savefig(str(tail)[:-4] + "_faster_rcnn.png")
    # plt.show()


if __name__ == "__main__":
    # object_detection_api('./people.jpg', threshold=0.8)

    # # COCO val2014 -- should be extremely accurate
    # dir_name = "/Users/leonard/Desktop/coco/images/val2014"
    # COCO_dir = os.fsencode(dir_name)
    # for file in os.listdir(COCO_dir):
    #     filename = os.fsdecode(file)
    #     if filename.endswith(".jpg"):
    #         print(os.path.join(dir_name, filename))
    #         object_detection_api(os.path.join(dir_name, filename), threshold=0.8)
    #         continue
    #     else:
    #         continue

    # TDW initial 50 scenes
    dir_name = "/Users/leonard/Desktop/coco/images/to_replicate/finished_replicating"
    for path in Path(dir_name).rglob('*.png'):
        print(path)
        object_detection_api(path, threshold=0.5)

    # with os.scandir(dir_name) as it:
    #     for entry in it:
    #         print(entry)
    #         if entry.name.endswith(".png") and entry.is_file():
    #             object_detection_api(os.path.join(dir_name, entry), threshold=0.8)
