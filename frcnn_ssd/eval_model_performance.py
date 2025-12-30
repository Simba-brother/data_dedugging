
'''
用于评估FRCNN和SSD模型的mAP
'''
import os
import json
from datasets import CocoDetectionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import torch,torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from engine import evaluate
from base_data_manager import (get_correct_ann_file_path,get_error_train_model_weight_file_path,
                               get_imgs_dir,exp_data_root_dir)

def get_transform():
    return ToTensor()

def build_ssd_model(num_classes):
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.head.classification_head = SSDClassificationHead(
        [512, 1024, 512, 256, 256, 256],
        model.anchor_generator.num_anchors_per_location(), 
        num_classes
    )
    return model

def build_frcnn_model(num_classes):
    model =torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def model_load_weight(model,model_weights_path):
    # 加载模型
    state_dict = torch.load(model_weights_path,map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def get_coco_results(model, data_loader, device, score_thresh=0.5):
    results = []
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        outputs = model(images)

        for target, out in zip(targets, outputs):
            image_id = target["image_id"]
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # xyxy -> xywh
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
            boxes_xywh[:, 0] = boxes[:, 0]                # x
            boxes_xywh[:, 1] = boxes[:, 1]                # y

            for box, score, label in zip(boxes_xywh, scores, labels):
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(x) for x in box.tolist()],
                    "score": float(score),
                })
    return results


def set_nms(model, model_name, conf_threshold=0.25,iou_threshold=0.5):
    if model_name == "SSD":
        model.score_thresh = conf_threshold
        model.nms_thresh = iou_threshold
    elif model_name == "FRCNN":
        model.roi_heads.nms_thresh = iou_threshold
        model.roi_heads.score_thresh = conf_threshold
    else:
        raise Exception("模型名称错误")
    return model

def offset_category_id(cocoGt):
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    for cat in cats:
        cat["id"] += 1
    anns = cocoGt.loadAnns(cocoGt.getAnnIds())
    for ann in anns:
        ann["category_id"] = ann["category_id"] + 1
    return cocoGt
    


def eval_performance():
    # 加载数据
    dataset = CocoDetectionDataset(
        image_dir=get_imgs_dir(dataset_name,train_or_val,style="coco"),
        annotation_path=ANN_FILE,
        transforms=get_transform()
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # 加载模型
    num_classes = len(dataset.coco.getCatIds()) + 1
    if model_name == "SSD":
        model = build_ssd_model(num_classes)
    elif model_name == "FRCNN":
        model = build_frcnn_model(num_classes)
    else:
        raise Exception("模型名称错误")
    
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    model = model_load_weight(model,model_weights_path)
    # model = set_nms(model, model_name, conf_threshold=0.25,iou_threshold=0.5)
    
    model.eval()
    evaluate(model, train_loader, device=device)  # Using val_loader for evaluation
    '''
    # 开始评估
    coco_results = get_coco_results(model, train_loader, device, score_thresh=0.0)
    # 加载ground truth data
    cocoGt = COCO(ANN_FILE)
    cocoGt = offset_category_id(cocoGt)
    cocoDt = cocoGt.loadRes(coco_results)
    coco_eval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
    '''


if __name__ == "__main__":
    dataset_name = "VOC2012" # VOC2012|KITTI|VisDrone
    model_name = "FRCNN" # FRCNN, SSD
    gpu_id = 0
    train_or_val = "val" # train|val
    ANN_FILE = get_correct_ann_file_path(dataset_name,train_or_val)
    model_weights_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch=49)
    eval_performance()