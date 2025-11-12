from pycocotools.coco import COCO
import torch
from pycocotools.cocoeval import COCOeval
import json
import os


def eval_f(gt_json,pred_json):
    coco_anno = COCO(gt_json)
    coco_dets = coco_anno.loadRes(pred_json)
    coco_eval = COCOeval(coco_anno, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

def eval(model,data_loader,device,save_dir):
    model.eval()
    eval_result = []
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        for i in range(len(targets)):
            image_id = targets[i]["image_id"].item()
            bbox_num = outputs[i]["boxes"].shape[0]
            outputs[i]["boxes"][:,2:] -= outputs[i]["boxes"][:,:2]
            for j in range(bbox_num):
                bbox = outputs[i]["boxes"][j].tolist()
                score = outputs[i]["scores"][j].item()
                category_id = outputs[i]["labels"][j].item()
                item = {
                    "image_id":image_id,
                    "category_id":category_id,
                    "bbox":bbox,
                    "score":score
                }
                eval_result.append(item)
    eval_result_json_file_path = f"{save_dir}/eval_result.json"
    with open(eval_result_json_file_path, 'w') as f:
        json.dump(eval_result, f)
    gt_json = "/data/mml/data_debugging/datasets/football-player/valid/_annotations.coco.json"
    pred_json = eval_result_json_file_path
    coco_eval = eval_f(gt_json,pred_json)
    os.remove(eval_result_json_file_path)



