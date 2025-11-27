import os
import argparse
import torch
from utils.torch_utils import select_device
from utils.datasets import create_dataloader
from models.yolo import Model
import yaml
import json
from utils.general import colorstr,non_max_suppression




def main():
    # 拿到模型
    weights = "runs/train/voc2012_error/weights/epoch_20.pt"
    device = select_device('0')
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=20, anchors=3).to(device)  # create
    state_dict = torch.load(weights, map_location=device)  # load checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # 拿到数据
    data = "data/VOC2012.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    dataloader = create_dataloader(data["train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'train: '))[0]
    predicted_bboxs = []
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with torch.no_grad():
            out, train_out = model(img, augment=False)
            lb = []  # for autolabelling
            out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.65, labels=lb, multi_label=True)  # inference and training outputs
            for i in range(len(out)):
                img_name = paths[i].split("/")[-1]
                predicted_bbox_nums = out[i].shape[0]
                for j in range(predicted_bbox_nums):
                    item = {}
                    item["img_name"] = img_name
                    item["bbox"] = out[i][j][:4].tolist()
                    item["conf"] = out[i][j][4].item()
                    item["predicted_cls"] = int(out[i][j][5].item())
                    predicted_bboxs.append(item)
    save_dir = "/data/mml/data_debugging_data/collection_indicator_bbox_level/VOC2012/YOLOv7"
    save_json_file_name = "epoch_20_predicted_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(predicted_bboxs, f, indent=4)

if __name__ == "__main__":
    main()