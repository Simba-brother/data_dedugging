
'''
YOLOv7在3个数据集上的其他基线方法复现代码
'''

import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
from utils.datasets import create_dataloader
from utils.general import (colorstr,non_max_suppression,
                           non_max_suppression_with_probs,
                           scale_coords,xyxy2xywh)
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from models.yolo import Model
from base_data_manager import (exp_data_root_dir,
                            get_error_train_model_weight_file_path, 
                            get_nc_by_datasetname, 
                            get_error_ann_file_path)


def model_load_weight(model:nn.Module,device,weight_path:str):
    # 加载模型权重
    
    if weight_path.endswith("last.pt"):
        state_dict = torch.load(weight_path, map_location=device, weights_only=False)
        state_dict = state_dict['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    return model

def get_name_id_map(ANN_FILE):
    
    coco = COCO(ANN_FILE)

    # 1) file_name -> img_id（假设 file_name 唯一）
    name2id = {img_info["file_name"]: img_id for img_id, img_info in coco.imgs.items()}

    # 2) 反向：img_id -> file_name
    id2name = {img_id: img_info["file_name"] for img_id, img_info in coco.imgs.items()}

    return name2id,id2name

def get_batch_res(imgs,outs,paths,shapes,name2id):
    batch_res = []
    for si, pred in enumerate(outs):
        # si: 这批图像的局部索引
        # 拷贝一份这张图像的预测
        predn = pred.clone()
        path = Path(paths[si])
        img_name = path.name
        img_id = name2id[img_name]
        scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
        for *xyxy, conf, cls in predn.tolist():
            xywh = xyxy2xywh(xyxy)
            batch_res.append({
                "image_id": int(img_id),
                "category_id": int(cls),
                "bbox":xywh,
                "score": float(conf),
                "probs": 
            })
    return batch_res

def main():
    # 加载设备
    device = torch.device(f"cuda:{gpu_id}")
    # 加载最后的模型
    model_weight_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch=49)
    # 加载模型结构
    nc = get_nc_by_datasetname(dataset_name)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    
    model = model_load_weight(model,device,model_weight_path)
    model.eval()

    # 加载error数据集
    # 读取data yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    # 数据加载器
    dataloader = create_dataloader(data["origin_train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]
    imgs_num = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        imgs_num += img.shape[0]
    print(f"总共图像数量:{imgs_num}")

    imgname2id,id2imgname = get_name_id_map(error_anno_file_path)

    # 将数据集喂给model
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        '''
        shapes: 该批次中每张图像的origin_size
        '''
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.float()
        imgs /= 255.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, wid
        with torch.no_grad():
            # out:shape:(bs,anchors*grids,nc+5)
            outs, train_outs = model(imgs, augment=False)  # inference and training outputs
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            lb = []
            # len(outs) eq batch_size, type(outs) eq list
            # outs[0][0].shape eq torch.Size([82, 6]): dets xyxy,conf,cls
            # outs[0][1].shape eq torch.Size([82, 20]): probs
            outs = non_max_suppression_with_probs(outs, conf_thres=0.001, iou_thres=0.65, labels=lb, multi_label=True)
            for dets, probs in outs:
                print(dets.shape, probs.shape)
            batch_res = get_batch_res(imgs,outs,paths,shapes,imgname2id)

if __name__ == "__main__":
    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "YOLOv7"
    gpu_id = 0
    error_anno_file_path = get_error_ann_file_path(dataset_name)
    main()