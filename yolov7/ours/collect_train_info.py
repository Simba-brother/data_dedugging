'''
收集gt_box和p_box的信息
'''
import os
import argparse
import torch
from utils.torch_utils import select_device
from utils.datasets import create_dataloader
from models.yolo import Model
import yaml
import json
from utils.general import colorstr,non_max_suppression,scale_coords,xyxy2xywh
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd

def collect_one_epoch(model,dataloader,epoch):
    predicted_box_dict = {}
    predicted_box_id = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # target.shape = (obj_num,6), 
        # obj_num:这一整个 batch 里所有图片的 GT 框数量之和；6:b_i,cls,
        # targets[i, 2:6] = [x, y, w, h] 为归一化后的 xywh 坐标：
            # x, y：目标框中心点相对图像宽高的比例（0~1）
            # w, h：目标框宽高相对图像宽高的比例（0~1）
            # 这个“图像”是指经过 letterbox 之后送进网络的那张（img），不是原始图
        targets = targets.to(device)
        # img经过了数据增强
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with torch.no_grad():
            out, train_out = model(img, augment=False)
            lb = []  # for autolabelling
            # out:list, len(out):batch_size, out[i]:shape:(obj_num,6):第i个img的box_num和xyxy,conf,cls
            out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.65, labels=lb, multi_label=True)  # inference and training outputs
            # Statistics per image
            for si, pred in enumerate(out):
                if len(pred) == 0:
                    # 如果当前图像没有预测信息，则直接跳过该图像
                    continue
                img_name = paths[si].split("/")[-1]
                predn = pred.clone()
                # shapes[si][0]:si这个图像的原始h,w
                # shapes[si][1]:si这个图像resize比例和padding信息
                # img[si].shape[1:]:si这个增强后的图像的h,w
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
                # predn[:, :4]（即 xyxy）已经是原图坐标系中的像素坐标。
                # 存每张图像的预测bbox
                predicted_bbox_list = []
                for *xyxy, conf, cls in predn.tolist():
                    predicted_box = {
                        "predicted_box_id":predicted_box_id,
                        "img_name":img_name,
                        "predicted_cls":int(cls),
                        "conf":conf,
                        "bbox":xyxy
                    }
                    predicted_box_id += 1
                    predicted_bbox_list.append(predicted_box)
                predicted_box_dict[img_name] = {
                        "predicted_bboxs":predicted_bbox_list,
                        "height":shapes[si][0][0],
                        "weight":shapes[si][0][1]
                }
                '''
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    item = {}
                    item["img_name"] = path.stem
                    item["bbox"] = list(xywh)
                    item["conf"] = conf
                    item["predicted_cls"] = int(cls)
                '''

    save_dir = os.path.join(exp_data_root,"collection_indicator_bbox_level",dataset_name,"YOLOv7","collected_predicted_box")
    os.makedirs(save_dir,exist_ok=True)
    save_json_file_name = f"epoch_{epoch}_predicted_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(predicted_box_dict, f, indent=4)

def collect_predicted_box():
    # 拿到数据yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    # 数据加载器
    dataloader = create_dataloader(data["train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]
    imgs_num = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        imgs_num += img.shape[0]
    print(f"总共图像数量:{imgs_num}")

    for epoch in range(epochs):
        # 轮次权重
        weights_path = os.path.join(exp_data_root,"models",f"{dataset_name.lower()}_error","yolov7",f"epoch_{epoch}.pt")
        state_dict = torch.load(weights_path, map_location=device)  # load checkpoint
        # 注入权重
        model.load_state_dict(state_dict, strict=True)
        # 模型评估
        model.eval()
        collect_one_epoch(model,dataloader,epoch)


def get_all_files(directory):
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    return files


def search_annotations_by_img_id(img_id,annotations_no_miss):
    annos_of_img = []
    annotations = annotations_no_miss["annotations"]
    # 按照顺序变量annos
    for anno in annotations:
        if anno["image_id"] == img_id:
            annos_of_img.append(anno)
    # 这张图像顺序的anns
    return annos_of_img

def collect_gt_box():
    annotations_no_miss_path = os.path.join(exp_data_root,"error_anno",dataset_name,"annotations_no_miss.json")
    with open(annotations_no_miss_path, 'r') as f:
        annotations_no_miss = json.load(f)
    images_list = annotations_no_miss["images"]
    gt_box_dict  = defaultdict(list)
    box_id = 0
    no_anno_count = 0
    for image in images_list:
        img_id = image["id"]
        # 这张图像顺序的annos,与line是对齐的
        annos_of_img = search_annotations_by_img_id(img_id,annotations_no_miss)
        img_name = image["file_name"]
        imge_name_no_ext = img_name.split(".")[0]
        # 这张图像的yolo anno txt
        txt_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-yolo","train","labels",f"{imge_name_no_ext}.txt")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            no_anno_count += 1 # 统计了不含有anno的img的数量
        assert len(lines) == len(annos_of_img), "标注对应错误"
        for l_id, line in enumerate(lines):
            box_line = line.split()
            cls = int(box_line[0])
            x_center = float(box_line[1])
            y_center = float(box_line[2])
            width = float(box_line[3])
            height = float(box_line[4])
            fault_type = annos_of_img[l_id]["fault_type"]
            box = {
                "box_id":box_id,
                "img_name":img_name,
                "cls":cls,
                "gt_bbox":[x_center,y_center,width,height],
                "fault_type":fault_type
            }
            box_id += 1
            gt_box_dict[img_name].append(box)
    save_dir = os.path.join(exp_data_root,"collection_indicator_bbox_level",dataset_name,"YOLOv7")
    save_json_file_name = "gt_bboxs.json"
    save_json_path = os.path.join(save_dir,save_json_file_name)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_box_dict, f, indent=4)
    print(f"collect_gt_box完成，并保存在:{save_json_path}")

def merge_gt_predicted_box():
    # 加载gt box
    gt_bboxs_json_path = os.path.join(exp_data_root,"collection_indicator_bbox_level",dataset_name,model_name,"gt_bboxs.json")
    with open(gt_bboxs_json_path,"r") as file:
        gt_bboxs_dict = json.load(file)
    all_dict = {}
    for img_name in gt_bboxs_dict.keys():
        # 读取图像
        image_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-yolo","train","images",img_name)
        image = Image.open(image_path)
        width, height = image.size
        # 得到该张图像下所有gt boxs
        gt_bboxs = gt_bboxs_dict[img_name]
        # 遍历该张图像所有轮次
        predicted_bboxs_over_epoch = []
        for epoch in range(epochs):
            epoch_json_path = os.path.join(exp_data_root,"collection_indicator_bbox_level",dataset_name,model_name,f"epoch_{epoch}_predicted_bboxs.json")
            with open(epoch_json_path,"r") as file:
                epoch_dict = json.load(file)
            value = epoch_dict.get(img_name)
            if value is not None:
                epoch_predicted_bboxs = epoch_dict[img_name]["predicted_bboxs"]
                predicted_bboxs_over_epoch.append(epoch_predicted_bboxs)
            else:
                predicted_bboxs_over_epoch.append(None)

        all_dict[img_name] = {
            "gt_bboxs":gt_bboxs,
            "predicted_bboxs_over_epoch":predicted_bboxs_over_epoch,
            "size":(width,height)
        }
        print(f"{img_name}合并完毕")
    save_dir = os.path.join(exp_data_root,"collection_indicator_bbox_level",dataset_name,model_name)
    save_file_name = "all_gt_predicted.json"
    save_file_path = os.path.join(save_dir,save_file_name)
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(all_dict, f, indent=4)

def record_no_anno_imgs():

    annotations_no_miss_path = os.path.join(exp_data_root,"error_anno",dataset_name,"annotations_no_miss.json")
    with open(annotations_no_miss_path, 'r') as f:
        annotations_no_miss = json.load(f)
    images_list = annotations_no_miss["images"]
    
    no_anno_img_name_list = []
    for image in images_list:
        img_id = image["id"]
        annos_of_img = search_annotations_by_img_id(img_id,annotations_no_miss)
        img_name = image["file_name"]
        imge_name_no_ext = img_name.split(".")[0]
        txt_path = os.path.join(exp_data_root,"datasets",f"{dataset_name}-yolo","train","labels",f"{imge_name_no_ext}.txt")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                no_anno_img_name_list.append(img_name)
    


    print(len(no_anno_img_name_list))
    
                

    # 拿到数据yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    dataloader = create_dataloader(data["train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                prefix=colorstr(f'train: '))[0]
    loaded_img_size = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        loaded_img_size += len(paths)
    print(loaded_img_size)
    
def get_nc(dataset_name):
    if dataset_name == "VOC2012":
        nc = 20
    elif dataset_name == "KITTI":
        nc = 9
    elif dataset_name == "VisDrone":
        nc = 10
    else:
        raise Exception("数据集参数错误")
    return nc



if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012, KITTI, VisDrone
    nc = get_nc(dataset_name)
    model_name = "YOLOv7"
    # 脚本设备
    device = select_device('0')
    # create model 结构
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    epochs = 50
    # collect_predicted_box()
    # collect_gt_box()
    # merge_gt_predicted_box()
    # record_no_anno_imgs()