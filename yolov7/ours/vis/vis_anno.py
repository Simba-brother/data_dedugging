'''
基于train set的anno json file 可视化bbox
'''
import os
import joblib
import json
import time
from datetime import datetime

from ours.base_data_manager import (exp_data_root_dir,
                                    get_ours_rank_res_path,get_collected_gt_box_json_path,
                                    get_error_ann_file_path,get_correct_ann_file_path,
                                    get_annotations_with_miss_json_path,get_datactive_rank_res_path)
from ours.data_organization_tools import (conver_ours_rank,conver_datactive_rank,
                                          get_img_name_to_ann_ids,get_annoId_to_anno,
                                          get_all_error_annoids,get_annoid_to_imgname)
from ours.small_utils import read_json
import pprint
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt

def vis_bbox(img_path:str, annos:list):
    # 读取图像（注意 cv2 是 BGR，需要转 RGB）
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    

    # 显示correct的标注框
    ax1 = axes[0]
    ax1.imshow(img)
    ax1.set_title('Correct Annotation', fontsize=14, color='green')
    ax1.axis('off')
    
    for _anno in annos:
        x, y, w, h = _anno["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_id = _anno['category_id']
        class_name = cls_id_to_name[class_id]
        # 绘制边界框
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
        ax1.add_patch(rect)
        # 添加类别标签
        ax1.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="green", 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax1.axis('off')

    

    plt.savefig(save_path)

def draw_bbox_for_missFault(imgfilepath,correct_annos,missed_annoidset,cls_id_to_name:dict,save_path):
    # 读取图像（注意 cv2 是 BGR，需要转 RGB）
    img = cv2.imread(imgfilepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    

    # 显示correct的标注框
    ax1 = axes[0]
    ax1.imshow(img)
    ax1.set_title('Correct Annotation', fontsize=14, color='green')
    ax1.axis('off')
    
    for _anno in correct_annos:
        x, y, w, h = _anno["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        class_id = _anno['category_id']
        class_name = cls_id_to_name[class_id]
        # 绘制边界框
        rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
        ax1.add_patch(rect)
        # 添加类别标签
        ax1.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="green", 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax1.axis('off')

    # 显示error标注框
    ax2 = axes[1]
    ax2.imshow(img)
    ax2.set_title('Error Annotation', fontsize=14, color='red')
    ax2.axis('off')

    for _anno in correct_annos:
        if _anno['id'] in missed_annoidset:
            x, y, w, h = _anno["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_id = _anno['category_id']
            class_name = cls_id_to_name[class_id]
            # 绘制边界框
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2,linestyle='--',edgecolor="red", facecolor='none')
            ax2.add_patch(rect)
            # 添加类别标签
            ax2.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="red", 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax2.axis('off')
        else:
            x, y, w, h = _anno["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_id = _anno['category_id']
            class_name = cls_id_to_name[class_id]
            # 绘制边界框
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="green", facecolor='none')
            ax2.add_patch(rect)
            # 添加类别标签
            ax2.text(x1, y1 - 5, f'{class_name}', fontsize=10, color="green", 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            ax2.axis('off')

    plt.savefig(save_path)
    pass

def main():
    anno_json = read_json(anno_json_path)
    imgname_to_annoIds = get_img_name_to_ann_ids(anno_json)
    images = anno_json["images"]
    for image in images:
        img_name = image["file_name"]
        img_id = image["id"]
        img_path = os.path.join(exp_data_root_dir,"datasets", f"{dataset_name}-coco", "train",img_name)
        annoIds = imgname_to_annoIds[img_name]
        vis_bbox(img_path, annoIds)





if __name__ == "__main__":
    dataset_name = "KITTI_8"
    model_name = "YOLOv7"
    method_name = "ours" # clean|ours|datactive
    exp_id = "exp_02"

    
    anno_json_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",
                                          "train","_annotations.coco_correct.json")
    
    # repaired_anno_json_path = os.path.join(exp_data_root_dir, "Results", method_name, dataset_name, model_name,
    #                                        exp_id, "repair", "_annotations.coco_repair.json")
    
