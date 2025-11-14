'''
error img 可视化
'''
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pycocotools.coco import COCO

def load_yolo_labels(txt_path):
    """读取YOLO标注文件"""
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append((int(cls), x, y, w, h))
    return boxes

def plot_boxes(ax, image, boxes, box_type, title):
    """在图像上绘制YOLO标注框"""
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w, _ = image.shape
    if box_type == "yolo":
        for cls, x, y, bw, bh in boxes:
            # 将相对坐标转换为像素坐标
            x_min = (x - bw / 2) * w
            y_min = (y - bh / 2) * h
            rect_w = bw * w
            rect_h = bh * h

            # 绘制矩形框
            rect = patches.Rectangle((x_min, y_min), rect_w, rect_h,
                                    linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, f'Class {cls}', color='yellow',
                    fontsize=10, backgroundcolor='black')
    if box_type == "coco":
        for cls, x_min, y_min, rect_w, rect_h in boxes:
            # 绘制矩形框
            rect = patches.Rectangle((x_min, y_min), rect_w, rect_h,
                                    linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, f'Class {cls}', color='yellow',
                    fontsize=10, backgroundcolor='black')
            
    ax.set_title(title)
    ax.axis('off')

def vis_ccoco_anno():
    img_name = "2008_000095.jpg"
    coco  = COCO(correct_annotation_json_path)
    anns = get_anns_by_imgname(coco,img_name)
    for ann in anns:
        bbox = ann["bbox"]
        cls_id = ann["category_id"]




def compare_annotations_yolo_format(image_path, correct_txt, tampered_txt, error_type, img_file_name):
    """比较正确与篡改标注"""
    # 读取图像和标注
    image = cv2.imread(image_path)
    correct_boxes = load_yolo_labels(correct_txt)
    tampered_boxes = load_yolo_labels(tampered_txt)

    # 绘制对比图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_boxes(axs[0], image, correct_boxes, "yolo", 'correct annotation')
    plot_boxes(axs[1], image, tampered_boxes, "yolo", f'error annotation:{error_type}')

    plt.tight_layout()
    
    save_dir = os.path.join(exp_data_root,"datasets",f"{dataset_name}_vis_error_annotaion","yolo_format")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = img_file_name
    save_path = os.path.join(save_dir, save_file_name)
    plt.savefig(save_path)
    print(save_path)

def compare_annotations_coco_format(image_path, correct_bboxes, error_bboxes, error_type, img_file_name):
    """比较正确与篡改标注"""
    # 读取图像和标注
    image = cv2.imread(image_path)
    # 绘制对比图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_boxes(axs[0], image, correct_bboxes, "coco", 'correct annotation')
    plot_boxes(axs[1], image, error_bboxes, "coco", f'error annotation:{error_type}')

    plt.tight_layout()
    save_dir = os.path.join(exp_data_root,"datasets",f"{dataset_name}_vis_error_annotaion","coco_format")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = img_file_name
    save_path = os.path.join(save_dir, save_file_name)
    plt.savefig(save_path)
    print(save_path)

def vis_yolo_format():
    for row_id, row in error_record_df.iterrows():
        img_file_name = row["img_file_name"]
        label_file_name = img_file_name.replace(".jpg", ".txt")
        error_type = row["error_type"]
        img_file_path = os.path.join(f"{exp_data_root}/datasets/{dataset_name}-yolo/train/images", img_file_name)
        correct_label_file_path = os.path.join(f"{exp_data_root}/datasets/{dataset_name}-yolo/train/labels_correct", label_file_name)
        error_label_file_path = os.path.join(f"{exp_data_root}/datasets/{dataset_name}-yolo/train/labels_error", label_file_name)
        compare_annotations_yolo_format(img_file_path,correct_label_file_path,error_label_file_path, error_type,img_file_name)

def get_anns_by_imgname(coco,img_name):
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    img_id = None
    for img in imgs:
        if img["file_name"] == img_name:
            img_id = img["id"]
            break
    if img_id is None:
        raise ValueError(f"找不到文件名为 {img_name} 的图片")
    # 2. 通过 image_id 找到对应的 annotation
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    return anns

def get_bboxes(anns):
    bboxes = []
    for anno in anns:
        xmin,ymin,w,h = anno["bbox"]
        cls_id = anno["category_id"]
        bboxes.append([cls_id,xmin,ymin,w,h])
    return bboxes


def vis_coco_format():
    error_coco = COCO(error_annotation_json_path)
    correct_coco = COCO(correct_annotation_json_path)
    for row_id, row in error_record_df.iterrows():
        img_name = row["img_file_name"]
        error_type = row["error_type"]
        error_anns = get_anns_by_imgname(error_coco,img_name)
        correct_anns = get_anns_by_imgname(correct_coco,img_name)
        error_bboxes = get_bboxes(error_anns)
        correct_bboxes = get_bboxes(correct_anns)
        img_path = os.path.join(exp_data_root,"datasets", f"{dataset_name}-coco", "train", img_name)
        compare_annotations_coco_format(img_path, correct_bboxes, error_bboxes, error_type, img_name)

if __name__ == "__main__":

    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    error_record_df = pd.read_csv(f"{exp_data_root}/datasets/{dataset_name}_error_record/error_record_simple.csv")

    # vis_yolo_format()

    
    error_annotation_json_path = os.path.join(f"{exp_data_root}/datasets/{dataset_name}-coco/train/_annotations.coco_error.json")
    correct_annotation_json_path = os.path.join(f"{exp_data_root}/datasets/{dataset_name}-coco/train/_annotations.coco_correct.json")
    vis_coco_format()
    
