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

def load_yolo_labels(txt_path):
    """读取YOLO标注文件"""
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append((int(cls), x, y, w, h))
    return boxes

def plot_boxes(ax, image, boxes, title):
    """在图像上绘制YOLO标注框"""
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w, _ = image.shape
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
    ax.set_title(title)
    ax.axis('off')

def compare_annotations(image_path, correct_txt, tampered_txt, error_type, img_file_name):
    """比较正确与篡改标注"""
    # 读取图像和标注
    image = cv2.imread(image_path)
    correct_boxes = load_yolo_labels(correct_txt)
    tampered_boxes = load_yolo_labels(tampered_txt)

    # 绘制对比图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_boxes(axs[0], image, correct_boxes, 'correct annotation')
    plot_boxes(axs[1], image, tampered_boxes, f'error annotation:{error_type}')

    plt.tight_layout()
    save_dir = "/data/mml/data_debugging/datasets/VOC2012_vis_error_annotation"
    save_file_name = img_file_name
    save_path = os.path.join(save_dir, save_file_name)
    plt.savefig(save_path)
    print(save_path)

def vis():
    error_record_df = pd.read_csv("/data/mml/data_debugging/datasets/VOC2012_error_record/error_record_simple.csv")
    for row_id, row in error_record_df.iterrows():
        img_file_name = row["img_file_name"]
        label_file_name = img_file_name.replace(".jpg", ".txt")
        error_type = row["error_type"]
        img_file_path = os.path.join("/data/mml/data_debugging/datasets/VOC2012-yolo/train/images", img_file_name)
        correct_label_file_path = os.path.join("/data/mml/data_debugging/datasets/VOC2012-yolo/train/labels_correct", label_file_name)
        error_label_file_path = os.path.join("/data/mml/data_debugging/datasets/VOC2012-yolo/train/labels_error", label_file_name)
        compare_annotations(img_file_path,correct_label_file_path,error_label_file_path, error_type,img_file_name)
if __name__ == "__main__":
    vis()
