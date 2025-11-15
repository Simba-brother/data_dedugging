import os
import glob
import random
import shutil

# 固定随机种子，保证每次划分一致（如不需要可去掉）
random.seed(42)

# 原始目录（当前所有数据都在这里）
img_dir = '/data/mml/data_debugging_data/datasets/KITTI-yolo/train/images'
label_dir = '/data/mml/data_debugging_data/datasets/KITTI-yolo/train/labels'

# 新建验证集目录
val_img_dir = '/data/mml/data_debugging_data/datasets/KITTI-yolo/val/images'
val_label_dir = '/data/mml/data_debugging_data/datasets/KITTI-yolo/val/labels'
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 读取所有 png 图像
img_paths = glob.glob(os.path.join(img_dir, '*.png'))
num_total = len(img_paths)
print('总图像数:', num_total)

# 想要的数量
num_val = 3741  # 验证集数量
num_train = num_total - num_val

if num_val >= num_total:
    raise ValueError("验证集数量不能大于等于总数量")

# 随机抽取验证集图像
val_img_paths = random.sample(img_paths, num_val)

# 移动到 val 目录
for img_path in val_img_paths:
    filename = os.path.basename(img_path)         # xxx.png
    stem, _ = os.path.splitext(filename)          # xxx
    label_path = os.path.join(label_dir, stem + '.txt')

    # 移动图像
    shutil.move(img_path, os.path.join(val_img_dir, filename))

    # 移动标签
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(val_label_dir, stem + '.txt'))
    else:
        print('警告：找不到标签文件', label_path)

print('划分完成：')
print('训练集 imgs 剩余:', len(glob.glob(os.path.join(img_dir, "*.png"))))   # 期望 3740
print('验证集 imgs 数量:', len(glob.glob(os.path.join(val_img_dir, "*.png"))))  # 期望 3741)
