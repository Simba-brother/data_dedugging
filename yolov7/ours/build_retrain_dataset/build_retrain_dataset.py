'''
Docstring for ours.build_retrain_dataset
'''
import os

from pathlib import Path
import random
import shutil
from ours.small_utils import get_all_files
from tqdm import tqdm


def extract_ops(file_name_list,source_dir,target_dir):
    for file_name in tqdm(file_name_list):
        source_file_path = os.path.join(source_dir,file_name)
        target_file_path = os.path.join(target_dir,file_name)
        shutil.copyfile(source_file_path, target_file_path)

def extract_imgs():
    random.seed(random_seed)
    img_path_list = get_all_files(source_imgs_dir)
    img_name_list = []
    for img_path in img_path_list:
        img_name = Path(img_path).name
        img_name_list.append(img_name)
    val_num = int(val_rate*len(img_name_list))

    val_img_name_list = random.sample(img_name_list,val_num)
    train_img_name_list = list(set(img_name_list) - set(val_img_name_list))
    print("抽取train imgs...")
    extract_ops(train_img_name_list,source_imgs_dir,target_train_imgs_dir)
    print("抽取val imgs...")
    extract_ops(val_img_name_list,source_imgs_dir,target_val_images_dir)




def extract_labels():
    def extract_labels_help(imgs_dir,source_labels_dir,target_labels_dir):
        img_path_list = get_all_files(imgs_dir)
        img_name_list = [Path(img_path).name for img_path in img_path_list]
        label_name_list = []
        for img_name in img_name_list:
            base_name,ext = os.path.splitext(img_name)
            label_name = base_name+".txt"
            label_name_list.append(label_name)
        extract_ops(label_name_list,source_labels_dir,target_labels_dir)
    print("抽取train labels...")
    extract_labels_help(target_train_imgs_dir,source_labels_dir,target_train_labels_dir)
    print("抽取val labels...")
    extract_labels_help(target_val_imgs_dir,source_labels_dir,target_val_labels_dir)


if __name__ == "__main__":
    '''
    # 选择从总的训练集中选择的数据用于验证集 20%, 剩下80%数据用于训练集
    random_seed = "42"
    val_rate = 0.2
    source_imgs_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/images/origin"
    target_train_imgs_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/images/split/train"
    target_val_images_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/images/split/val"
    extract_imgs()
    '''


    target_train_imgs_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/images/split/train"
    target_val_imgs_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/images/split/val"
    source_labels_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/labels/repair_datactive/origin"
    target_train_labels_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/labels/repair_datactive/split/train"
    target_val_labels_dir = "/data/mml/data_debugging_data/retrain_dataset_split/VisDrone/labels/repair_datactive/split/val"
    extract_labels()


