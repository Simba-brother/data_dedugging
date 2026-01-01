'''
对数据集元信息进行检查
'''
from pycocotools.coco import COCO
import os
from base_data_manager import get_error_ann_file_path,get_correct_ann_file_path
def get_dataset_size_and_obj_size_and_cls_size(anno_json_path):
    coco = COCO(anno_json_path)
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    cat_ids = coco.getCatIds()
    img_num = len(img_ids)
    obj_num = len(ann_ids)
    cat_num = len(cat_ids)
    return img_num,obj_num,cat_num

def get_no_anno_img_num(anno_json_path):
    coco = COCO(anno_json_path)
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    with_ann_img_id_set = set()
    for ann in anns:
        with_ann_img_id_set.add(ann["image_id"])
    no_ann_img_id_set = set(img_ids) - with_ann_img_id_set
    print(f"没有标注的图像数量为:{len(no_ann_img_id_set)}")

def main():
    train_img_num, train_obj_num, train_cat_num = get_dataset_size_and_obj_size_and_cls_size(train_anno_correct_json_path)
    val_img_num, val_obj_num, val_cat_num = get_dataset_size_and_obj_size_and_cls_size(val_anno_correct_json_path)
    error_train_img_num, error_train_obj_num, error_train_cat_num = get_dataset_size_and_obj_size_and_cls_size(train_anno_error_json_path)
    print(f"Train set: #Img:{train_img_num}, #Class:{train_cat_num}, #Obj:{train_obj_num}, #Obj/#Img:{round(train_obj_num/train_img_num,1)}")
    print(f"Val set: #Img:{val_img_num}, #Class:{val_cat_num}, #Obj:{val_obj_num}, #Obj/#Img:{round(val_obj_num/val_img_num,1)}")
    print(f"Error_Train set: #Img:{error_train_img_num}, #Class:{error_train_cat_num}, #Obj:{error_train_obj_num}, #Obj/#Img:{round(train_obj_num/train_img_num,1)}")




if __name__ == "__main__":
    exp_dir = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI|VisDrone

    
    train_anno_correct_json_path = get_correct_ann_file_path(dataset_name,"train")
    train_anno_error_json_path = get_error_ann_file_path(dataset_name)
    val_anno_correct_json_path = get_correct_ann_file_path(dataset_name,"val")
    train_anno_error_json_path = get_error_ann_file_path(dataset_name)
    
    main()
    get_no_anno_img_num(train_anno_error_json_path)