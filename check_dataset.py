'''
对数据集元信息进行检查
'''
from pycocotools.coco import COCO
from base_data_manager import get_error_ann_file_path,get_correct_ann_file_path

def get_dataset_size_and_obj_size_and_cls_size(anno_json_path):
    '''
    基于coco anno file path获得img数量,obj数量和cat数量
    '''
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
    all_image_ids = list(coco.imgs.keys())

    no_count = 0
    no_ann_img_id_set_2 = set()
    for img_id in all_image_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            no_count += 1
            no_ann_img_id_set_2.add(img_id)
            continue
    assert no_count == len(no_ann_img_id_set), "err"
    return len(no_ann_img_id_set)

def main():
    '''
    打印数据集基本信息
    '''
    train_img_num, train_obj_num, train_cat_num = get_dataset_size_and_obj_size_and_cls_size(train_anno_correct_json_path)
    val_img_num, val_obj_num, val_cat_num = get_dataset_size_and_obj_size_and_cls_size(val_anno_correct_json_path)
    test_img_num, test_obj_num, test_cat_num = get_dataset_size_and_obj_size_and_cls_size(test_anno_correct_json_path)
    error_train_img_num, error_train_obj_num, error_train_cat_num = \
        get_dataset_size_and_obj_size_and_cls_size(train_anno_error_json_path)
    

    # 正确标注下的无标img num
    train_correct_ann_no_anno_img_num = get_no_anno_img_num(train_anno_correct_json_path)
    val_ann_no_anno_img_num = get_no_anno_img_num(val_anno_correct_json_path)
    test_ann_no_anno_img_num = get_no_anno_img_num(test_anno_correct_json_path)
    # 错误标注下的无标img num
    train_error_ann_no_anno_img_num = get_no_anno_img_num(train_anno_error_json_path)
    
    print(f"Train set: #Img:{train_img_num}, #Class:{train_cat_num}, #Obj:{train_obj_num}, #Img+#Obj:{train_img_num+train_obj_num}, #Obj/#Img:{round(train_obj_num/train_img_num,1)}")
    print(f"Val set: #Img:{val_img_num}, #Class:{val_cat_num}, #Obj:{val_obj_num}, #Img+#Obj:{val_img_num+val_obj_num}, #Obj/#Img:{round(val_obj_num/val_img_num,1)}")
    print(f"Test set: #Img:{test_img_num}, #Class:{test_cat_num}, #Obj:{test_obj_num}, #Img+#Obj:{test_img_num+test_obj_num}, #Obj/#Img:{round(test_obj_num/test_img_num,1)}")
    print(f"Error Train set: #Img:{error_train_img_num}, #Class:{error_train_cat_num}, #Obj:{error_train_obj_num}, #Img+#Obj:{error_train_img_num+error_train_obj_num}, #Obj/#Img:{round(error_train_obj_num/error_train_img_num,1)}")

    print("train correct noAnno imgs num:",train_correct_ann_no_anno_img_num)
    print("train error noAnno imgs num:",train_error_ann_no_anno_img_num)
    print("val noAnno imgs num:",val_ann_no_anno_img_num)
    print("test noAnno imgs num:",test_ann_no_anno_img_num)


if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    print("exp_root_dir:",exp_root_dir)
    print("dataset_name:",dataset_name)
    # 得到该数据集3个部分（tvt）的正确的coco anno file path
    train_anno_correct_json_path = get_correct_ann_file_path(dataset_name,"train")
    val_anno_correct_json_path = get_correct_ann_file_path(dataset_name,"val")
    test_anno_correct_json_path = get_correct_ann_file_path(dataset_name,"test")
    # 得到该数据集的训练集的错误的coco anno file path
    train_anno_error_json_path = get_error_ann_file_path(dataset_name)
    main()
