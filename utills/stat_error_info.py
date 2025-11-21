'''
对错误生成的结果进行统计分析
'''
import os
import pandas as pd
from pycocotools.coco import COCO

def main():
    correct_anno_json_path = os.path.join(exp_data_dir,"datasets",f"{dataset_name}-coco","train","_annotations.coco_correct.json")
    coco = COCO(correct_anno_json_path)
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    obj_ids = coco.getAnnIds()
    num_objs = len(obj_ids)
    fault_df = pd.read_csv(os.path.join(exp_data_dir,"error_anno",dataset_name,"fault_records.csv")) 
    error_img_size = len(set(fault_df["img_name"].tolist()))
    error_obj_size = len(fault_df["obj_id"].tolist())
    fault_type_count = fault_df["fault_type"].value_counts()
    error_img_percentage = round(error_img_size/num_images,3)
    error_obj_percentage = round(error_obj_size/num_objs,3)

    print(f"数据集:{dataset_name}")
    print(f"总img数量:{num_images};error img 数量:{error_img_size};error img 占比:{error_img_percentage:.2%}")
    print(f"总obj数量:{num_objs};error obj 数量:{error_obj_size};error obj 占比:{error_obj_percentage:.2%}")
    print(f"cls error数量:{fault_type_count[1]};loc error 数量:{fault_type_count[2]};redundancy error 数量:{fault_type_count[3]};mis error 数量:{fault_type_count[4]}")

if __name__ == "__main__":
    exp_data_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI" # VOC2012|VisDrone|KITTI
    fault_name_to_fault_id = {
        'no_fault': 0,
        'cls_fault': 1,
        'loc_fault': 2,
        'redundancy_fault': 3,
        'missing_fault': 4,
    }

    fault_id_to_fault_name = {
        0: "no_fault",
        1: 'cls_fault',
        2: 'loc_fault',
        3: 'redundancy_fault',
        4: 'missing_fault'
    }
    main()
