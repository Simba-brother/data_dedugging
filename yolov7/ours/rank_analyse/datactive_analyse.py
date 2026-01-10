'''
分析DataDetective的rank
'''
import os
import json
import joblib
from pycocotools.coco import COCO
from ours.base_data_manager import (exp_data_root_dir, get_datactive_rank_res_path,
                                    get_error_ann_file_path,get_annotations_with_miss_json_path)
from common import *

def get_image_id_to_image_name_for_coco(annos_with_miss_json:dict) -> dict:
    id2name = {}
    images = annos_with_miss_json["images"]
    for image in images:
        id2name[image["id"]] = image["file_name"] 
    return id2name


def convert_rank_list(ranked_list:list,bg_id:int):
    converted_rank_list = []
    for instance in ranked_list:
        gt_category_id = instance["gt_category_id"]
        if gt_category_id == bg_id:
            converted_rank_list.append(instance["image_name"])
        else:
            converted_rank_list.append(instance["anno_id"])
    return converted_rank_list

def get_missed_img_name_set(annotations_with_miss_json):
    miss_img_name_set = set()
    imgId_to_imgName = get_image_id_to_image_name_for_coco(annotations_with_miss_json)
    annos = annotations_with_miss_json["annotations"]
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = imgId_to_imgName[anno["image_id"]]
            miss_img_name_set.add(img_name)
    return miss_img_name_set

def get_error_ann_id_set(coco:COCO):
    anns = coco.loadAnns(coco.getAnnIds())
    error_ann_id_set = set()
    for ann in anns:
        if ann["fault_type"] in [1,2,3]: # cls,loc,red
            error_ann_id_set.add(ann["id"])
    return error_ann_id_set


def look_annid_rank(ranked_gid_list:list[int], all_errored_g_box_id_set:set[int]):
    pic_save_path = os.path.join(exp_data_root_dir,"temp", "annid_rank.png")
    error_flag_list = []
    for gid in ranked_gid_list:
        if gid in all_errored_g_box_id_set:
            error_flag_list.append(1)
        else:
            error_flag_list.append(0)
    draw_rank_hot(error_flag_list,pic_save_path)
    print(f"图片保存在：{pic_save_path}")

def main():
    coco = COCO(anno_coco_error_json_path)
    catIds = coco.getCatIds()
    bg_id = catIds[-1]+1
    converted_rank_list = convert_rank_list(ranked_list,bg_id)
    print(f"rank list 长度:{len(converted_rank_list)}")
    ranked_annid_list = []
    ranked_img_name_list = []
    for idd in converted_rank_list:
        if type(idd) is str:
            ranked_img_name_list.append(idd)
        else:
            ranked_annid_list.append(idd)
    error_ann_id_set = get_error_ann_id_set(coco)
    
    with open(annotations_with_miss_json_path,'r') as f:
        annotations_with_miss_json = json.load(f)
    missed_img_name_set =  get_missed_img_name_set(annotations_with_miss_json)
    look_annid_rank(ranked_annid_list, error_ann_id_set)
    look_img_rank(ranked_img_name_list, missed_img_name_set)
    look_total_rank(converted_rank_list,error_ann_id_set,missed_img_name_set)
    total_error_set = error_ann_id_set | missed_img_name_set

    # 计算APFD,FPR和FNR
    APFD = compute_apfd(total_error_set, converted_rank_list)
    FPR,FNR =calc_fpr_fnr(converted_rank_list, total_error_set)
    print(f"AFFD:{APFD},FPR:{FPR},FNR:{FNR}")

if __name__ == "__main__":
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    # datactive 排序的idd
    ranked_list = joblib.load(get_datactive_rank_res_path(dataset_name))
    anno_coco_error_json_path = get_error_ann_file_path(dataset_name)
    annotations_with_miss_json_path =get_annotations_with_miss_json_path(dataset_name)
    main()




