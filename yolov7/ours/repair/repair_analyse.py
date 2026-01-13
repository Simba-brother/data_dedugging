'''
分析ours,datactive修复box数量
'''

import os
import joblib
from ours.base_data_manager import exp_data_root_dir,get_collected_gt_box_json_path
from ours.small_utils import read_json
from pycocotools.coco import COCO
from ours.data_organization_tools import (get_gid_to_anno_id,get_error_annoid_set,
                                          get_all_miss_img_name_list,
                                          get_imgid_to_imgname,get_all_error_clean_set, 
                                          conver_ours_rank, 
                                          conver_datactive_rank,
                                          get_img_name_to_missed_annids,
                                          get_annoId_to_anno,
                                          get_all_error_annoids)


def count_repair_rate(rank:list,imgname_to_missed_annids:dict,all_error_annoids:list[int],annoId_to_anno:dict,cut_off_rate:float=0.4):
    
    cut_off =  int(len(rank)*cut_off_rate)
    cut_off_rank = rank[:cut_off]
    # 修复的miss count
    repaired_miss_box_count = 0
    # 修复的other count
    repaired_other_box_count = 0
    for idd in cut_off_rank:
        if type(idd) is str:
            img_name = idd
            repaired_miss_box_count += len(imgname_to_missed_annids[img_name])
        else:
            anno_id = idd
            anno = annoId_to_anno[anno_id]
            if anno["fault_type"] != 0:
                repaired_other_box_count += 1
    repaired_box_count = repaired_miss_box_count + repaired_other_box_count
    repair_rate = round(repaired_box_count/len(all_error_annoids),4)
    return repaired_box_count,repair_rate
    



def main():
    # 得到我们方法的排序
    ours_rank = joblib.load(ours_rank_path)
    g_boxes_json = read_json(gt_json_path)
    anno_error = read_json(anno_error_path)
    converted_ours_rank = conver_ours_rank(ours_rank, g_boxes_json, anno_error)

    # 得到datactive方法的排序
    datactive_rank = joblib.load(datactive_rank_path)
    coco = COCO(anno_error_path)
    bg_catId = coco.getCatIds()[-1]+1
    converted_datactive_rank = conver_datactive_rank(datactive_rank, bg_catId)

    anno_with_miss_error = read_json(anno_with_miss_error_path)
    all_error_annoids = get_all_error_annoids(anno_with_miss_error)
    annoId_to_anno = get_annoId_to_anno(anno_with_miss_error)
    imgname_to_missed_annids = get_img_name_to_missed_annids(anno_with_miss_error) 
    
    print(f"总共有错误的box数量（包括miss error）:{len(all_error_annoids)}")
    cut_off_rate = 0.4
    repaired_box_count,repair_rate = count_repair_rate(converted_datactive_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate)
    print(f"datactive修复数量: {repaired_box_count}, 修复率: {repair_rate}")

    repaired_box_count,repair_rate = count_repair_rate(converted_ours_rank,imgname_to_missed_annids,all_error_annoids,annoId_to_anno,cut_off_rate)
    print(f"ours修复数量: {repaired_box_count}, 修复率: {repair_rate}")





if __name__ == "__main__":
    dataset_name = "VisDrone"
    model_name = "YOLOv7"
    ours_rank_path = os.path.join(exp_data_root_dir,"final_res","ours",dataset_name,model_name,"rank_res",
                            "alpha=1.5","rank_topsis.joblib")
    datactive_rank_path = os.path.join(exp_data_root_dir,"final_res","datactive",dataset_name,"ranked_result",
                            "ranked_list.joblib")
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    anno_error_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",
                            "_annotations.coco_error.json")
    anno_with_miss_error_path = os.path.join(exp_data_root_dir,"error_anno",dataset_name,"coco_format",
                            "annotations_with_miss.json")
    main()

