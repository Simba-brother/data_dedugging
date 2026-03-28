'''
4种错误的特征
'''
import numpy as np
import os
import joblib
from pycocotools.coco import COCO
from ours.base_data_manager import (exp_data_root_dir,get_collected_gt_box_json_path,
                                    get_error_ann_file_path,
                                    get_annotations_with_miss_json_path,
                                    get_correct_ann_file_path)

from ours.data_organization_tools import (conver_ours_rank, conver_datactive_rank,
                                          get_all_miss_error_img_name_set,
                                          get_all_annoids_detail,
                                          get_all_img_name,get_gid_to_anno_id,
                                          get_annoid_to_imgname,get_annoId_to_anno,
                                          get_cls_id_to_name,get_img_name_to_ann_ids,
                                          get_all_errored_g_box_id_set,get_all_gids)
from ours.small_utils import read_json

from ours.rank.box_rank import box_rank
from ours.rank.img_rank import img_rank_2

import matplotlib.pyplot as plt
import cv2





def main():
    fault_type = "class_fault" # class_fault|loc_fault|redun_fault|miss_fault|clean
    ours_rank = joblib.load(ours_rank_path)
    ours_rank = conver_ours_rank(ours_rank, g_boxes_json, anno_error)
    cut = int(len(ours_rank)*0.5)
    positive_rank = ours_rank[:cut]
    negative_rank = ours_rank[cut:]
    detail = get_all_annoids_detail(anno_error_with_miss)
    fault_annoid_set = set(detail[fault_type])
    idd_list = []
    for idd in positive_rank:
        if idd in fault_annoid_set:
            idd_list.append(idd)
    # annid to gid
    gid_to_annoid = get_gid_to_anno_id(g_boxes_json,anno_error)
    annoid_to_gid = dict(zip(gid_to_annoid.values(), gid_to_annoid.keys()))

    fault_gid_list = []
    for idd in idd_list:
        fault_gid_list.append(annoid_to_gid[idd])
    

    all_gids = get_all_gids(g_boxes_json)
    matched_gid_set = set()
    for gid_str in match_json.keys():
        matched_gid_set.add(int(gid_str))
    
    matched_fault_gid_list = []
    for gid in fault_gid_list:
        if gid in matched_gid_set:
            matched_fault_gid_list.append(gid)
    

    gid = matched_fault_gid_list[500]

    all_fault_gid_set = get_all_errored_g_box_id_set(g_boxes_json)
    
    rank_res = box_rank(gt_json_path,metric_json_path)
    ranked_gids =  rank_res["ranked_gids"]
    X =  rank_res["feature_data"]
    feature_names = rank_res["feature_names"]
    feature_signs = rank_res["sign_list"]
    idx = ranked_gids.index(gid)
    feature =  X[idx]

    fault_idx_list = []
    for fault_gid in all_fault_gid_set:
        fault_idx_list.append(ranked_gids.index(fault_gid))


    rows_to_use = np.delete(X, fault_idx_list, axis=0)  # 删除所有fault gid对应的X，即剩下的都是correct gid的X
    mean_values = np.mean(rows_to_use, axis=0)
    print(f"gid: {gid}")
    print(f"annoid: {idd}")
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        feature_sign = feature_signs[i]
        cur_val = float(feature[i])
        mean_val = float(mean_values[i])
        print(f"{feature_name}|{feature_sign}|cur_val:{cur_val}|mean:{mean_val}")


if __name__ == "__main__":

    dataset_name = "VOC2012" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"

    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    anno_error_path = get_error_ann_file_path(dataset_name)
    anno_with_miss_error_path = get_annotations_with_miss_json_path(dataset_name)
    correct_anno_path = get_correct_ann_file_path(dataset_name,"train")

    
    match_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,
                                   "gp_box_match","match_v2.json")
    metric_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,
                                   "collection_metric","collection_metrics_v2.json")
    img_to_nomatched_pboxs_json_path = os.path.join(exp_data_root_dir,"collection_indicator_bbox_level",
                dataset_name,model_name,"img_to_nomatched_pboxs.json")
    
    imgs_dir = os.path.join(exp_data_root_dir,"retrain_dataset_split",dataset_name,"images","origin")
    anno_error_with_miss = read_json(anno_with_miss_error_path)
    g_boxes_json = read_json(gt_json_path)
    anno_error = read_json(anno_error_path)
    anno_correct = read_json(correct_anno_path)
    match_json = read_json(match_json_path)
    metric_json = read_json(metric_json_path)

    ours_rank_path = os.path.join(exp_data_root_dir,"Results","ours",
                                  dataset_name,model_name,
                                  "exp_01","rank","rank.joblib")

    main()