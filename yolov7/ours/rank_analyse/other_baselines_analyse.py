import os
import joblib
from ours.base_data_manager import (get_annotations_with_miss_json_path,
                                    get_collected_gt_box_json_path
                                    )
from ours.small_utils import read_json
from ours.rank_analyse.common import compute_apfd,calc_fpr_fnr_f1
from ours.data_organization_tools import get_all_errored_g_box_id_set,get_all_miss_error_img_name_set


def analyse_rank(gt_json_path:str, rank_res:list):
    '''
    rank_res: 我们方法获得的排序结果（idd:img_name or gid）
    '''
    g_boxes_json = read_json(gt_json_path)
    # 得到错误的gid_set
    all_errored_g_box_id_set = get_all_errored_g_box_id_set(g_boxes_json)
    # 得到missed_error_img_name_set
    all_miss_error_img_name_set = get_all_miss_error_img_name_set(annos_with_miss_json_path)

    # 计算APFD,FPR和FNR
    error_set = all_errored_g_box_id_set | all_miss_error_img_name_set
    APFD = compute_apfd(error_set, rank_res)
    FPR,FNR,F1 =calc_fpr_fnr_f1(rank_res, error_set)
    print(f"排序总长度:{len(rank_res)}")
    print(f"APFD:{APFD},FPR:{FPR},FNR:{FNR},F1:{F1}")

def compare():
    rank_entropy = joblib.load(os.path.join(exp_root_dir,"Results",
                             "other_baselines","entropy",dataset_name,model_name,"exp_01","rank","rank.joblib"))
    
    rank_deepgini = joblib.load(os.path.join(exp_root_dir,"Results",
                             "other_baselines","deepgini",dataset_name,model_name,"exp_01","rank","rank.joblib"))

    rank_margin = joblib.load(os.path.join(exp_root_dir,"Results",
                             "other_baselines","margin",dataset_name,model_name,"exp_01","rank","rank.joblib"))
    
    rank_loss = joblib.load(os.path.join(exp_root_dir,"Results",
                             "other_baselines","loss",dataset_name,model_name,"exp_01","rank","rank.joblib"))
    
    for idd_e,idd_d,idd_m,idd_l in zip(rank_entropy,rank_deepgini,rank_margin,rank_loss):
        print(f"idd_e:{idd_e},idd_d:{idd_d},idd_m:{idd_m},idd_l:{idd_l}")

if __name__ == "__main__":
    
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    baseline_name = "loss" # entropy|loss|deepgini|margin
    rank_path = os.path.join(exp_root_dir,"Results",
                             "other_baselines",baseline_name,dataset_name,model_name,"exp_01","rank","rank.joblib")
    rank = joblib.load(rank_path)

    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    annos_with_miss_json_path = get_annotations_with_miss_json_path(dataset_name)
    analyse_rank(gt_json_path,rank)
    # compare()