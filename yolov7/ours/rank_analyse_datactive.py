'''
分析DataDetective的rank
'''
import os
import json
import joblib
from pycocotools.coco import COCO
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
from ours.base_data_manager import get_datactive_rank_res_path,get_error_ann_file_path,get_annotations_with_miss_json_path

def compute_apfd(fault_set:set, rankded_list):
    """
    fault_set: set/list, 真实错误idd(box_id/anno_id|img_name)
    rankded_list: list, 按可疑度排序的图像路径
    """
    # n: 排序总量
    n = len(rankded_list)
    
    TF_positions = []

    # 遍历 rankded_list 找到真实错误的位置
    for idx, ID in enumerate(rankded_list, start=1):  # 从1开始计数
        if ID in fault_set:
            TF_positions.append(idx)

    # m:错误总量
    m = len(fault_set)
    if m == 0:
        return 0.0  # 防止除零

    apfd = 1 - sum(TF_positions) / (n * m) + 1 / (2 * n)
    apfd = round(apfd,4)
    return apfd


def get_imgId_to_imgName(annotations_with_miss_json):
    imgId_to_imgName = {}
    images = annotations_with_miss_json["images"]
    for image in images:
        imgId_to_imgName[image["id"]] = image["file_name"]
    return imgId_to_imgName


def get_needed_rank_list():
    needed_rank_list = []
    for instance in ranked_list:
        gt_category_id = instance["gt_category_id"]
        if gt_category_id == bg_id:
            needed_rank_list.append(instance["image_name"])
            
        else:
            needed_rank_list.append(instance["anno_id"])
            
    return needed_rank_list

def get_missed_img_name_set(annotations_with_miss_json):
    miss_img_name_set = set()
    imgId_to_imgName = get_imgId_to_imgName(annotations_with_miss_json)
    annos = annotations_with_miss_json["annotations"]
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = imgId_to_imgName[anno["image_id"]]
            miss_img_name_set.add(img_name)
    return miss_img_name_set

def get_error_ann_id_set():
    anns = coco.loadAnns(coco.getAnnIds())
    error_ann_id_set = set()
    for ann in anns:
        if ann["fault_type"] in [1,2,3]: # cls,loc,red
            error_ann_id_set.add(ann["id"])
    return error_ann_id_set


def draw_rank(isError_list,save_path):
    # 话图看一下中毒样本在序中的分布
    distribution = [1 if flag else 0 for flag in isError_list]
    # 绘制热力图
    # 创建图形时设置较小的高度
    plt.style.use(['science','ieee'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6
    })
    plt.figure(figsize=(3, 0.5))  # 宽度为10，高度为2（可根据需要调整）
    plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    # plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('ranking',fontsize='3')
    # 调整横轴刻度字号
    plt.xticks(fontsize=3)  # 明确设置横轴刻度字号为6pt
    # plt.colorbar()
    plt.yticks([])
    plt.savefig(save_path, bbox_inches='tight', dpi=800) # pad_inches=0.0
    plt.close()

def main():
    needed_rank_list = get_needed_rank_list()
    error_ann_id_set = get_error_ann_id_set()
    missed_img_name_set =  get_missed_img_name_set(annotations_with_miss_json)
    
    union_fault_set = error_ann_id_set | missed_img_name_set

    apfd = compute_apfd(union_fault_set,needed_rank_list)
    print(f"APFD:{apfd}")

    rank_error_flag = []
    for idd in needed_rank_list:
        if idd in union_fault_set:
            rank_error_flag.append(True)
        else:
            rank_error_flag.append(False)

    draw_rank(rank_error_flag,hot_pic_save_path)
    print(f"hot_pic_save_path: {hot_pic_save_path}")


if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI|VisDrone
    # datactive 排序的idd
    ranked_list = joblib.load(get_datactive_rank_res_path(dataset_name))
    print(f"rank_res长度:{len(ranked_list)}")
    
    anno_coco_error_json_path = get_error_ann_file_path(dataset_name)
    annotations_with_miss_json_path =get_annotations_with_miss_json_path(dataset_name)

    hot_pic_save_path = os.path.join(exp_root_dir,"imgs","hot_ranking","datactive", f"{dataset_name}.png")

    coco = COCO(anno_coco_error_json_path)
    catIds = coco.getCatIds()
    bg_id = catIds[-1]+1

    with open(annotations_with_miss_json_path,'r') as f:
        annotations_with_miss_json = json.load(f)
    main()




