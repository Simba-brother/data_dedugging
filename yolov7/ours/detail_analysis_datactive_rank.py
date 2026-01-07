'''
分析DataDetective的rank
'''
import os
import json
import joblib
from pycocotools.coco import COCO
import scienceplots
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from ours.base_data_manager import exp_data_root_dir, get_datactive_rank_res_path,get_error_ann_file_path,get_annotations_with_miss_json_path


def get_image_id_to_image_name_for_coco(annos_with_miss_json:dict) -> dict:
    id2name = {}
    images = annos_with_miss_json["images"]
    for image in images:
        id2name[image["id"]] = image["file_name"] 
    return id2name


def get_needed_rank_list(bg_id):
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


def draw_rank_hot(isError_list,save_path):
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

def draw_total_rank(error_flag_list, save_path):
    # error_flag_list: 包含 0/1/2 的列表

    distribution = list(error_flag_list)

    plt.style.use(['science', 'ieee'])
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

    # 自定义 colormap：0 -> 白色；1 -> 红色；2 -> 蓝色
    cmap = ListedColormap(['white', 'red', 'blue'])
    # 使用 BoundaryNorm 保证 0/1/2 三个离散值正确落到对应颜色
    bounds = [-0.5, 0.5, 1.5, 2.5]  # 分段边界
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(3, 0.5))
    plt.imshow(
        [distribution],
        aspect='auto',
        cmap=cmap,
        norm=norm,
        interpolation='nearest'
    )

    plt.xlabel('ranking', fontsize=3)
    plt.xticks(fontsize=3)
    plt.yticks([])

    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    plt.close()


def look_annid_rank(ranked_annid_list:list[int], all_errored_annid_set:set[int]):
    pic_save_path = os.path.join(exp_data_root_dir,"temp", "annid_rank.png")
    error_flag_list = []
    for annid in ranked_annid_list:
        if annid in all_errored_annid_set:
            error_flag_list.append(1)
        else:
            error_flag_list.append(0)
    draw_rank_hot(error_flag_list,pic_save_path)
    print(f"图片保存在：{pic_save_path}")


def look_img_rank(ranked_img_name_list:list[str], all_miss_error_img_name_set:set[str]):
    pic_save_path = os.path.join(exp_data_root_dir,"temp", "image_name_rank.png")
    error_flag_list = []
    for img_name in ranked_img_name_list:
        if img_name in all_miss_error_img_name_set:
            error_flag_list.append(1)
        else:
            error_flag_list.append(0)
    draw_rank_hot(error_flag_list,pic_save_path)
    print(f"图片保存在：{pic_save_path}")


def look_total_rank(total_rank_list,error_ann_id_set,missed_img_name_set):
    pic_save_path = os.path.join(exp_data_root_dir,"temp","total_rank.png")
    total_error_set = error_ann_id_set | missed_img_name_set
    error_flags = []
    for idd in total_rank_list:
        if idd in total_error_set:
            if type(idd) is int:
                error_flags.append(1) # red, box id
            else:
                error_flags.append(2) # blue, img
        else:
            error_flags.append(0)
    draw_total_rank(error_flags, pic_save_path)
    print(f"图片保存在：{pic_save_path}")



def main():
    coco = COCO(anno_coco_error_json_path)
    catIds = coco.getCatIds()
    bg_id = catIds[-1]+1
    needed_rank_list = get_needed_rank_list(bg_id)

    ranked_annid_list = []
    ranked_img_name_list = []
    for idd in needed_rank_list:
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
    look_total_rank(needed_rank_list,error_ann_id_set,missed_img_name_set)

if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    # datactive 排序的idd
    ranked_list = joblib.load(get_datactive_rank_res_path(dataset_name))
    print(f"rank_res长度:{len(ranked_list)}")
    anno_coco_error_json_path = get_error_ann_file_path(dataset_name)
    annotations_with_miss_json_path =get_annotations_with_miss_json_path(dataset_name)
    main()




