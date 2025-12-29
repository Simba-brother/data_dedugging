'''
分析rank结果
'''
import os
import joblib
import json
import scienceplots
import matplotlib
import matplotlib.pyplot as plt


def get_imgId_to_imgName(annotations_with_miss_json):
    imgId_to_imgName = {}
    images = annotations_with_miss_json["images"]
    for image in images:
        imgId_to_imgName[image["id"]] = image["file_name"]
    return imgId_to_imgName



def get_err_set(gt_box_json,annotations_with_miss_json):
    err_set = set()
    for img_name, boxs in gt_box_json.items():
        for box in boxs:
            if box["fault_type"] in [1,2,3]: # cls,loc,red
                err_set.add(box["box_id"])
    
    imgId_to_imgName = get_imgId_to_imgName(annotations_with_miss_json)
    annos = annotations_with_miss_json["annotations"]
    for anno in annos:
        fault_type = anno["fault_type"]
        if fault_type == 4: # mis
            img_name = imgId_to_imgName[anno["image_id"]] 
            err_set.add(img_name)
    return err_set


def compute_apfd(fault_set:set, rankded_list):
    """
    list_A: set/list, 真实错误图像路径
    list_B: list, 按可疑度排序的图像路径
    """
    n = len(rankded_list)
    
    TF_positions = []

    # 遍历 list_B 找到真实错误的位置
    for idx, ID in enumerate(rankded_list, start=1):  # 从1开始计数
        if ID in fault_set:
            TF_positions.append(idx)

    m = len(fault_set)
    if m == 0:
        return 0.0  # 防止除零

    apfd = 1 - sum(TF_positions) / (n * m) + 1 / (2 * n)
    apfd = round(apfd,4)
    return apfd

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
    err_set = get_err_set(gt_box_json,annotations_with_miss_json)
    rank_err_flag_list = []
    for idd in rank_res:
        if idd in err_set:
            rank_err_flag_list.append(True)
        else:
            rank_err_flag_list.append(False)

    apfd = compute_apfd(err_set,rank_res)
    print(f"APFD:{apfd}")
    draw_rank(rank_err_flag_list,hot_pic_save_path)
    print(f"hot_pic_save_path: {hot_pic_save_path}")

if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012, KITTI, VisDrone
    model_name = "FRCNN" # YOLOv7, FRCNN, SSD
    rank_res = joblib.load(os.path.join(exp_data_root,"Ours",dataset_name,model_name,"rank_res","rank.joblib"))
    gt_box_json_path = os.path.join(exp_data_root,"collection_indicator_bbox_level",dataset_name,"YOLOv7","gt_bboxs.json")
    annotations_with_miss_json_path = os.path.join(exp_data_root,"error_anno",dataset_name,"annotations_with_miss.json")

    with open(gt_box_json_path,'r') as f:
        gt_box_json = json.load(f)

    with open(annotations_with_miss_json_path,'r') as f:
        annotations_with_miss_json = json.load(f)
    
    hot_pic_save_path = os.path.join(exp_data_root,"imgs","hot_ranking",f"{dataset_name}_{model_name}_boxLevel.png")

    main()