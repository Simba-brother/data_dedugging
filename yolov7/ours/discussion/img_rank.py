import os
import json
import math
import numpy as np
import topsispy as tp
from collections import defaultdict, Counter
from ours.rank_analyse.common import draw_rank_hot
from ours.small_utils import read_json,save_json_file
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ks_2samp

def box_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def cluster_boxes_by_iou(p_boxes, iou_threshold=0.6, require_same_cls=False):
    n = len(p_boxes)
    if n == 0:
        return []

    graph = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            iou = box_iou(p_boxes[i]["bbox"], p_boxes[j]["bbox"])
            if iou < iou_threshold:
                continue
            if require_same_cls and p_boxes[i]["predicted_cls"] != p_boxes[j]["predicted_cls"]:
                continue
            graph[i].append(j)
            graph[j].append(i)

    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(p_boxes[cur])
            for nei in graph[cur]:
                if not visited[nei]:
                    visited[nei] = True
                    stack.append(nei)
        clusters.append(comp)

    return clusters


def box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def cluster_features(cluster, total_epoch_num=5):
    confs = [b["conf"] for b in cluster]
    clses = [b["predicted_cls"] for b in cluster]
    epochs = [b["epoch"] for b in cluster]

    size = len(cluster)
    mean_conf = float(np.mean(confs))
    max_conf = float(np.max(confs))

    epoch_count = len(set(epochs))
    epoch_coverage = epoch_count / total_epoch_num

    most_common_cls_count = Counter(clses).most_common(1)[0][1]
    class_consistency = most_common_cls_count / size

    pairwise_ious = []
    for i in range(size):
        for j in range(i + 1, size):
            pairwise_ious.append(box_iou(cluster[i]["bbox"], cluster[j]["bbox"]))
    mean_iou = float(np.mean(pairwise_ious)) if pairwise_ious else 0.0

    centers = np.array([box_center(b["bbox"]) for b in cluster], dtype=float)
    if len(centers) <= 1:
        center_std = 1e6   # 单点簇视为稳定性很弱
    else:
        std_x = np.std(centers[:, 0])
        std_y = np.std(centers[:, 1])
        center_std = float(math.sqrt(std_x**2 + std_y**2))

    
    center_stability = 1.0 / (1.0 + center_std)

    normalized_cluster_size = min(size / total_epoch_num, 1.0)
    
    return {
        "cluster_size": size,
        "normalized_cluster_size": normalized_cluster_size,
        "epoch_coverage": epoch_coverage,
        "mean_conf": mean_conf,
        "max_conf": max_conf,
        "class_consistency": class_consistency,
        "mean_iou": mean_iou,
        "center_std": center_std,
        "center_stability": center_stability
    }

def score_cluster(feat):
    """
    输入: 单个cluster的特征字典
    输出: 该cluster的综合分数，越大表示越像潜在miss annotation区域
    """
    score = (
        0.28 * feat["mean_conf"]
        + 0.22 * feat["epoch_coverage"]
        + 0.18 * feat["normalized_cluster_size"]
        + 0.14 * feat["class_consistency"]
        + 0.12 * feat["mean_iou"]
        + 0.06 * feat["center_stability"]
    )
    return float(score)

def image_features_from_clusters(clusters, total_epoch_num=5):
    feature_names = [
        "cluster_count", # F1
        "max_cluster_size", # F2
        "max_epoch_coverage", # F3
        "max_mean_conf", # # F4
        "max_conf", # F5
        "max_class_consistency", # F6
        "max_mean_iou", # F7
        "best_center_stability" # F8
    ]

    if len(clusters) == 0:
        return [0.0] * 8, feature_names

    feats = [cluster_features(c, total_epoch_num=total_epoch_num) for c in clusters]
    # cluster_scores = [score_cluster(f) for f in feats] F9


    

    F1 = float(len(clusters))
    F2 = float(max(f["cluster_size"] for f in feats))
    F3 = float(max(f["epoch_coverage"] for f in feats))
    F4 = float(max(f["mean_conf"] for f in feats))
    F5 = float(max(f["max_conf"] for f in feats))
    F6 = float(max(f["class_consistency"] for f in feats))
    F7 = float(max(f["mean_iou"] for f in feats))
    F8 = float(max(f["center_stability"] for f in feats)) # 越稳定越好
    # F9 = float(max(cluster_scores))   # strongest_cluster_score
    return [F1, F2, F3, F4, F5, F6, F7, F8],feature_names



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



def main():
    data = read_json(img_to_nomatched_pboxs_json_path)

    img_names = []
    X = []
    y = []

    # 你可以自动计算总epoch数
    all_epochs = set()
    for img_name, info in data.items():
        for b in info["No_matched_p_box_list"]:
            all_epochs.add(b["epoch"])
    total_epoch_num = len(all_epochs) if len(all_epochs) > 0 else 1

    for img_name, info in data.items():
        p_boxes = info["No_matched_p_box_list"]
        clusters = cluster_boxes_by_iou(
            p_boxes,
            iou_threshold=0.6,
            require_same_cls=False
        )

        feats, feature_names = image_features_from_clusters(clusters, total_epoch_num=total_epoch_num)

        img_names.append(img_name)
        X.append(feats)
        y.append(info["with_miss_fault_flag"])

    # weights = [0.08, 0.16, 0.18, 0.16, 0.12, 0.10, 0.12, 0.08]
    X = np.array(X)
    y = np.array(y)
    weights = np.ones(X.shape[1]) / X.shape[1]
    signs = [1] * X.shape[1]
    # 基于topsis获得clusters的score
    best_id, score_array = tp.topsis(X, weights, signs)
    # 从大到小排序并返回索引
    sorted_id = np.argsort(score_array, kind="mergesort")[::-1]
    ranked_id_list = [int(id) for id in sorted_id]
    ranked_score_list = []
    for id in ranked_id_list:
        ranked_score_list.append(score_array[id])

    
    ranked_imgs = []
    fault_set = set()
    ranked_flag_list = []
    for id in ranked_id_list:
        img_name = img_names[id]
        ranked_imgs.append(img_name)
        if data[img_name]["with_miss_fault_flag"] == 1:
            fault_set.add(img_name)
            ranked_flag_list.append(1)
        else:
            ranked_flag_list.append(0)

    fault_img_count = len(fault_set)
    all_img_count = len(ranked_imgs)
    print(f"总共的img数量:{all_img_count}")
    print(f"包含miss fault的img数量:{fault_img_count}")
    apfd = compute_apfd(fault_set,ranked_imgs)
    print(f"apfd:{apfd}")

    # rank可视化
    save_dir = os.path.join(exp_root_dir,"temp")
    save_file_path = os.path.join(save_dir,"img_rank1.png")
    draw_rank_hot(ranked_flag_list,save_file_path)
    for i, name in enumerate(feature_names):
        x = X[:, i]
        x1 = x[y == 1]
        x0 = x[y == 0]
        visualization(x0,x1,name)
    importance_df = analyze_feature_importance(X, y, feature_names)

    
    print(importance_df)



def cohens_d(x1, x0):

    mean1 = np.mean(x1)
    mean0 = np.mean(x0)

    std1 = np.std(x1, ddof=1)
    std0 = np.std(x0, ddof=1)

    n1 = len(x1)
    n0 = len(x0)

    pooled_std = np.sqrt(
        ((n1-1)*std1**2 + (n0-1)*std0**2) / (n1+n0-2)
    )

    if pooled_std == 0:
        return 0

    return (mean1 - mean0) / pooled_std

def visualization(correct_list,error_list,save_file_name:str):
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 箱线图
    ax1.boxplot([correct_list, error_list], labels=['correct', 'error'])
    ax1.set_title('Box plot: Data distribution comparison')
    ax1.set_ylabel('Numerical value')
    # 直方图+核密度估计（KDE）
    # 1. 可视化：箱线图（看分布位置、离散程度）+ 直方图（看分布形态）
    
    sns.histplot(correct_list, kde=True, ax=ax2, label='correct', alpha=0.5)
    sns.histplot(error_list, kde=True, ax=ax2, label='error', alpha=0.5)
    ax2.set_title('Histogram +KDE: Shape of distribution')
    ax2.legend()
    plt.savefig(f"/data/mml/data_debugging_data/temp/img_rank/{save_file_name}.png")


def analyze_feature_importance(X, y, feature_names):

    results = []

    for i, name in enumerate(feature_names):

        x = X[:, i]

        x1 = x[y == 1]
        x0 = x[y == 0]

        # AUC
        try:
            auc = roc_auc_score(y, x)
        except:
            auc = 0.5

        auc_importance = abs(auc - 0.5)

        # KS
        ks_stat, _ = ks_2samp(x1, x0)

        # Cohen's d
        d = cohens_d(x1, x0)

        results.append({
            "feature": name,
            "AUC": auc,
            "AUC_importance": auc_importance,
            "KS": ks_stat,
            "Cohen_d": d
        })

    df = pd.DataFrame(results)

    # mutual information
    mi = mutual_info_classif(X, y)

    df["MutualInfo"] = mi

    return df.sort_values("AUC_importance", ascending=False)



if __name__ == "__main__":
    exp_root_dir= "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    epochs = 50
    img_to_nomatched_pboxs_json_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",
                                                    dataset_name,model_name,"img_to_nomatched_pboxs.json")
    main()
    
    