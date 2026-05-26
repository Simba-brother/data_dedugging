'''
对排序使用的features进行讨论
'''
import os
import numpy as np
from ours.data_organization_tools import (get_all_gids,get_g_id_to_metric,
                                          get_all_errored_g_box_id_set,get_all_correct_g_box_id_set)
from ours.base_data_manager import get_collected_gt_box_json_path,exp_data_root_dir
from ours.small_utils import read_json
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import topsispy as tp


def split_gid_clean_error(gt_json):
    error_gid_set = get_all_errored_g_box_id_set(gt_json)
    correct_gid_set = get_all_correct_g_box_id_set(gt_json)
    return correct_gid_set,error_gid_set

def build_gid_feature(all_gids:list[int],g_box_id_to_metric:dict, K:float=0.2) -> tuple:
    g_id_to_features = {}
    for g_id in g_box_id_to_metric.keys():
        conf_list = g_box_id_to_metric[g_id]["conf_list"]
        iou_list = g_box_id_to_metric[g_id]["iou_list"]
        epochs = len(conf_list)
        W_e = int(K*epochs)
        W_l = int(K*epochs)
        # 早期置信度均值，越小越可疑
        early_conf_mean = np.mean(conf_list[0:W_e])
        # 后期置信度均值，越小越可疑
        lastly_conf_mean = np.mean(conf_list[-W_l:])
        # 早期iou均值，越小越可疑
        early_iou_mean = np.mean(iou_list[0:W_e])
        # 后期iou均值，越小越可疑
        lastly_iou_mean = np.mean(iou_list[-W_l:])

        # 全局均值，越小越可疑
        conf_mean = np.mean(conf_list)
        iou_mean = np.mean(iou_list)

        conf_threshold = 0.5*lastly_conf_mean
        iou_threshold = 0.5*lastly_iou_mean

        min_e_conf = epochs
        min_e_iou = epochs
        for e in range(epochs):
            if conf_list[e] > conf_threshold:
                min_e_conf = e
                break
        for e in range(epochs):
            if iou_list[e] > iou_threshold:
                min_e_iou = e
                break
        # 起量延迟（显式刻画“涨得晚”）
        # 越大越可疑
        D_conf = min_e_conf / epochs
        D_iou = min_e_iou / epochs

        g_id_to_features[g_id] = {
            "early_conf_mean":early_conf_mean, # 早期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "early_iou_mean":early_iou_mean, # 早期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "lastly_conf_mean":lastly_conf_mean, # 后期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "lastly_iou_mean":lastly_iou_mean, # 后期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "conf_mean":conf_mean, # 全期conf mean, 越小越可疑 -> topsis分数越高 -> -1
            "iou_mean":iou_mean, # 全期iou mean, 越小越可疑 -> topsis分数越高 -> -1
            "D_conf":D_conf, # 起量延迟 conf，越大越可疑 -> topsis分数越高 -> 1
            "D_iou":D_iou, # 起量延迟 iou，越大越可疑 -> topsis分数越高 -> 1
        }
    feature_name_to_sign = {
        "early_conf_mean":-1, # 越小越可疑
        "early_iou_mean":-1,
        "lastly_conf_mean":-1,
        "lastly_iou_mean":-1,
        "conf_mean":-1,
        "iou_mean":-1,
        "D_conf":1,
        "D_iou":1
    }

    print(f"all gbox数量:{len(all_gids)}")
    print(f"matched gbox数量:{len(g_id_to_features)}")
    
    for g_id in all_gids:
        if g_id not in g_id_to_features:
            # 没有匹配上的gid都是最可疑的
            g_id_to_features[g_id] = {
                "early_conf_mean":0,
                "early_iou_mean":0,
                "lastly_conf_mean":0,
                "lastly_iou_mean":0,
                "conf_mean":0,
                "iou_mean":0,
                "D_conf":1,
                "D_iou":1, 
            }
    return (g_id_to_features,feature_name_to_sign)

def hypothesis_testing(list_1:list[float],list_2:list[float],alternative:str="two-sided"):
    def mannwhitneyu_effect_size(u_stat, n1, n2):
        """
        计算Mann-Whitney U检验的效应量r（正确版本）
        参数：
            u_stat: mannwhitneyu返回的U统计量
            n1: 第一组数据的样本量
            n2: 第二组数据的样本量
        返回：
            效应量r（绝对值），越大表示差异越明显
        """
        # 步骤1：计算U统计量的均值（零假设下的期望U值）
        mean_u = (n1 * n2) / 2
        # 步骤2：计算U统计量的标准差
        std_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
        # 步骤3：将U值转换为Z分数（标准化）
        z = (u_stat - mean_u) / std_u
        # 步骤4：计算效应量r（Cohen's r）
        r = abs(z) / np.sqrt(n1 + n2)
        return r
    
    u_stat, u_p = stats.mannwhitneyu(list_1, list_2, alternative=alternative)
    
    print(f"Mann-Whitney U检验：U值={u_stat:.3f}, p值={u_p:.3f}")
    # 计算效应量r
    r = mannwhitneyu_effect_size(u_stat, len(list_1), len(list_2))
    print(f"效应量r：{r:.3f}")

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
    plt.savefig(f"/data/mml/data_debugging_data/temp/{save_file_name}.png")

def plot_roc_auc(g_id_to_features, feature_name_to_sign, correct_gid_set, error_gid_set, save_file_name:str="roc_auc"):
    """
    把所有 feature 各自的 ROC 以及 topsis 综合 score 的 ROC 画在同一张图上。
    label: error=1, correct=0
    feature score: 把 feature 转成"越大越可疑"。sign==1 直接用；sign==-1 取相反数。
    topsis score: 所有 feature 一起传入 topsis 得到的综合分数（越大越可疑）。
    """
    plt.figure(figsize=(8, 7))
    name_to_auc = {}

    # 各个 feature 的 ROC
    for feature_name, sign in feature_name_to_sign.items():
        y_true = []
        y_score = []
        for gid in correct_gid_set:
            y_true.append(0)
            y_score.append(sign * float(g_id_to_features[gid][feature_name]))
        for gid in error_gid_set:
            y_true.append(1)
            y_score.append(sign * float(g_id_to_features[gid][feature_name]))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        name_to_auc[feature_name] = roc_auc
        plt.plot(fpr, tpr, lw=1.5, label=f"{feature_name} (AUC={roc_auc:.3f})")

    # topsis 综合 score 的 ROC
    g_id_list = sorted(g_id_to_features.keys())
    gid_to_idx = {gid: i for i, gid in enumerate(g_id_list)}
    feature_names = list(feature_name_to_sign.keys())
    sign_list = [feature_name_to_sign[fn] for fn in feature_names]
    data = np.array([[float(g_id_to_features[gid][fn]) for fn in feature_names] for gid in g_id_list])
    weights = np.ones(len(feature_names)) / len(feature_names)
    _, score_array = tp.topsis(data, weights, sign_list)
    score_array = np.asarray(score_array)

    eval_gids = [gid for gid in g_id_list if gid in correct_gid_set or gid in error_gid_set]
    y_true_t = np.array([1 if gid in error_gid_set else 0 for gid in eval_gids])
    y_score_t = np.array([score_array[gid_to_idx[gid]] for gid in eval_gids])
    fpr_t, tpr_t, _ = roc_curve(y_true_t, y_score_t)
    roc_auc_t = auc(fpr_t, tpr_t)
    name_to_auc["TOPSIS_score"] = roc_auc_t
    plt.plot(fpr_t, tpr_t, lw=2.5, color='black', label=f"TOPSIS_score (AUC={roc_auc_t:.3f})")

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='random (AUC=0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves of Box Features & TOPSIS Score ({save_file_name})')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/data/mml/data_debugging_data/temp/{save_file_name}.png", dpi=150)
    plt.close()

    print("="*60)
    print("AUC 排名:")
    for fn, a in sorted(name_to_auc.items(), key=lambda x: -x[1]):
        print(f"  {fn:20s}  AUC={a:.4f}")
    return name_to_auc

def main():
    gt_json = read_json(gt_json_path)
    all_gids = get_all_gids(gt_json)
    gid_to_metric = get_g_id_to_metric(g_box_metrics_json_path)
    g_id_to_features,feature_name_to_sign = build_gid_feature(all_gids,gid_to_metric,K=0.2)
    correct_gid_set,error_gid_set = split_gid_clean_error(gt_json)
    plot_roc_auc(g_id_to_features, feature_name_to_sign, correct_gid_set, error_gid_set,
                 save_file_name=f"roc_auc_{dataset_name}_{model_name}")
    for feature_name,sign in feature_name_to_sign.items():
        print("="*60)
        print("feature_name:",feature_name)
        print("feature sign:",sign)
        correct_data_list = []
        error_data_list = []
        for gid in correct_gid_set:
            correct_data_list.append(float(g_id_to_features[gid][feature_name]))
        for gid in error_gid_set:
            error_data_list.append(float(g_id_to_features[gid][feature_name]))
        # 数据可视化
        visualization(correct_data_list,error_data_list,feature_name)
        # 统计这两个序列的差异
        if sign == -1: 
            # 我们直觉认为 error data list < correct data list, 因为sign == -1, 说明越小topsis分数（可疑）越高，排名越靠前。
            # 单侧检验是否 correct > error
            hypothesis_testing(correct_data_list,error_data_list,"greater")
        elif sign == 1:
            # 我们直觉认为 error data list > correct data list, 因为sign == -1, 说明越小topsis分数（可疑）越高，排名越靠前。
            # 单侧检验是否 correct < error
            hypothesis_testing(correct_data_list,error_data_list,"less")
        else:
            # 我们不知 error data list 与 correct data list的大小关系
            hypothesis_testing(correct_data_list,error_data_list,"two-sided")


if __name__ == "__main__":
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "YOLOv7"
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    g_box_metrics_json_path = os.path.join(exp_data_root_dir,"collection_bbox_level",
                                           dataset_name,model_name,"collection_metric",
                                           "collection_metrics_v2.json")
    main()