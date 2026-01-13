
import os
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from ours.base_data_manager import exp_data_root_dir

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

def calc_fpr_fnr_f1(rank_list,error_set):
    cut_off = 0.4
    cut_point = int(len(rank_list) * cut_off)
    P_list = rank_list[:cut_point]
    N_list = rank_list[cut_point:]
    fp = 0
    fn = 0
    tp = 0
    correct_set = set(rank_list) - error_set
    for idd in P_list:
        if idd not in error_set:
            fp += 1
        else:
            tp += 1
    for idd in N_list:
        if idd in error_set:
            fn += 1
    fpr = fp / len(correct_set)
    fnr = fn / len(error_set)
    fpr = round(fpr,4)
    fnr = round(fnr,4)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall / (precision + recall)
    f1 = round(f1,4)
    return fpr,fnr,f1

def draw_total_rank(error_flag_list, save_path):
    # error_flag_list: 包含 0/1/2 的列表

    distribution = list(error_flag_list)
    n = len(distribution)
    split_idx = int(n * 0.4)  # 40% 位置对应的列索引

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
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(3, 0.5))
    plt.imshow(
        [distribution],
        aspect='auto',
        cmap=cmap,
        norm=norm,
        interpolation='nearest'
    )

    # 在 40% 处画一条黑色分割线（画在像素列 split_idx 的左边界）
    ax = plt.gca()
    ax.axvline(x=split_idx - 0.5, color='black', linewidth=0.5)

    plt.xlabel('ranking', fontsize=3)
    plt.xticks(fontsize=3)
    plt.yticks([])

    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    plt.close()

def look_total_rank(total_rank,all_errored_g_box_id_set,all_miss_error_img_name_set):
    pic_save_path = os.path.join(exp_data_root_dir,"temp","total_rank.png")
    total_error_set = all_errored_g_box_id_set | all_miss_error_img_name_set
    error_flags = []
    for idd in total_rank:
        if idd in total_error_set:
            if type(idd) is int:
                error_flags.append(1) # red, box id
            else:
                error_flags.append(2) # blue, img
        else:
            error_flags.append(0)
    draw_total_rank(error_flags, pic_save_path)
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
