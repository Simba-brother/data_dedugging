import os
import joblib
import pandas as pd
import scienceplots
import matplotlib
import matplotlib.pyplot as plt

def compute_apfd(ground_truth_fault_data_list:list, ranked_data_list:list):
    """
    计算 APFD (Average Percentage of Faults Detected)

    参数：
        ground_truth_fault_data_set: 可迭代对象，包含所有 fault 的标识（如 id、索引、对象等）
        ranked_data_list:           按“可疑度”从高到低排序后的完整数据列表

    返回：
        apfd_value: float, 范围大致在 [0, 1] 之间，值越大表示 ranked_data_list 越有效
    """
    fault_set = set(ground_truth_fault_data_list)
    if not fault_set:
        raise ValueError("ground_truth_fault_data_set 不能为空")

    m = len(ranked_data_list)          # 被排序的总单元数
    if m == 0:
        raise ValueError("ranked_data_list 不能为空")

    # 找到每个 fault 在 ranked_data_list 中的 rank（1-based）
    rank_list = []
    for idx, item in enumerate(ranked_data_list, start=1):
        if item in fault_set:
            rank_list.append(idx)

    n = len(fault_set)                 # fault 的数量

    # 如果有 ground truth 里的 fault 没在 ranked_data_list 里出现，按需要决定策略
    # 这里选择报错，也可以改成把缺失的 fault 视为排在最后（rank = m）
    if len(rank_list) != n:
        missing = fault_set - {ranked_data_list[i - 1] for i in rank_list}
        raise ValueError(f"以下 fault 未在 ranked_data_list 中找到: {missing}")

    # 按公式：
    # APFD = 1 - (sum(rank_i - 0.5) / (n * m)) + 1 / (2m)
    s = sum(r - 0.5 for r in rank_list)
    apfd_value = 1 - s / (n * m) + 1 / (2 * m)

    return apfd_value

def calculate_pr_metrics(list_A, list_B, thresholds=None):
    """
    计算在不同截断阈值下的 Precision 和 Recall
    
    参数:
    - list_A: 按出错概率排序的样本ID列表（越靠前越可能是错误样本）
    - list_B: 真实的错误样本ID列表
    - thresholds: 阈值列表，可以是百分比或绝对数量
    
    返回:
    - 包含不同阈值下 Precision 和 Recall 的 DataFrame
    """
    
    # 如果没有提供阈值，使用默认的百分比阈值
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,0.8,1.0]
    
    results = []
    total_error_samples = len(list_B)
    
    for threshold in thresholds:
        # 根据阈值类型确定截取数量
        if threshold <= 1:  # 百分比阈值
            n = int(len(list_A) * threshold)
            threshold_type = f"{threshold*100}%"
        else:  # 绝对数量阈值
            n = min(int(threshold), len(list_A))
            threshold_type = f"{n}个"
        
        # 获取预测的错误样本
        predicted_errors = set(list_A[:n])
        
        # 计算 TP, FP, FN
        true_positives = len(predicted_errors.intersection(list_B))
        false_positives = n - true_positives
        false_negatives = total_error_samples - true_positives
        
        # 计算 Precision 和 Recall
        precision = true_positives / n if n > 0 else 0
        recall = true_positives / total_error_samples if total_error_samples > 0 else 0
        
        # 计算 F1-score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold_type,
            'n_samples': n,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        })
    
    return pd.DataFrame(results)

def PR_visualization(gt_error_img_path_list,sorted_img_path_list,save_dir):
    results_df = calculate_pr_metrics(sorted_img_path_list,gt_error_img_path_list)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['precision'], 'b-', linewidth=2, label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], 'r-', linewidth=2, label='Recall')
    plt.xlabel('cut off')
    plt.ylabel('score')
    plt.title('Precision and Recall over cut off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir,"PR.png")
    plt.savefig(save_path)
    print(f"PR曲线保存在:{save_path}")

def draw_rank(isError_list,save_dir):
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
    save_path = os.path.join(save_dir,"rank_distribution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=800) # pad_inches=0.0
    plt.close()
    
    '''
    plt.style.use('science')
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
    distribution1 = [1 if flag else 0 for flag in isPoisoned_list_1]
    distribution2 = [1 if flag else 0 for flag in isPoisoned_list_2]
    
    # 创建2行1列的子图
    fig, axs = plt.subplots(2, 1, figsize=(3, 1.0))  # 总高度调整为1.0，每个子图高度约0.5

    # 确保axs是数组形式（即使只有一行）
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # 绘制第一个子图
    axs[0].imshow([distribution1], aspect='auto', cmap='Reds', interpolation='nearest')
    axs[0].set_xlabel('Sample ranking', fontsize=3)
    axs[0].tick_params(axis='x', labelsize=3)  # 修正：使用tick_params设置刻度标签字号
    axs[0].set_yticks([])

    # 绘制第二个子图
    axs[1].imshow([distribution2], aspect='auto', cmap='Reds', interpolation='nearest')
    axs[1].set_xlabel('Sample ranking', fontsize=3)
    axs[1].tick_params(axis='x', labelsize=3)  # 修正：使用tick_params设置刻度标签字号
    axs[1].set_yticks([])

    # 调整子图间距
    plt.subplots_adjust(hspace=0.3)  # 调整垂直间距

    # 保存为高分辨率图像
    plt.savefig(f"imgs/Motivation/SampleRanking/{file_name}", 
                bbox_inches='tight', 
                pad_inches=0.02,
                dpi=800,
                facecolor='white',
                edgecolor='none')

    plt.close()
    '''

def main(model_name=None):
    gt_error_list = list(error_record_df["img_file_name"])
    apfd = compute_apfd(gt_error_list,ranked_img_name_list)
    print(f"apfd:{apfd}")
    if model_name:
        save_dir = os.path.join(exp_root_dir,method_name,dataset_name,model_name)
    else:
        save_dir = os.path.join(exp_root_dir,method_name,dataset_name)
    PR_visualization(gt_error_list,ranked_img_name_list,save_dir)
    print("PR over cut off绘制完成")
    error_set = set(gt_error_list)
    isError_list = [img_name in error_set for img_name in ranked_img_name_list]
    draw_rank(isError_list,save_dir)
    print("ranking分布绘制完成")
    

if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    method_name = "Ours" # DataDetective|Ours
    dataset_name = "VOC2012"
    model_name = "SSD" # YOLOv7,FRCNN,SSD
    error_record_df = pd.read_csv(os.path.join(exp_root_dir,"datasets",f"{dataset_name}_error_record","error_record_simple.csv"))
    if method_name == "Ours":
        ranked_img_name_list = joblib.load(os.path.join(exp_root_dir,method_name,dataset_name, model_name,"ranked_img_name_list.joblib"))
        main(model_name)
    elif method_name == "DataDetective": # baseline_1
        ranked_img_name_list = joblib.load(os.path.join(exp_root_dir,method_name,dataset_name,"ranked_img_name_list.joblib"))
        main()
