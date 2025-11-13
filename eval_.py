import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def compute_apfd(list_A, list_B):
    """
    list_A: set/list, 真实错误图像路径(gt)
    list_B: list, 按可疑度排序的图像路径
    """
    n = len(list_B)
    list_A_set = set(list_A)
    TF_positions = []

    # 遍历 list_B 找到真实错误的位置
    for idx, img in enumerate(list_B, start=1):  # 从1开始计数
        if img in list_A_set:
            TF_positions.append(idx)

    m = len(list_A)
    if m == 0:
        return 0.0  # 防止除零

    apfd = 1 - sum(TF_positions) / (n * m) + 1 / (2 * n)
    return apfd

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
    plt.savefig(os.path.join(save_dir,"PR.png"))


def main():
    gt_error_list = list(error_record_df["img_file_name"])
    apfd = compute_apfd(gt_error_list,ranked_img_name_list)
    print(f"apfd:{apfd}")
    save_dir = os.path.join(exp_root_dir,method_name,dataset_name,model_name)
    PR_visualization(gt_error_list,ranked_img_name_list,save_dir)
    print("PR over cut off绘制完成")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    method_name = "Ours"
    dataset_name = "VOC2012"
    model_name = "FRCNN"

    ranked_img_name_list = joblib.load(f"{exp_root_dir}/{method_name}/ranked_img_name_list.joblib")
    error_record_df = pd.read_csv(f"{exp_root_dir}/datasets/{dataset_name}_error_record/error_record_simple.csv")

    main()