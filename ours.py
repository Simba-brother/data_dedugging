import numpy as np
import pandas as pd
import os
from scipy.stats import linregress
import topsispy as tp
import matplotlib.pyplot as plt

def concat_all_epoch_df(epoch_nums,epoch_csv_dir,save_dir):
    epoch_df_list = []
    epoch_idx_list = list(range(epoch_nums))
    for epoch_idx in epoch_idx_list:
        df = pd.read_csv(f"{epoch_csv_dir}/epoch_{epoch_idx}.csv")
        epoch_df_list.append(df)
    # 合并所有epoch的数据
    all_data = pd.concat(epoch_df_list, ignore_index=True)
    save_file_path = os.path.join(save_dir,"all_epochs.csv")
    all_data.to_csv(save_file_path, index=False)
    print(f"\n所有epoch的数据已被concat保存到: {save_file_path}")

def pivot_metric(metric_list,save_dir):
    all_data = pd.read_csv(f"{save_dir}/all_epochs.csv")
    os.makedirs(save_dir,exist_ok=True)
    for metric_name in metric_list:
        pivot_data = all_data.pivot_table(
            index='sample_id', # 行索引
            columns='epoch', # 列索引
            values= metric_name # 值
        )
        save_file_path = os.path.join(save_dir,f"{metric_name}_over_epoch.csv")
        pivot_data.to_csv(save_file_path)
        print(f"每个样本的_{metric_name}_over_epoch变化已保存到: {save_file_path}")

def metric_statis_features(save_dir,metric_name:str):
    '''
    metric_name:str:"loss_box","loss_obj","loss_cls","loss","conf_avg"
    '''
    df = pd.read_csv(f"{save_dir}/{metric_name}_over_epoch.csv")
    # 提取损失数据（假设第1列是sample_id，其余列是epoch损失）
    image_names = df.iloc[:, 0]
    metric_data = df.iloc[:, 1:].values # (nums,epochs)

    data = []
    for i, metrics in enumerate(metric_data):
        epochs = np.arange(len(metrics))
        # 计算均值
        mean_ = metrics.mean()
        '''
        std_ = metrics.std()
        max_ = metrics.max()
        min_ = metrics.min()
        '''
        slope_, _, _, _, _ = linregress(epochs, metrics) # 斜率
        bodong = np.std(np.diff(metrics)) # 波动性（相邻 epoch 变化量的标准差）
        item = {
            "sample_id":i,
            "mean":mean_,
            # "std":std_,
            # "max":max_,
            # "min":min_,
            "slop":slope_,
            "bodong":bodong,
        }
        data.append(item)
    df = pd.DataFrame(data)
    save_path = os.path.join(save_dir,f"{metric_name}_feature.csv")
    df.to_csv(save_path, index=False)
    print(f"度量{metric_name}的feature已保存到: {save_path}")

def feature_splice(save_dir,metric_name_list):
    # metric_name_list:["loss_box","loss_obj","loss_cls","loss","conf_avg"]
    files = []
    for metirc_name in metric_name_list:
        files.append(f"{save_dir}/{metirc_name}_feature.csv")

    # 读取第一个 CSV（保留 sample_id）
    df_main = pd.read_csv(files[0])
    # 依次拼接后面 4 个 CSV
    for i, f in enumerate(files[1:], start=1):
        df_tmp = pd.read_csv(f)
        # 删除重复的 sample_id 列
        df_tmp = df_tmp.drop(columns=["sample_id"], errors='ignore')
        # 重命名列加后缀
        df_tmp = df_tmp.add_suffix(f"_{i}")
        # 合并（按行索引对齐）
        df_main = pd.concat([df_main, df_tmp], axis=1)
    save_path = os.path.join(save_dir,"merged_feature.csv")
    # 保存结果
    df_main.to_csv(save_path, index=False)
    print(f"已生成合并后的 merged_feature,保存在{save_path}")

def build_dataset_features(epoch_csv_dir,save_dir,epoch_num:int):
    # 把这些epoch csv拼接成一个大的csv
    concat_all_epoch_df(epoch_num,epoch_csv_dir,save_dir)
    # 每个metric over epoch分割成一个 feature csv
    metric_list = ["loss_box","loss_obj","loss_cls","loss","conf_avg"]
    pivot_metric(metric_list,save_dir)
    for metric_name in metric_list:
        metric_statis_features(save_dir,metric_name)
    # 吧所有的metric feature拼接成一个整体feature
    feature_splice(save_dir,metric_list) # 特征拼接
    print("build dataset featrues 完成")


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

def ranking_(save_dir,epoch_dir):
    merged_feature_df = pd.read_csv(f"{save_dir}/merged_feature.csv")
    sample_ids = merged_feature_df.iloc[:, 0]
    dataset_features = merged_feature_df.iloc[:, 1:].values # (nums,features)
    '''
    ["loss_box","loss_obj","loss_cls","loss","conf_avg"]
    loss_box:
        mean:越大越可疑 True 0
        slop:越大越可疑 True 1
        bodong:越大越可疑 True 2
    loss_obj:
        mean_1:越大越可疑 True 3
        slop_1:越大越可疑 True 4
        bodong_1:越大越可疑 True 5
    loss_cls:
        mean_2:越大越可疑 True 6
        slop_2:越大越可疑 True 7
        bodong_2:越大越可疑 True 8
    loss:
        mean_3:越大越可疑 True 9
        slop_3:越大越可疑 True 10
        bodong_3:越大越可疑 True 11
    conf_avg:
        mean_4:越小越可疑 False 12
        slop_4:越大越可疑 True 13
        bodong_4:越大越可疑 True 14
    '''

    # feature_indices = [0,4,5,6,10,11,12,16,17,18,22,23]
    # dataset_subfeatures = dataset_features[:,feature_indices]

    feature_signs = [1 for _ in range(15)]
    feature_signs[-3] = -1
    n_features = len(feature_signs)
    weights = np.ones(n_features) / n_features
    best_id, score_array = tp.topsis(dataset_features, weights, feature_signs)
    # 从大到小排序并返回索引
    sorted_sample_indices = np.argsort(score_array)[::-1]

    sample_id_to_imgname = {}
    epoch_0_df = pd.read_csv(f"{epoch_dir}/epoch_0.csv")
    for row_i,row in epoch_0_df.iterrows():
        sample_id = row["sample_id"]
        image_name = row["image_name"]
        sample_id_to_imgname[sample_id] = image_name

    sorted_img_name_list = []
    for sample_id in sorted_sample_indices:
        sorted_img_name_list.append(sample_id_to_imgname[sample_id])
    return sorted_img_name_list

def get_error_img_name_list():
    error_img_name_list = []
    record_df = pd.read_csv("/data/mml/data_debugging/datasets/VOC2012_error_record/error_record_simple.csv")
    for row_i, row in record_df.iterrows():
        img_file_name = row["img_file_name"]
        error_img_name_list.append(img_file_name)
    return error_img_name_list

def PR_visualization(gt_error_img_path_list,sorted_img_path_list):
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
    plt.savefig("exp_results/imgs/PR_FRCNN.png")

def compute_apfd(list_A, list_B):
    """
    list_A: set/list, 真实错误图像路径
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

def case_study(epoch_nums):
    cases = {
        "class_error_case":{
            "img_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/images/train/2 (155).jpg",
            "label_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/labels/train/2 (155).txt"
        },
        "box_error_case":{
            "img_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/images/train/3 (191).jpg",
            "label_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/labels/train/3 (191).txt"
        },
        "drop_error_case":{
            "img_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/images/train/4 (365).jpg",
            "label_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/labels/train/4 (365).txt"
        },
        "redundancy_error_case":{
            "img_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/images/train/4 (129).jpg",
            "label_file":"/home/mml/workspace/ultralytics/datasets/african-wildlife/labels/train/4 (129).txt"
        }
    }
    record_df = pd.read_csv("exp_results/datas/modified_labels_record_2.csv")
    error_label_file_list = list(record_df["label_file_path"])

    metric_over_epoch_df = pd.read_csv("exp_results/datas/sample_training_metrics_2/sample_box_loss_by_epoch.csv")

    imgPath_to_sampleId = {}
    epoch_0_df = pd.read_csv("exp_results/datas/sample_training_metrics_2/epoch_0_sample_metrics.csv")
    for row_i,row in epoch_0_df.iterrows():
        image_path = row["image_path"]
        sample_id = row["sample_id"]
        imgPath_to_sampleId[image_path] = sample_id

    error_sample_id_list = []
    for error_label_file in error_label_file_list:
        error_img_file = error_label_file.replace("labels", "images").replace(".txt", ".jpg")
        error_img_file = os.path.join("/home/mml/workspace/ultralytics/",error_img_file)
        error_sample_id_list.append(imgPath_to_sampleId[error_img_file])

    img_file = cases["class_error_case"]["img_file"]
    sample_id = imgPath_to_sampleId[img_file]
    error_metric_over_epoch = metric_over_epoch_df.iloc[sample_id, 1:].values

    metric_over_epoch = metric_over_epoch_df.iloc[:, 1:].values # (nums,features)

    # 过滤掉指定行
    filtered_data = np.delete(metric_over_epoch, error_sample_id_list, axis=0)
    # 计算过滤后每列的均值
    means_over_epoch = np.mean(filtered_data, axis=0)

    plt.figure(figsize=(10, 6))
    epoch_list = list(range(epoch_nums))
    plt.plot(epoch_list, error_metric_over_epoch, 'r-', linewidth=2, label='redundancy_error_case')
    plt.plot(epoch_list, means_over_epoch, 'g-', linewidth=2, label='clean_mean')
    plt.xlabel('epoch')
    plt.ylabel('box_loss')
    plt.title('box_loss over epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exp_results/datas/redundancy_error_case.png")

if __name__ == "__main__":
    epoch_csv_dir = "/data/mml/data_debugging/collection_indicator/VOC2012/FRCNN"
    save_dir = os.path.join(epoch_csv_dir,"feature_gc")
    os.makedirs(save_dir,exist_ok=True)
    # 构建特征集
    build_dataset_features(epoch_csv_dir,save_dir,epoch_num=50)
    print("构建完成")
    # 排序
    sorted_img_name_list = ranking_(save_dir,epoch_csv_dir)
    print("排序完成")
    # 评估排序结果
    gt_error_img_name_list = get_error_img_name_list()
    print("gt error img path list获得完成")
    PR_visualization(gt_error_img_name_list,sorted_img_name_list)
    print("PR over cut off绘制完成")
    apfd = compute_apfd(gt_error_img_name_list,sorted_img_name_list)
    print(f"apfd:{apfd}")
    # case study






















