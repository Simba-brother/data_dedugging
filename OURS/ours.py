import numpy as np
import pandas as pd
import os
import joblib
import topsispy as tp
import matplotlib.pyplot as plt



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


def main():
    # 可疑排序
    sorted_img_name_list = ranking_(source_data_dir,epoch_csv_dir)
    save_path = os.path.join(result_save_dir,"ranked_img_name_list.joblib")
    joblib.dump(sorted_img_name_list,save_path)
    print(f"排序完成,ranked_img_name_list保存在:{save_path}")


if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "FRCNN"
    source_data_dir = os.path.join(exp_data_root,"collection_indicator",dataset_name,model_name,"feature_fc")
    epoch_csv_dir = os.path.join(exp_data_root,"collection_indicator",dataset_name,model_name)
    result_save_dir = os.path.join(exp_data_root,"Ours")
    main()

    
    # case study






















