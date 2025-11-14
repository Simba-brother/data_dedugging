import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

def concat_epoch_df(epoch_nums,epoch_csv_dir):
    epoch_df_list = []
    epoch_idx_list = list(range(epoch_nums))
    for epoch_idx in epoch_idx_list:
        df = pd.read_csv(f"{epoch_csv_dir}/epoch_{epoch_idx}.csv")
        epoch_df_list.append(df)
    # 合并所有epoch的数据
    all_epoch_df = pd.concat(epoch_df_list, ignore_index=True)
    return all_epoch_df

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

def pivot_metric(metric_list,all_epoch_df,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    for metric_name in metric_list:
        pivot_data = all_epoch_df.pivot_table(
            index='sample_id', # 行索引
            columns='epoch', # 列索引
            values= metric_name # 值
        )
        save_file_path = os.path.join(save_dir,f"{metric_name}_over_epoch.csv")
        pivot_data.to_csv(save_file_path)
        print(f"每个样本的_{metric_name}_over_epoch变化已保存到: {save_file_path}")

def metric_statis_features(save_dir,metric_name:str):
    df = pd.read_csv(f"{save_dir}/{metric_name}_over_epoch.csv")
    # 提取损失数据（假设第1列是sample_id，其余列是epoch损失）
    image_sample_id = df.iloc[:, 0]
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
            "slop":slope_, # loss 下降 为 负数
            "bodong":bodong,
        }
        data.append(item)
    df = pd.DataFrame(data)
    save_path = os.path.join(save_dir,f"{metric_name}_feature.csv")
    df.to_csv(save_path, index=False)
    print(f"度量{metric_name}的feature已保存到: {save_path}")


def build_dataset_features(epoch_csv_dir,save_dir,epoch_num:int):
    # 把这些epoch csv拼接成一个大的csv
    all_epoch_df = concat_epoch_df(epoch_num,epoch_csv_dir)
    # 每个metric over epoch分割成一个 feature csv
    pivot_metric(metric_list,all_epoch_df,save_dir)
    for metric_name in metric_list:
        metric_statis_features(save_dir,metric_name)
    # 把所有的metric feature拼接成一个整体feature
    feature_splice(save_dir,metric_list) # 特征拼接
    print("build dataset featrues 完成")

if __name__ == "__main__":
    '''
    一般先行脚本为 epochCSV_xiufu.py
    '''
    exp_data_root = "/data/mml/data_debugging_data"
    exp_stage_name = "collection_indicator"
    dataset_name = "VOC2012"
    model_name = "SSD"
    epoch_csv_dir = os.path.join(exp_data_root,exp_stage_name,dataset_name,model_name)
    save_dir = os.path.join(epoch_csv_dir,"feature_gc")
    os.makedirs(save_dir,exist_ok=True)
    if model_name == "FRCNN":
        metric_list = ["loss_box","loss_obj","loss_cls","loss","conf_avg"]
    elif model_name == "SSD":
        metric_list = ["loss_box","loss_objcls","loss","conf_avg"]
    # 构建特征集
    build_dataset_features(epoch_csv_dir,save_dir,epoch_num=50)