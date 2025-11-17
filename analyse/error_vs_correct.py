import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt




def get_error_sample_id_list(error_record_df,imagename2sampleid):
    img_name_list = error_record_df["img_file_name"].tolist()
    error_sample_id_list = []
    for img_name in img_name_list:
        error_sample_id_list.append(imagename2sampleid[img_name])
    return error_sample_id_list

def getf_correct_sample_id_list(error_sample_id_list,sampleId2imagename):
    correct_sample_id_list = []
    all_sample_id_list = list(sampleId2imagename.keys())
    for sample_id in all_sample_id_list:
        if sample_id not in error_sample_id_list:
            correct_sample_id_list.append(sample_id)
    return correct_sample_id_list


def get_sampleId22imagename(epoch0_csv_path):
    sampleId2imagename = defaultdict(str)
    imagename2sampleid = defaultdict(int)
    epoch0_df = pd.read_csv(epoch0_csv_path) 
    sample_id_list = epoch0_df["sample_id"].tolist() 
    image_name_list = epoch0_df["image_name"].tolist() 
    for sample_id,image_name in zip(sample_id_list,image_name_list):
        sampleId2imagename[sample_id] = image_name
        imagename2sampleid[image_name] = sample_id
    return sampleId2imagename,imagename2sampleid




def main():
    sampleId2imagename,imagename2sampleid = get_sampleId22imagename(epoch0_csv_path)
    gt_error_sample_id_list = get_error_sample_id_list(error_record_df,imagename2sampleid)
    gt_correct_sample_id_list = getf_correct_sample_id_list(gt_error_sample_id_list,sampleId2imagename)
    for metric_name in metric_name_list:
        df = pd.read_csv(os.path.join(exp_root_dir,"collection_indicator",dataset_name,model_name,"feature_gc", f"{metric_name}_over_epoch.csv"))
        # 取出 loss 列名（第2~51列）
        loss_cols = df.columns[1:51]   # 50 个 epoch
        # 筛选 correct / error 的数据
        df_correct = df[df["sample_id"].isin(gt_correct_sample_id_list)][loss_cols]
        df_error = df[df["sample_id"].isin(gt_error_sample_id_list)][loss_cols]

        # 计算逐 epoch 均值（列均值）
        correct_mean = df_correct.mean(axis=0)
        error_mean = df_error.mean(axis=0)

        # 准备 x 轴 epoch
        epochs = range(1, 51)
        # 绘图
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, correct_mean.values, label="Correct Samples", marker='o')
        plt.plot(epochs, error_mean.values, label="Error Samples", marker='o')

        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.title("Mean Loss of Samples  Over 50 Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_dir = os.path.join(exp_root_dir,"imgs","correct_vs_error")
        save_path = os.path.join(save_dir,f"{dataset_name}_{model_name}.png")
        plt.savefig(save_path)
        print("correct_vs_error_loss",save_path)

if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI" # VOC2012|VisDrone|KITTI
    model_name = "SSD" # YOLOv7,FRCNN,SSD
    error_record_df = pd.read_csv(os.path.join(exp_root_dir,"datasets",f"{dataset_name}_error_record","error_record_simple.csv"))
    epoch0_csv_path = os.path.join(exp_root_dir,"collection_indicator",dataset_name,model_name,"epoch_0.csv")
    if model_name in["YOLOv7","FRCNN"]:
        # metric_name_list =  ["loss_box","loss_obj","loss_cls","loss","conf_avg"]
        metric_name_list = ["loss"]
    elif model_name in["SSD"]:
        # metric_name_list =  ["loss_box","loss_objcls","loss","conf_avg"]
        metric_name_list = ["loss"]
    main()
    