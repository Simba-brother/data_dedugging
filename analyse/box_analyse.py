import os
import pandas as pd
from collections import defaultdict
from analyse.draw import draw_box

def extract_feature(metric_name,stat_name):
    metric_feature_df = pd.read_csv(os.path.join(exp_root_dir,"collection_indicator", dataset_name, model_name, "feature_gc", f"{metric_name}_feature.csv")) 
    feature_list = metric_feature_df[stat_name].tolist()
    return feature_list


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
        for stat_name in stat_name_list:
            feature_name  = metric_name+"_"+stat_name
            feature_list = extract_feature(metric_name,stat_name)
            gt_error_feature_list = [feature_list[i] for i in gt_error_sample_id_list]
            gt_correct_feature_list = [feature_list[i] for i in gt_correct_sample_id_list]
            save_dir = os.path.join(exp_root_dir,"Ours",dataset_name,model_name,"boxPlot")
            os.makedirs(save_dir,exist_ok=True)
            save_file_path = os.path.join(save_dir,f"{feature_name}.png")
            draw_box(gt_error_feature_list,gt_correct_feature_list,feature_name,save_file_path)
            print(f"Boxplot:{dataset_name}|{model_name}|{feature_name}")
            print(f"save_path:{save_file_path}")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "SSD" # YOLOv7,FRCNN,SSD
    error_record_df = pd.read_csv(os.path.join(exp_root_dir,"datasets",f"{dataset_name}_error_record","error_record_simple.csv"))
    epoch0_csv_path = os.path.join(exp_root_dir,"collection_indicator",dataset_name,model_name,"epoch_0.csv")
    if model_name in["YOLOv7","FRCNN"]:
        metric_name_list =  ["loss_box","loss_obj","loss_cls","loss","conf_avg"]
    elif model_name in["SSD"]:
        metric_name_list =  ["loss_box","loss_objcls","loss","conf_avg"]
    stat_name_list = ["mean","slop","bodong"]
    main()
    