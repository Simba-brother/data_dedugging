import os
import random
import joblib
import pandas as pd
import json
import torch
from collections import defaultdict
from pycocotools.coco import COCO

def aggregation(obj_list:list):
    '''
    聚合到img level
    '''
    image2loss = defaultdict(float)
    for obj in obj_list:
        img_name = obj["image_name"]
        loss = obj["loss"]
        image2loss[img_name] += loss
    
    sorted_img_name_list = sorted(image2loss, key=image2loss.get, reverse=True)
    return sorted_img_name_list

def calcu_afpd(ranked_results):
    fault_num = 0
    rank_sum = 0
    for i in range(len(ranked_results)):
        if ranked_results[i]['fault_type'] != fault_type["no_fault"]:
            fault_num += 1
            rank_sum += i+1
    apfd = 1-(rank_sum-1)/(fault_num*len(ranked_results))
    apfd = round(apfd,3)
    return apfd


def main():
    
    with open(crop_infer_results_path, 'r') as f:
        crop_list = json.load(f)
    with open(others_infer_results_path, 'r') as f:
        others_list = json.load(f)
    coco = COCO(annotation_path)
    imageId2boxes = defaultdict(list)
    ann_ids = coco.getAnnIds()
    annotations = coco.loadAnns(ann_ids)
    for instance in annotations:
        bbox = instance["bbox"]
        label  = instance["category_id"]
        imageId2boxes[instance["image_id"]].append([bbox,label])

    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(len(crop_list)):
        scores = crop_list[i]['full_scores'] # prob_list
        label = crop_list[i]['gt_category_id']
        loss = loss_func(torch.tensor([scores]), torch.tensor([label]))
        crop_list[i]['loss'] = loss.item()
    crop_list.extend(others_list)
    results = sorted(crop_list, key=lambda x: x['loss'], reverse=True)
    
    fault_records_df = pd.read_csv(fault_records_csv_path)
    miss_img_name_list = []
    for row_i, row in fault_records_df.iterrows():
        if row["fault_type"] == fault_type["missing_fault"]:
            miss_img_name_list.append(row["img_name"])
    for i in range(len(results)):
        if int(results[i]["gt_category_id"]) == bg_clss_id and results[i]["image_name"] in miss_img_name_list:
            results[i]["fault_type"] = fault_type["missing_fault"]
    joblib.dump(results,rank_result_save_path)
    afpd = calcu_afpd(results)
    afpd = round(afpd,3)
    return afpd


def ours_effect():

    ranked_list =joblib.load(os.path.join(exp_data_root,"DataDetective",dataset_name,"ranked_result","ranked_list.joblib"))
    apfd = calcu_afpd(ranked_list)
    print("baseline:",apfd)
    
    shuffled = random.sample(ranked_list, k=len(ranked_list))
    apfd = calcu_afpd(shuffled)
    print("random:",apfd)

    ours_img_ranked_list_path = os.path.join(exp_data_root,"Ours",dataset_name,"YOLOv7","ranked_img_name_list.joblib")
    ours_img_ranked_list = joblib.load(ours_img_ranked_list_path)
    cut_off = 0.7
    cut_num = int(len(ours_img_ranked_list)*cut_off)
    suspicious_img_list = ours_img_ranked_list[:cut_num]

    new_rank_list = []
    removed_obj_list = []
    for obj in ranked_list:
        if obj["image_name"] in suspicious_img_list:
            new_rank_list.append(obj)
        else:
            removed_obj_list.append(obj)
    new_rank_list.extend(removed_obj_list)
    apfd = calcu_afpd(new_rank_list)
    print("baseline+ours:",apfd)

'''
def get_labelmap():
    
    coco = COCO(annotation_path)
    # 读取类别信息
    cats = coco.loadCats(coco.getCatIds())
    # 生成 {category_id: category_name} 映射
    labelmap = {cat['id']: cat['name'] for cat in cats}
    return labelmap
'''

if __name__ == "__main__":
    random.seed(42)
    fault_type = {
            'no_fault': 0,
            'cls_fault': 1,
            'loc_fault': 2,
            'redundancy_fault': 3,
            'missing_fault': 4,
    }
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012" # VOC2012|VisDrone|KITTI
    crop_infer_results_path=f'{exp_data_root}/DataDetective/{dataset_name}/infer_results/crop.json'
    others_infer_results_path=f'{exp_data_root}/DataDetective/{dataset_name}/infer_results/other_objects.json'
    annotation_path=f'{exp_data_root}/datasets/{dataset_name}-coco/train/_annotations.coco_error.json'
    fault_records_csv_path = os.path.join(exp_data_root,"error_anno",dataset_name,"fault_records.csv")
    rank_result_save_path = f"{exp_data_root}/DataDetective/{dataset_name}/ranked_result/ranked_list.joblib"
    if dataset_name == "VOC2012":
        bg_clss_id = 20
    elif dataset_name == "KITTI":
        bg_clss_id = 9
    elif dataset_name == "VisDrone":
        bg_clss_id = 10

    # apfd = main()
    # print(apfd)
    ours_effect()
    
