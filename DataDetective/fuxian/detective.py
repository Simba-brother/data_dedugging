import joblib
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
    # 越靠前的imgname越可疑
    sorted_img_name_list = aggregation(crop_list)
    joblib.dump(sorted_img_name_list,rank_result_save_path)
    # 按照obj loss从大到小排序
    # results = sorted(crop_list, key=lambda x: x['loss'], reverse=True)

if __name__ == "__main__":
    labelmap = {1: 'person', 2: 'car', 3: 'chair', 4: 'book', 5: 'bottle', 6: 'cup', 7: 'dining table', 8: 'traffic light'}
    crop_infer_results_path='./casestudydata/crop_test_inf.json',
    others_infer_results_path='./casestudydata/mask_others_test_inf.json',
    annotation_path='./dataset/COCO/casestudy_test.json'
    rank_result_save_path = "/data/mml/data_debugging/DataDetective/ranked_img_name_list.joblib."