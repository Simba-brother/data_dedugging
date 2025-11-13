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

def get_labelmap():
    
    coco = COCO(annotation_path)
    # 读取类别信息
    cats = coco.loadCats(coco.getCatIds())
    # 生成 {category_id: category_name} 映射
    labelmap = {cat['id']: cat['name'] for cat in cats}
    return labelmap

if __name__ == "__main__":
    
    exp_data_root = "/data/mml/data_debugging_data"
    crop_infer_results_path=f'{exp_data_root}/DataDetective/infer_results/crop.json'
    others_infer_results_path=f'{exp_data_root}/DataDetective/infer_results/other_objects.json'
    annotation_path=f'{exp_data_root}/datasets/VOC2012-coco/train/_annotations.coco.json'
    rank_result_save_path = f"{exp_data_root}/DataDetective/ranked_img_name_list.joblib"
    main()