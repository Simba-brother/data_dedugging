import json
from pycocotools.coco import COCO
from collections import defaultdict

def main():
    split = "val"
    anno_json_path = f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_coco_format/{split}/_annotations.coco.json"
    with open(anno_json_path,"r") as f:
        anno_json = json.load(f)
    DontCare_cat_id = 8

    new_ann_json = {}
    new_ann_json["images"] = []
    new_ann_json["categories"] = []
    new_ann_json["annotations"] = []

    # 剔除cat8 ann
    anns = anno_json["annotations"]
    for ann in anns:
        if ann["category_id"] == DontCare_cat_id:
            continue
        new_ann_json["annotations"].append(ann)
    
    # 剔除cat8 cat
    cats = anno_json["categories"]
    for cat in cats:
        if cat["id"] == DontCare_cat_id:
            continue
        new_ann_json["categories"].append(cat)

    new_ann_json["images"] = anno_json["images"]
    '''
    不需要了因为没有这种情况
    # 剔除只有cat8 ann 的 imgs
    img_id_to_cat_ids = defaultdict(list)
    for ann in anns:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        img_id_to_cat_ids[img_id].append(cat_id)

    # only cat8 ann img_ids
    finded_img_id_set = set()
    for img_id,cat_ids in img_id_to_cat_ids.items():
        unique_cat_id_list = list(set(cat_ids))
        if len(unique_cat_id_list) == 1 and unique_cat_id_list[0] == DontCare_cat_id:
            finded_img_id_set.add(img_id)

    images = anno_json["images"]
    for image in images:
        if image["id"] in finded_img_id_set:
            continue
        new_ann_json["images"].append(image)
    print(f"finded_img_id_set数:{len(finded_img_id_set)}")
    '''
    
    with open(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_coco_format/{split}/_annotations.coco_noDonCare.json",'w') as f:
        json.dump(new_ann_json,f)


if __name__ ==  "__main__":
    main()