import json
import torch
from pycocotools.coco import COCO
def test_1():
    coco = COCO("/data/mml/data_debugging/datasets/VOC2012-coco/train/_annotations.coco.json")
    ann_ids = coco.getAnnIds()
    annotations = coco.loadAnns(ann_ids)
    instance = annotations[0]
    image_id = instance["image_id"]
    image_info = coco.loadImgs(image_id)[0] 
    image_name = image_info['file_name']

def test_2():
    d = {"a": 3, "b": 1, "c": 5, "d": 9}

    # 按 value 从大到小得到 key 的列表
    keys_desc = sorted(d, key=d.get, reverse=True)

    print()

if __name__ == "__main__":
    test_2()
