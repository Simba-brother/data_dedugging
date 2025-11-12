import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class COCO_type_Dataset():
    def __init__(self, 
                 img_root,
                 annotation_file_path, 
                 transforms=None):
        self.img_root = img_root
        self.annotation_file_path = annotation_file_path
        self.transforms = transforms
        self.annotation = json.load(open(self.annotation_file_path, "r"))
        self.img_id2img_name = self.get_img_id2img_name()
        self.img_name2objects = self.get_img_name2objects()
        self.img_id_list = list(self.img_id2img_name.keys())
        print()

    def get_img_id2img_name(self):
        img_id2img_name = {}
        images = self.annotation["images"]
        for image in images:
            image_id = image["id"]
            image_name = image["file_name"]
            img_id2img_name[image_id] = image_name
        return img_id2img_name

    def get_img_name2objects(self):
        img_name_2_objects = {}
        objs = self.annotation["annotations"]
        for obj in objs:
            image_id = obj["image_id"]
            image_name = self.img_id2img_name[image_id]
            img_name_2_objects[image_name] = obj
        return img_name_2_objects

    def __getitem__(self, idx:int):
        img_id = self.img_id_list[idx]
        img_name = self.img_id2img_name[img_id]
        img_path = os.path.join(self.img_root,img_name)
        img = Image.open(img_path).convert("RGB")
        objects = self.img_name2objects[img_name]
        boxes = [x['bbox'] for x in objects]
        labels = [x['category_id'] for x in objects]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["image_name"] = img_name
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_id_list)

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
if __name__ == "__main__":
    train_root = "/data/mml/data_debugging/datasets/VOC2012-coco/train"
    train_annotation_json_path = "/data/mml/data_debugging/datasets/VOC2012-coco/train/_annotations.coco_correct.json"
    d = COCO_type_Dataset(train_root, train_annotation_json_path, None)
    print(d)