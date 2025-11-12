import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict

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
        self.img_name_list = list(self.img_name2objects.keys())

    def get_img_id2img_name(self):
        img_id2img_name = {}
        images = self.annotation["images"]
        for image in images:
            image_id = image["id"]
            image_name = image["file_name"]
            img_id2img_name[image_id] = image_name
        return img_id2img_name

    def get_img_name2objects(self):
        img_name_2_objects = defaultdict(list)
        objs = self.annotation["annotations"]
        for obj in objs:
            image_id = obj["image_id"]
            image_name = self.img_id2img_name[image_id]
            img_name_2_objects[image_name].append(obj)
        
        return img_name_2_objects

    def __getitem__(self, idx:int):
        img_name = self.img_name_list[idx]
        img_path = os.path.join(self.img_root,img_name)
        img = Image.open(img_path).convert("RGB")
        objects = self.img_name2objects[img_name]
        # bbox xywh -> xmin,ymin,xmax,ymax
        for obj in objects:
            x,y,w,h = obj["bbox"]
            x1,y1,x2,y2 = x,y,x+w,y+h
            obj["bbox"] = [x1,y1,x2,y2]
        boxes = [x['bbox'] for x in objects]
        # area = [x['area'] for x in objects]

        labels = [x['category_id'] for x in objects]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)

        target["image_name"] = img_name
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0])
        # target["area"] = torch.as_tensor(area, dtype=torch.int64)
        iscrowd = [0 for _ in range(len(boxes))] # 都常规obj
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64) 
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_name_list)

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))