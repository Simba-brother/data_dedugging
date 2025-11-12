import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image, ImageFilter

class Inference_classificationDataSet(Dataset):
    def __init__(self, 
                img_root,
                annotation_path,
                mask_type,
                transforms=None):
        self.img_root = img_root
        self.mask_type = mask_type # other object | all background
        self.coco = COCO(annotation_path)
        ann_ids = self.coco.getAnnIds()
        annotations = self.coco.loadAnns(ann_ids)
        self.instances = []
        for instance in annotations:
            xmin, ymin, width, height = instance["bbox"]
            xmax = xmin + width
            ymax = ymin + height
            instance["bbox"] = [int(xmin),int(ymin),int(xmax),int(ymax)]
            # 保证box是个有效的整数矩形
            if instance["bbox"][0] == instance["boxes"][2]:
                instance["bbox"][2] += 1
            if instance["bbox"][1] == instance["boxes"][3]:
                instance["bbox"][3] += 1
        # 一个图像映射到其boxes
        self.imageid2boxes = defaultdict(list)
        # 遍历每个实例
        for instance in annotations:
            self.instances.append(instance)
            self.imageid2boxes[instance["image_id"]].append(instance["bbox"])
        print(f"INFO: {len(self.instances)} instances nomissing-loaded.")

        if self.mask_type == 'other objects' or self.mask_type == 'mask all':
            bg_instances = []
            bg_image_names = []
            for instance in self.instances:
                # 拿到该实例所属image path
                image_id = instance["image_id"]
                image_info = self.coco.loadImgs(image_id)[0] 
                image_name = image_info['file_name']
                item = {}
                item["image_id"] = image_id
                item["image_name"] = image_name
                item["image_size"] = [image_info["width"],image_info["height"]]
                item["bbox"] = instance["bbox"]
                item["label"] = 0
                item["image_id"] = instance["image_id"]
                item["area"] = self.caclu_area(instance["bbox"])
                item["iscrowd"] = 0
                # item["fault_type"] = instance["fault_type"]

                if item["image_name"] not in bg_image_names:
                    bg_instances.append(item)
                    bg_image_names.append(item["image_name"])
            self.instances = bg_instances #只需推理背景部分即可
            print(f"INFO: {len(self.instances)} instances only bkg-loaded.")
            # self.fault_gt_instances = bkg_fault_gt_instances
            # merge fault_gt_instances and bkg_fault_gt_instances
            # self.fault_gt_instances.extend(bkg_fault_gt_instances)
        self.transforms = transforms

    def caclu_area(self,bbox:list):
        xmin,ymin,xmax,ymax = bbox
        area = int((xmax-xmin) * (ymax - ymin))
        return area
    
    def gaussian_blur(self, img, box):
        if box[2] - box[0] <= 0:
            box[2] = box[0] + 1
        if box[3] - box[1] <= 0:
            box[3] = box[1] + 1
        img_box = img.crop(box)
        img_box = img_box.filter(ImageFilter.GaussianBlur(radius=20))
        img.paste(img_box, box)
        return img

    def __getitem__(self, idx):
        instance = self.instances[idx]
        img_path = os.path.join(self.img_root, instance['image_name'])
        img = Image.open(img_path).convert("RGB")
        boxes = instance["bbox"]
        in_boxes_list = []
        label = instance["category_id"]
        img_need = None
        if label != 0:
            img_need = img.crop(boxes)
        if self.mask_type == 'mask others':
            for box in self.imageid2boxes[instance["image_id"]]:
                box = [int(i) for i in box]
                if box == boxes and label != 0:
                    continue
                # Gaussian blur the box area of the image
                # if box belong to the part of boxes
                if box[0] > boxes[0] and box[1] > boxes[1] and box[2] < boxes[2] and box[3] < boxes[3]:
                    in_boxes_list.append(box)
                else:
                    img = self.gaussian_blur(img, box)
            if label != 0:
                img.paste(img_need, boxes)

            for box in in_boxes_list:
                img = self.gaussian_blur(img, box)


        elif self.mask_type == 'crop':
            img = img.crop(boxes)

        target = {}
        target["image_name"] = instance["image_name"]
        target["category_id"] = torch.tensor(instance["label"])
        target["boxes"] = torch.tensor(boxes)
        img = img.resize((224, 224))
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.instances)

    # collate_fn needs for batch
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))