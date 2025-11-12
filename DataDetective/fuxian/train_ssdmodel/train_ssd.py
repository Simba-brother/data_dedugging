import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from UTILS import transforms as T
from fuxian.train_ssdmodel.datasets_builder import COCO_type_Dataset
from UTILS.engine import evaluate, train_one_epoch
from fuxian.train_ssdmodel.eval_utils import eval

def build_dataset(imgs_root, annotation_json_path, train:bool):
    train_transforms = T.Compose(
        [
            T.RandomPhotometricDistort(),
            T.RandomZoomOut(fill=list((123.0, 117.0, 104.0))),
            T.RandomIoUCrop(),
            T.RandomHorizontalFlip(p=0.5),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ]
    )
    val_transforms = T.Compose(
        [
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ]
    )
    if train:
        return COCO_type_Dataset(imgs_root, annotation_json_path, train_transforms)
    else:
        return COCO_type_Dataset(imgs_root, annotation_json_path, val_transforms)


def main():
    train_dataset = build_dataset(train_root,train_annotation_json_path, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=train_dataset.collate_fn)
    val_dataset = build_dataset(val_root,val_annotation_json_path, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4,
                              collate_fn=val_dataset.collate_fn)
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.head.classification_head = SSDClassificationHead(
        [512, 1024, 512, 256, 256, 256],
        model.anchor_generator.num_anchors_per_location(), 
        num_classes
    )
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14, 22], gamma=0.1)

    device = torch.device('cuda:0')
    model.to(device)

    for epoch in range(epochs):
        metric_log, loss, classloss, boxloss = train_one_epoch(model, optimizer, train_dataloader, device, 
                                                               epoch, 
                                                               print_freq=10,
                                                               modeltype="ssd")
        with torch.no_grad():
            # coco_evaluator = evaluate(model, val_dataloader, device=device)
            eval(model, val_dataloader, device=device, save_dir="checkpoints/SSD_football")
        # map = coco_evaluator.coco_eval["bbox"].stats[0]
        # checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "loss": loss
        } 
        checkpoint_save_path = os.path.join(model_save_dir,f"epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_save_path)

if __name__ == "__main__":
    epochs = 30
    batch_size = 16
    model_save_dir = "checkpoints/SSD_football"
    train_root = "/data/mml/data_debugging/datasets/football-player/train"
    train_annotation_json_path = "/data/mml/data_debugging/datasets/football-player/train/_annotations.coco.json"
    val_root = "/data/mml/data_debugging/datasets/football-player/valid"
    val_annotation_json_path = "/data/mml/data_debugging/datasets/football-player/valid/_annotations.coco.json"
    num_classes = 5 # VOC2012
    main()

