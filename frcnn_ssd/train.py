
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from datasets import CocoDetectionDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from torchvision.models.detection.ssd import SSDClassificationHead
import torch,torchvision
from engine import train_one_epoch, evaluate
from torchvision import models, transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from base_data_manager import (get_correct_ann_file_path,get_error_ann_file_path, get_imgs_dir, 
                               get_error_train_model_weight_file_path, get_repair_ann_file_path)
import time
# conf_threshold = 0.8
# Transform PIL image --> PyTorch tensor
def get_transform():
    return ToTensor()

def get_train_and_val_dataset():
    # Load training dataset
    train_dataset = CocoDetectionDataset(
        image_dir=train_imgs_dir,
        annotation_path=train_annotation_path,
        transforms=get_transform()
    )

    # Load validation dataset
    val_dataset = CocoDetectionDataset(
        image_dir=val_imgs_dir,
        annotation_path=val_annotation_path,
        transforms=get_transform()
    )
    return train_dataset, val_dataset

def test():
    # class names
    label_list= ["","ball", "goalkeeper", "player", "referee",""]
    
    # Number of classes (include background)
    num_classes = 6   # this has to be 5 in normally, but because of some labeling issues in dataset this is 6.
    
    # Load the same model 
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    
    # Load trained Faster R-CNN model
    model.load_state_dict(torch.load("check_points/model_epoch_10.pth"))
    model.eval()
    
    # Load image with OpenCV and convert to RGB
    img_path = "football-players-detection/valid/2e57b9_1_6_png.rf.74724a3814311da25a648a48d778d589.jpg"
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_pil).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # detection data
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    """
    Higher threshold give you more accurate detections, 
    but number of predictions is reduced; there is a simple trade-off
    """
    conf_threshold = 0.8
    for i in range(len(boxes)):
        if scores[i] > conf_threshold:
            box = boxes[i].cpu().numpy().astype(int)
            label = label_list[labels[i]]
            score = scores[i].item()
            text = f"{label}: {score:.2f}"
            cv2.putText(image_bgr, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
            # Draw bbox and label
            cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    
    
    # Convert BGR --> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Show image with larger figure size
    plt.figure(figsize=(16, 12)) 
    plt.imshow(image_rgb)
    plt.axis('off')
    # plt.show()
    plt.savefig("test.png")

def build_model(model_name,nc):
    if model_name == "FRCNN":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # model =torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # Number of input features for the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        """  
        Number of classes must be equal to your label number
        """
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc)
    elif model_name == "SSD":
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        '''
        model.head.classification_head = SSDClassificationHead(
            [512, 1024, 512, 256, 256, 256],
            model.anchor_generator.num_anchors_per_location(), 
            nc
        )
        '''

        c_in_features=[model.head.classification_head.module_list[i].in_channels for i in range(len(model.head.classification_head.module_list))]
        num_anchors=model.anchor_generator.num_anchors_per_location()
        # # 用新的头部替换预先训练好的头部
        model.head.classification_head=SSDClassificationHead(in_channels=c_in_features,num_anchors=num_anchors,num_classes=nc)

    else:
        raise Exception("模型名称传入错误")
    return model


def custom_train_one_epoch(model,train_loader,epoch,NUM_EPOCHS,DEVICE,optimizer):
    model.train()
    epoch_loss = 0
    total_batches = len(train_loader)

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} -----------------------------")
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        print(f"  Batch {batch_idx+1}/{total_batches} | Loss: {losses.item():.4f}")

def save_epoch(model,epoch):
    save_file_name = f"epoch_{epoch}.pth"
    save_path = os.path.join(epoch_save_dir,save_file_name)
    torch.save(model.state_dict(), save_path)
    return save_path

def model_load_weight(model,model_weight_path):
    # 加载模型
    state_dict = torch.load(model_weight_path,map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def train(model_weight_path=None):
    start_time = time.time()  # 记录开始时间
    # 加载FRCNN模型（预训练）
    # Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN, , you change this 
    # weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    # weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT (up-to-date weights)
    # model =torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Number of classes in the dataset (including background)
    # +1 for bg class

    # 加载数据集
    train_dataset, val_dataset = get_train_and_val_dataset()
    # Load dataset with DataLoaders, you can change batch_size 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    # train_t_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # 构建模型
    num_classes = len(train_dataset.coco.getCatIds()) + 1
    model = build_model(model_name,num_classes)

    # 设备
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    if model_weight_path is not None:
        model_load_weight(model,model_weight_path)
    # 训练参数
    '''
    # 锁住Backbone params
    for param in model.backbone.parameters():
        param.requires_grad = False
    '''
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 参数优化器

    # optimizer = torch.optim.Adam(params, lr=1e-4)
    optimizer = torch.optim.SGD(params,lr=init_lr,momentum=0.9,weight_decay=0.0005)

    # lr 调度器
    '''
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    '''
    # train loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs}")
        # Train the model for one epoch, printing status every 25 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)  # Using train_loader for training
        # lr_scheduler.step()
        # custom_train_one_epoch(model,train_loader,epoch,num_epochs,device,optimizer)
        # Evaluate the model only on the validation dataset, not training
        evaluate(model, val_loader, device=device)  # Using val_loader for evaluation
        epoch_save_path = save_epoch(model,epoch)
        print(f"model pth is saved in: {epoch_save_path}")
        '''
        if model_name == "FRCNN":
            collection_FRCNN_indicator(model,device,train_t_loader,epoch)
        elif model_name == "SSD":
            collection_SSD_indicator(model,device,train_t_loader,epoch)
        '''
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数
    print(f"运行时间：{hours:02d}:{minutes:02d}:{seconds:02.0f}")

def collection_SSD_indicator(model,device,dataloader,epoch):
    item_list = []
    for images, targets in dataloader:
        item = {
            "image_name":None,
            "loss_box":None,
            "loss_objcls":None,
            "loss":None,
            "conf_avg":None,
            "box_count_dif":None
        }
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        image_name = targets[0]["image_path"].split("/")[-1]
        gt_box_count = targets[0]["boxes"].shape[0]
        item["image_name"] = image_name
        model.train()
        loss_dict = model(images, targets)
        
        item["loss_box"] = loss_dict["bbox_regression"].item()
        item["loss_objcls"] = loss_dict["classification"].item()
        item["loss"] = item["loss_objcls"] + item["loss_box"]
        # Inference
        model.eval()
        predictions = model(images)
        # detection data
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        
        """
        Higher threshold give you more accurate detections, 
        but number of predictions is reduced; there is a simple trade-off
        """
        conf_threshold = 0.8
        conf_list = []
        for i in range(len(boxes)):
            if scores[i] > conf_threshold:
                # box = boxes[i].cpu().numpy().astype(int)
                # label = label_list[labels[i]]
                conf = scores[i].item()
                conf_list.append(conf)
        # box_count_dif = len(conf) - 
        if len(conf_list) == 0:
            conf_avg = 0
        else:
            conf_avg = round(sum(conf_list) / len(conf_list),3)
        item["conf_avg"] = conf_avg
        box_count_dif = abs(len(conf_list)-gt_box_count)
        item["box_count_dif"] = box_count_dif
        item_list.append(item)
    df = pd.DataFrame(item_list)
    save_dir = f"{exp_data_root_dir}/collection_indicator/{dataset_name}/{model_name}"
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,f"epoch_{epoch}.csv")
    df.to_csv(save_path, index=False)
    print(f"保存在：{save_path}")


def collection_FRCNN_indicator(model,device,dataloader,epoch):
    item_list = []
    for images, targets in dataloader:
        item = {
            "image_name":None,
            "loss_box":None,
            "loss_obj":None,
            "loss_cls":None,
            "loss":None,
            "conf_avg":None,
            "box_count_dif":None
        }
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        image_name = targets[0]["image_path"].split("/")[-1]
        gt_box_count = targets[0]["boxes"].shape[0]
        item["image_name"] = image_name

        model.train()
        loss_dict = model(images, targets)
        loss_classifier = loss_dict["loss_classifier"].item()
        item["loss_cls"] = loss_classifier
        loss_box_reg = loss_dict["loss_box_reg"].item()
        item["loss_box"] = loss_box_reg
        loss_objectness = loss_dict["loss_objectness"].item()
        item["loss_obj"] = loss_objectness
        loss = loss_classifier + loss_box_reg + loss_objectness
        item["loss"] = loss
        # Inference
        model.eval()
        predictions = model(images)
        # detection data
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        
        """
        Higher threshold give you more accurate detections, 
        but number of predictions is reduced; there is a simple trade-off
        """
        conf_threshold = 0.8
        conf_list = []
        for i in range(len(boxes)):
            if scores[i] > conf_threshold:
                # box = boxes[i].cpu().numpy().astype(int)
                # label = label_list[labels[i]]
                conf = scores[i].item()
                conf_list.append(conf)
        
        if len(conf_list) == 0:
            conf_avg = 0
        else:
            conf_avg = round(sum(conf_list) / len(conf_list),3)
        item["conf_avg"] = conf_avg
        box_count_dif = abs(len(conf_list)-gt_box_count)
        item["box_count_dif"] = box_count_dif
        item_list.append(item)
    df = pd.DataFrame(item_list)
    save_dir = f"{exp_data_root_dir}/collection_indicator/{dataset_name}/{model_name}"
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,f"epoch_{epoch}.csv")
    df.to_csv(save_path, index=False)
    print(f"保存在：{save_path}")

if __name__ == "__main__":

    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
    model_name = "FRCNN" # FRCNN|SSD
    trainset_status = "repair_datactive" # clean|error|repair_datactive|repair_ours

    if trainset_status in ["repair_ours", "repair_datactive"]:
        model_weight_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch=49)
        num_epochs = 25
    else:
        model_weight_path = None
        num_epochs = 50
    gpu_id = 0
    init_lr = 5e-3  # SSD:1e-3
    epoch_save_dir = os.path.join(exp_data_root_dir,"models",dataset_name.lower(),
                                  model_name.lower(), trainset_status)
    os.makedirs(epoch_save_dir,exist_ok=True)
    train_imgs_dir = get_imgs_dir(dataset_name,"train",style="coco")
    val_imgs_dir = get_imgs_dir(dataset_name,"val",style="coco")

    if trainset_status == "clean":
        train_annotation_path = get_correct_ann_file_path(dataset_name,"train")
    elif trainset_status == "error":
        train_annotation_path = get_error_ann_file_path(dataset_name)
    elif trainset_status == "repair_ours":
        train_annotation_path = get_repair_ann_file_path(dataset_name,"ours",model_name)
    elif trainset_status == "repair_datactive":
        train_annotation_path = get_repair_ann_file_path(dataset_name,"datactive",model_name)

    val_annotation_path = get_correct_ann_file_path(dataset_name,"val")

    train(model_weight_path)
    # test()