import os
import json
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from fuxian.inference_disassemble_dataset import Inference_classificationDataSet

def build_dataset(mask_type):
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )
    dataset = Inference_classificationDataSet(
        img_root, 
        annotation_path,
        mask_type = mask_type,
        transforms=data_transform)
    return dataset

def build_model():
    # load model
    modelState = torch.load(trained_model_path, map_location="cpu")
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048, class_num)
    model.load_state_dict(modelState["model"])
    return model

def infer():
    dataset = build_dataset(mask_type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    model = build_model()
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    # 所有解构出的instance_img的推理结果
    results = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, targets = data
            outputs = model(images.to(device))
            # softmax outputs
            labels = targets['category_id'].to(device)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            # print(label,labels)
            loss = loss_func(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1) # predictd其实是预测的label 
            # progress bar
            print("\rInference: {}/{}".format(i + 1, len(dataloader)), end="")
            # save softmax outputs image_name category_id boxes
            for j in range(len(predicted)):
                content_dic = {
                    "image_name": targets["image_name"][j],
                    "full_scores": outputs[j].cpu().numpy().tolist(), # imgs[j]的prob_list
                    "pred_category_id":predicted[j],
                    "gt_category_id": int(targets["category_id"][j]),
                    "bbox": targets["boxes"][j].numpy().tolist(),
                    "loss": loss # imgs[j]的loss,其实就是一个instance的loss，因为由于解构了数据集所以一个instance就是一个img
                }
                results.append(content_dic)
    json_str = json.dumps(results, indent=4)
    with open(results_save_path,'w') as json_file:
        json_file.write(json_str)

if __name__ == "__main__":
    
    img_root = "/data/mml/data_debugging/datasets/VOC2012-coco/train"
    annotation_path = "/data/mml/data_debugging/datasets/VOC2012-coco/train/_annotations.coco_error.json"
    class_num = 21
    mask_type = "crop" # crop and other objects
    trained_model_path = f"/data/mml/data_debugging/DataDetective/saved_models/{mask_type}/epoch_13.pth"
    results_save_path = f"/data/mml/data_debugging/DataDetective/infer_results/{mask_type}.json"
    infer()