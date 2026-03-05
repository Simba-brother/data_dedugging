
import yaml
import argparse
from utils.datasets import create_dataloader
from utils.general import colorstr
import torch
import torch.nn as nn
from models.yolo import Model
from base_data_manager import exp_data_root_dir, get_error_train_model_weight_file_path, get_nc_by_datasetname


def model_load_weight(model:nn.Module,device,weight_path:str):
    # 加载模型权重
    
    if weight_path.endswith("last.pt"):
        state_dict = torch.load(weight_path, map_location=device, weights_only=False)
        state_dict = state_dict['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    return model

def main():
    # 加载最后的模型
    model_weight_path = get_error_train_model_weight_file_path(dataset_name,model_name,epoch=49)
    # 加载模型结构
    nc = get_nc_by_datasetname(dataset_name)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc, anchors=3).to(device)
    # 加载设备
    device = torch.device("cuda:0")
    model = model_load_weight(model,device,model_weight_path)
    model.eval()

    # 加载error数据集
    # 读取data yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    # 数据加载器
    dataloader = create_dataloader(data["train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]
    imgs_num = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        imgs_num += img.shape[0]
    print(f"总共图像数量:{imgs_num}")


if __name__ == "__main__":
    exp_data_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "YOLOv7"
    main()