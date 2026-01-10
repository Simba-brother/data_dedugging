import os
import json

def get_all_files(directory)->list[str]:
    files = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    return files

def get_nc(dataset_name)->int:
    if dataset_name == "VOC2012":
        nc = 20
    elif dataset_name == "KITTI_8":
        nc = 8
    elif dataset_name == "KITTI":
        nc = 9
    elif dataset_name == "VisDrone":
        nc = 10
    else:
        raise Exception("数据集参数错误")
    return nc

def read_json(json_path:str):
    _json = None
    with open(json_path, "r") as f:
        _json = json.load(f)
    return _json