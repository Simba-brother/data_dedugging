import os
import json
from datetime import datetime
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

def save_json_file(data, file_path):
    """
    保存JSON数据到文件
    
    Args:
        data (dict): 要保存的JSON数据
        file_path (str): 目标文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_formatted_time():
    """返回当前时间的格式化字符串（YYYY-MM-DD_HH:MM:SS）"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")

def is_directory_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def add_path_value(d:dict, keys:list, value):
    '''
    多层级字典，最后指向[]
    '''
    cur = d
    # 遍历所有层级的key
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur.setdefault(keys[-1], []).append(value)
