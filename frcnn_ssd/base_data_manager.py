'''
中间实验数据管理
'''
import os

exp_data_root_dir = "/data/mml/data_debugging_data/"

def get_correct_ann_file_path(dataset_name,train_or_val):
    ann_file_path = ""
    if train_or_val == "val":
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val,"_annotations.coco.json")
    else:
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val,"_annotations.coco_correct.json")
        
    return ann_file_path

def get_error_ann_file_path(dataset_name):
    ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train","_annotations.coco_error.json")
    return ann_file_path

def get_repair_ann_file_path(dataset_name,
                             method_name,
                             model_name:None):
    ann_file_path = ""    
    if method_name == "ours":
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",f"_annotations.coco_repair_ours_{model_name}.json")
    if method_name == "datactive":
        ann_file_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train",f"_annotations.coco_repair_datactive.json")
    return ann_file_path

def get_error_train_model_weight_file_path(dataset_name,model_name,epoch):
    model_weight_file_path = ""
    if model_name == "YOLOv7":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",f"{dataset_name.lower()}", model_name.lower(), "error", f"epoch_{epoch}.pt")
    elif model_name == "FRCNN":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",f"{dataset_name.lower()}", model_name.lower(), "error", f"epoch_{epoch}.pth")
    return model_weight_file_path

def get_repair_train_model_weight_file_path(dataset_name,model_name, method_name):
    model_weight_file_path = ""
    if model_name == "YOLOv7":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(), f"repair_{method_name}", "last.pt")
    elif model_name == "FRCNN":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(), f"repair_{method_name}", "epoch_49.pth")
    return model_weight_file_path

def get_clean_train_model_weight_file_path(dataset_name,model_name):
    model_weight_file_path = ""
    if model_name == "YOLOv7":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(),"clean","weights","last.pt")
    elif model_name == "FRCNN":
        model_weight_file_path = os.path.join(exp_data_root_dir,"models",dataset_name.lower(), model_name.lower(),"clean", "epoch_49.pth")
    return model_weight_file_path

def get_imgs_dir(dataset_name,train_or_val,style):
    imgs_dir = ""
    if style == "coco":
        imgs_dir = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco",train_or_val)
    elif style == "yolo":
        imgs_dir = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-yolo",train_or_val,"images")
    return imgs_dir
    




if __name__ == "__main__":
    pass



