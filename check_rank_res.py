import joblib
from base_data_manager import get_ours_rank_res_path,get_datactive_rank_res_path



if __name__ == "__main__":
    dataset_name = "VisDrone" # VOC2012|KITTI|VisDrone
    model_name = "FRCNN" # YOLOv7|FRCNN
    # rank_res = joblib.load(get_ours_rank_res_path(dataset_name,model_name))
    rank_res = joblib.load(get_datactive_rank_res_path(dataset_name))
    print(f"rank中元素的数量:{len(rank_res)}")