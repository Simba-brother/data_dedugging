import joblib
import pandas as pd

def compute_apfd(list_A, list_B):
    """
    list_A: set/list, 真实错误图像路径(gt)
    list_B: list, 按可疑度排序的图像路径
    """
    n = len(list_B)
    list_A_set = set(list_A)
    TF_positions = []

    # 遍历 list_B 找到真实错误的位置
    for idx, img in enumerate(list_B, start=1):  # 从1开始计数
        if img in list_A_set:
            TF_positions.append(idx)

    m = len(list_A)
    if m == 0:
        return 0.0  # 防止除零

    apfd = 1 - sum(TF_positions) / (n * m) + 1 / (2 * n)
    return apfd

def main():
    pass
if __name__ == "__main__":
    ranked_img_name_list = joblib.load("/data/mml/data_debugging_data/DataDetective/ranked_img_name_list.joblib")
    error_record_df = pd.read_csv("/data/mml/data_debugging_data/datasets/VOC2012_error_record/error_record_simple.csv")
    gt_error_list = list(error_record_df["img_file_name"])
    apfd = compute_apfd(gt_error_list,ranked_img_name_list)
    print(apfd)
    # main()