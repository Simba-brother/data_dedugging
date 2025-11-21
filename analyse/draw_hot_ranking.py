import os
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
import joblib
import pandas as pd

def draw_rank(isError_list,save_path):
    # 话图看一下中毒样本在序中的分布
    distribution = [1 if flag else 0 for flag in isError_list]
    # 绘制热力图
    # 创建图形时设置较小的高度
    plt.style.use(['science','ieee'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6
    })
    plt.figure(figsize=(3, 0.5))  # 宽度为10，高度为2（可根据需要调整）
    plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    # plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('ranking',fontsize='3')
    # 调整横轴刻度字号
    plt.xticks(fontsize=3)  # 明确设置横轴刻度字号为6pt
    # plt.colorbar()
    plt.yticks([])
    plt.savefig(save_path, bbox_inches='tight', dpi=800) # pad_inches=0.0
    plt.close()

if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "YOLOv7"
    ranked_img_name_list = joblib.load(os.path.join(exp_data_root,"Ours",dataset_name,model_name,"ranked_img_name_list.joblib"))
    fault_record_df = pd.read_csv(os.path.join(exp_data_root,"error_anno",dataset_name,"fault_records.csv"))
    fault_img_name_set = set(fault_record_df["img_name"].tolist())
    isError_list = []
    for img_name in ranked_img_name_list:
        if img_name in fault_img_name_set:
            isError_list.append(True)
        else:
            isError_list.append(False)
    save_dir = os.path.join(exp_data_root,"imgs","hot_ranking")
    save_file_name = f"{dataset_name}_{model_name}.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_rank(isError_list,save_path)
    print("save_path:",save_path)