import os
import pandas as pd
import matplotlib.pyplot as plt

exp_root_dir = "/data/mml/data_debugging_data"
dataset_name = "VOC2012"

error_record_df = pd.read_csv(os.path.join(exp_root_dir,"error_anno",dataset_name,"fault_records.csv"))
# 统计每个数值出现次数，按数值从小到大排
error_counts = error_record_df['fault_type'].value_counts().sort_values(ascending=False)

plt.figure()
plt.bar(error_counts.index, error_counts.values)
plt.xlabel('error_type')
plt.ylabel('count')
plt.title('Counts of error_type')
plt.tight_layout()
save_dir = os.path.join(exp_root_dir,"imgs")
save_path = os.path.join(save_dir,f"{dataset_name}_error_count.png") 
plt.savefig(save_path)
