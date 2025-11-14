'''
把epoch收集的imge的indicator csv中最后添加一列epoch,添加第一列sample_id
'''
import os
import pandas as pd
epoch_csv_root_dir = "/data/mml/data_debugging_data/collection_indicator/VOC2012/SSD"
Epochs = 50
for epoch in range(Epochs):
    csv_path = os.path.join(epoch_csv_root_dir, f"epoch_{epoch}.csv")
    df = pd.read_csv(csv_path)
    # 修改列名并提取文件名
    # df = df.rename(columns={'image_path': 'image_name'})
    # df['image_name'] = df['image_name'].apply(os.path.basename)
    df.insert(0,"sample_id",[i for i in range(df.shape[0])])
    df['epoch'] = [epoch for _ in range(df.shape[0])]
    df.to_csv(csv_path,index=False)
print("修改完毕")

