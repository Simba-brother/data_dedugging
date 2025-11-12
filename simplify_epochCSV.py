'''
把epoch收集的imge的indicator csv中的图像路径简化一下
'''
import os
import pandas as pd
epoch_csv_root_dir = "/data/mml/data_debugging/collection_indicator/VOC2012/FRCNN"
Epochs = 50
for epoch in range(Epochs):
    csv_path = os.path.join(epoch_csv_root_dir, f"epoch_{epoch}.csv")
    df = pd.read_csv(csv_path)
    # 修改列名并提取文件名
    # df = df.rename(columns={'image_path': 'image_name'})
    # df['image_name'] = df['image_name'].apply(os.path.basename)
    df['epoch'] = [epoch for _ in range(df.shape[0])]
    df.insert(0,"sample_id",[i for i in range(df.shape[0])])
    df.to_csv(csv_path,index=False)

