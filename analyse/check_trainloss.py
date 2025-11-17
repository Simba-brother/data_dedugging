import joblib
import os
import matplotlib.pyplot as plt


exp_data_root = "/data/mml/data_debugging_data/"
dataset_name = "VisDrone"
model_name = "FRCNN"

loss_data = joblib.load(os.path.join(exp_data_root,"check_train_effect",dataset_name,model_name,"epoch_trian_loss_value_list.joblib"))

epochs = range(1, len(loss_data) + 1)  # x 轴为 1~50 的 epoch

plt.figure()
plt.plot(epochs, loss_data, marker='o')  # 画折线图，点上加圆点标记
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.tight_layout()
save_dir = os.path.join(exp_data_root,"imgs",)
save_path = os.path.join(save_dir,f"{dataset_name}_{model_name}_loss.png")
plt.savefig(save_path)


