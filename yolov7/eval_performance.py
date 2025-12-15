'''看一下我们训练的模型在val上的性能'''

import os
import yaml
from pathlib import Path
import argparse
import torch
from utils.datasets import create_dataloader
from utils.general import colorstr,non_max_suppression,scale_coords,xyxy2xywh,xywh2xyxy,box_iou
from models.yolo import Model
from utils.torch_utils import select_device,time_synchronized
from tqdm import tqdm
import numpy as np
from utils.metrics import ap_per_class,ConfusionMatrix


def main():
    # 拿到数据yaml文件
    data = f"data/{dataset_name}.yaml"
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.single_cls = False
    # 数据加载器
    dataloader = create_dataloader(data["train"], 640, 32, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'train: '))[0]

    imgsz = 640
    batch_size = 32
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    epoch = 49
    # 加载模型权重
    weights_path = os.path.join(exp_data_root,"models",f"{dataset_name.lower()}_error","yolov7",f"epoch_{epoch}.pt")
    state_dict = torch.load(weights_path, map_location=device)  # load checkpoint
    # 注入权重
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    nc = int(data['nc'])  # number of classes
    confusion_matrix = ConfusionMatrix(nc=nc)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    # 批次遍历数据集
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, wid
        with torch.no_grad():
            # out:shape:(bs,anchors*grids,nc+5)
            t = time_synchronized()
            out, train_out = model(img, augment=False)  # inference and training outputs
            t0 += time_synchronized() - t
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            # [xyxy, conf, cls]
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=0.001, iou_thres=0.65, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # 在每张图像上做统计
        for si, pred in enumerate(out):
            seen += 1
            # si: 这批图像的局部索引
            # 拿到这张图像的所有标签: [[cls,xc,yc,width,height],...]
            labels = targets[targets[:, 0] == si, 1:]
            # 这张图像的标签数量
            nl = len(labels)
            path = Path(paths[si])
            # 这张图像的target class list
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # pred: shape: (n个预测框,xyxy+conf+cls)
            if len(pred) == 0:
                # 经过NMS后没有预测结果
                if nl:
                    # 本身具有label
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            # 拷贝一份这张图像的预测
            predn = pred.clone()
            # img[si].shape[1:]: 这张图像的chanel,width,height
            # shapes[si][0], shapes[si][1] 图像native原图size
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            '''
            # 这张图像的预测结果写入text
            for *xyxy, conf, cls in predn.tolist():
                line = (cls, *xyxy, conf)
                save_dir = os.path.join(exp_data_root,"eval_model_performance",dataset_name,model_name,"predicted_labels")
                save_file_name = path.stem + '.txt'
                save_file_path = os.path.join(save_dir,save_file_name)
                with open(save_file_path, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            '''
            img_name = path.name
            for *xyxy, conf, cls in predn.tolist():
                jdict.append({
                    "img_name":img_name,
                    "cls":int(p[5]),
                    "bbox":xyxy,
                    "conf":round(p[4], 5)

                })


            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                # 如果这张图像有预测输出
                detected = []  # target indices
                # 包含的所有cls
                tcls_tensor = labels[:, 0]
                # 转换target box的xywh => xyxy
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # 遍历unique cls
                for cls in torch.unique(tcls_tensor):
                    # 找gt_cls等于到当前cls的行索引list
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    # 找p_cls等于到当前cls的行索引list
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # Search for detections
                    if pi.shape[0]:
                        # 如果当前cls下有pi
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            # pi 中 的第j个位置的iou > 0.5
                            # i 中第j个位置存储了ti的位置
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                # pi[j]: 预测框p
                                # ious[j]: 这个预测框匹配到的最大iou
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # correct: 各个预测框在不同iouv下是否正确匹配了target
            # pred[:,4]: 各个预测框的坐标
            # pred[:,5]: 各个预测框的conf
            # tcls: 一个图像中所有目标的cls
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    # Compute statistics
    # zip(*stats): *stats从stats列表中解包出多个tuple,zip(*stats)打包成所有图像的correct,..
    # *stats 把列表解包成多个 tuple 传给 zip。
    # zip(*stats) 会把“按图片存的 tuple 列表”转为“按字段分组”的迭代器：
    # 第 1 组：所有图片的 correct
    # 第 2 组：所有图片的 pred[:, 4]
    # 第 3 组：所有图片的 pred[:, 5]
    # 第 4 组：所有图片的 tcls
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any(): # .any() 表示里面至少有一个元素为 True / 非零。
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=False, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    # 打印每张图像的推理和NMS速度(ms)
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)


if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "KITTI" # VOC2012, KITTI
    model_name = "YOLOv7"
    # 脚本设备
    device = select_device('0')
    # create model 结构
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=20, anchors=3).to(device)
    main()

