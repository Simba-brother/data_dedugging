
import os
import json
from PIL import Image
import numpy as np
from collections import defaultdict
import time

def calu_iou(gt_bbox,predicted_bbox):
    x1_min, y1_min, x1_max, y1_max = gt_bbox
    x2_min, y2_min, x2_max, y2_max = predicted_bbox

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def get_iou_matrix(gt_box_list, p_box_list):
    P = len(p_box_list)
    G = len(gt_box_list)
    iou_matrix = np.zeros((P,G))
    for i,p_box in enumerate(p_box_list):
        for j,g_box in enumerate(gt_box_list):
            p_bbox = p_box["bbox"]
            g_bbox = g_box["gt_bbox"]
            iou = calu_iou(g_bbox,p_bbox)
            iou_matrix[i][j] = iou
    return iou_matrix


def search_match(gt_box_list, predicted_box_list, iou_thre=0.5):
    '''
    一张图像的gt boxs与predicted boxs匹配函数
    '''
    # 将预测框按照conf从大到小进行排序
    predicted_box_list.sort(key=lambda x: x["conf"], reverse=True)
    # GT box数量
    G = len(gt_box_list)
    # predicted box数量
    P = len(predicted_box_list)
    # 记录已经匹配成功的GT box
    used_gt = set()
    # 匹配结果容器
    matches = []
    # 所有的cls
    cls_set = set([gt_box["cls"] for gt_box in gt_box_list])
    # 分类操作，遍历cls_set
    for cls in cls_set:
        # 当前类别的gt boxs
        cur_cls_gt_box_list = [box for box in gt_box_list if box["cls"] == cls]
        # 当前类别的p boxs，（已经按照conf从大到小）
        cur_cls_p_box_list = [box for box in predicted_box_list if box["predicted_cls"] == cls]
        if len(cur_cls_gt_box_list) == 0 or len(cur_cls_p_box_list) == 0:
            continue
        # 获得p_boxs与g_boxs的iou矩阵，shape:(len(p_boxs),len(g_boxs))
        iou_matrix = get_iou_matrix(cur_cls_gt_box_list,cur_cls_p_box_list)
        # 每个p_box(行)匹配最好的g_box
        best_gt_box_id_list = iou_matrix.argmax(axis=1)
        # 每个p_box(行)匹配最好的g_box对应的iou值
        best_iou_list = iou_matrix.max(axis=1)
        for i,iou_val in enumerate(best_iou_list):
            iou_val = iou_val.item()
            if iou_val < iou_thre:
                # 说明这个p_box与所有的g_box的iou都没达到阈值以上
                continue
            # 这个p_box匹配上了一个g_box
            best_gt_id = best_gt_box_id_list[i]
            # 这个g_box被p_box[i]匹配上了
            matched_gt_box = cur_cls_gt_box_list[best_gt_id]
            if matched_gt_box["box_id"] in used_gt:
                # p_box看中的g_box已经被conf 更大的p_box占有了，就不管你（当前p_box[i]）了
                continue
            used_gt.add(matched_gt_box["box_id"])
            p_box = cur_cls_p_box_list[i]
            matches.append((matched_gt_box,p_box,iou_val))
    return matches



def xcycwh_to_x1y1x2y2(bbox,W,H):
    xc = bbox[0]
    yc = bbox[1]
    w = bbox[2]
    h = bbox[3]

    # 1. 归一化 -> 像素
    x_c = xc * W
    y_c = yc * H
    bw  = w  * W
    bh  = h  * H

    # 2. 中心 -> 左上 / 右下
    x1 = x_c - bw / 2
    y1 = y_c - bh / 2
    x2 = x_c + bw / 2
    y2 = y_c + bh / 2

    # 3. 转 int + 裁剪
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))

    return [x1,y1,x2,y2]

def pretty_print(content,count,col_nums=10):
    print(content, end=' ')
    if count % col_nums == 0:  # 如果计数器是10的倍数
        print()  # 打印换行符

def match():
    start_time = time.time()  # 记录开始时间
    all_gt_predicted_json_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"all_gt_predicted.json")
    with open(all_gt_predicted_json_path,"r") as file:
        all_gt_predicted_dict = json.load(file)
    gt_box_match = defaultdict(list)
    # 遍历所有的图像
    count = 0
    for img_name in all_gt_predicted_dict.keys():
        pretty_print(img_name,count)
        count += 1
        # 当前图像的size
        width,height = all_gt_predicted_dict[img_name]["size"]
        # 当前图像的g_boxs
        gt_bboxs = all_gt_predicted_dict[img_name]["gt_bboxs"]
        # 当前图像的g_boxs的bbox格式进行转换
        for gt_box in gt_bboxs:
            gt_box["gt_bbox"] = xcycwh_to_x1y1x2y2(gt_box["gt_bbox"],width,height)
        # 当前图像的所有epochs下的p_boxs
        predicted_bboxs_over_epoch = all_gt_predicted_dict[img_name]["predicted_bboxs_over_epoch"]
        # 遍历epoch
        for epoch in range(epochs):
            # 该图像当前epoch下的p_boxs
            cur_epoch_p_boxs = predicted_bboxs_over_epoch[epoch]
            if cur_epoch_p_boxs == None:
                continue
            # 获得当前图像g_boxs与p_boxs的匹配关系
            matches = search_match(gt_bboxs,cur_epoch_p_boxs)
            for match in matches:
                matched_g_box = match[0]
                p_box = match[1]
                iou_val = match[2]
                g_box_id = matched_g_box["box_id"]
                gt_box_match[g_box_id].append({"epoch":epoch, "g_box":matched_g_box, "p_box":p_box,"iou_val":iou_val})

    save_dir = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name)
    save_file_name = "gt_box_match.json"
    save_path = os.path.join(save_dir,save_file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(gt_box_match, f, indent=4)
    print(f"\ngt_box_match is saved in {save_path}")
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数

    print(f"运行时间：{hours:02d}:{minutes:02d}:{seconds:02.0f}")


def main():
    gt_box_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"gt_bboxs.json")

    gt_box_match_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"gt_box_match.json")
    with open(gt_box_match_path, 'r') as f:
        gt_box_match = json.load(f)
    
    for g_box_id in gt_box_match.keys():
        matched_info_over_epoch = gt_box_match[g_box_id]

        temp_dict = {}
        for matched_info in matched_info_over_epoch:
            epoch = matched_info["epoch"]
            temp_dict[epoch] = {
                "g_box":matched_info["g_box"],
                "p_box":matched_info["p_box"],
                "iou_val":matched_info["iou_val"]
            }
        for epoch in range(epochs):
            match_info = temp_dict.get(epoch)
            if matched_info is None:
                pass
            else:
                conf = matched_info["p_box"]["conf"]
                iou = matched_info["iou_val"]

            
                
        

    print()
    
    



if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "YOLOv7"
    epochs = 50
    # match()
    main()
    
    
