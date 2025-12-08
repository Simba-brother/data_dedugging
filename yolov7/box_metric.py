
import math
import os
import json
from PIL import Image
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import pandas as pd

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
    gt_json_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"gt_bboxs.json")
    with open(gt_json_path,"r") as file:
        gt_json = json.load(file)
    gt_box_match = defaultdict(list)
    # 遍历所有的图像
    count = 0
    for img_name in gt_json.keys():
        count += 1
        pretty_print(img_name,count)
        # 当前图像的g_boxs
        gt_bboxs = gt_json[img_name]
        image_path = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-yolo","train","images",img_name)
        image = Image.open(image_path)
        width, height = image.size
        # 当前图像的g_boxs的bbox格式进行转换
        for gt_box in gt_bboxs:
            gt_box["gt_bbox"] = xcycwh_to_x1y1x2y2(gt_box["gt_bbox"],width,height)
        # 遍历epoch
        for epoch in range(epochs):
            epoch_predicted_bboxs_json_path =  os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,f"epoch_{epoch}_predicted_bboxs.json")
            with open(epoch_predicted_bboxs_json_path,"r") as f:
                 epoch_predicted_bboxs_dict = json.load(f)
            # 该图像当前epoch下的p_boxs
            if img_name not in epoch_predicted_bboxs_dict:
                continue
            cur_epoch_p_boxs = epoch_predicted_bboxs_dict[img_name]["predicted_bboxs"]
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


def gt_box_metric_collection():
    start_time = time.time()  # 记录开始时间
    # gt_box_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"gt_bboxs.json")
    # with open(gt_box_path, 'r') as f:
    #     gt_boxs = json.load(f)

    gt_box_match_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"gt_box_match.json")
    with open(gt_box_match_path, 'r') as f:
        gt_box_match = json.load(f)
    collect = []
    count = 0
    for g_box_id in gt_box_match.keys():
        count += 1
        pretty_print(g_box_id,count,col_nums=50)
        matched_info_over_epoch = gt_box_match[g_box_id]
        row = {
            "g_box_id":int(g_box_id),
            "conf_list":[],
            "iou_list":[]
        }
        temp_dict = {}
        for matched_info in matched_info_over_epoch:
            epoch = matched_info["epoch"]
            temp_dict[epoch] = {
                "g_box":matched_info["g_box"],
                "p_box":matched_info["p_box"],
                "iou_val":matched_info["iou_val"]
            }
        
        for epoch in range(epochs):
            matched_info = temp_dict.get(epoch)
            if matched_info is None:
                conf = 0
                iou = 0
            else:
                conf = matched_info["p_box"]["conf"]
                iou = matched_info["iou_val"]
            row["conf_list"].append(conf)
            row["iou_list"].append(iou)
        collect.append(row)
    save_dir = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name)
    save_file_name = "collection.json"
    save_path = os.path.join(save_dir,save_file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(collect, f, indent=4)
    print(f"\ncollection is saved in {save_path}")
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    hours = int(elapsed_time // 3600)  # 计算小时数
    minutes = int((elapsed_time % 3600) // 60)  # 计算分钟数
    seconds = elapsed_time % 60  # 计算剩余的秒数

    print(f"运行时间：{hours:02d}:{minutes:02d}:{seconds:02.0f}")

def correct_vs_fault():
    gt_box_metric_collection_json_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name, "collection.json")
    with open(gt_box_metric_collection_json_path, "r", encoding="utf-8") as f:
        gt_box_metric_collection_list = json.load(f)
    print(f"gt_box数量:{len(gt_box_metric_collection_list)}")

    g_box_id_to_metric = {}
    for collection in gt_box_metric_collection_list:
        g_box_id = collection["g_box_id"]
        conf_list = collection["conf_list"]
        iou_list = collection["iou_list"]
        g_box_id_to_metric[g_box_id] = {
            "conf_list":conf_list,
            "iou_list":iou_list,
        }
    
    
    gt_bboxs_json_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"gt_bboxs.json")
    with open(gt_bboxs_json_path, "r", encoding="utf-8") as f:
        gt_bboxs_dict = json.load(f)
    
    g_box_id_to_info = {}
    for img_name in gt_bboxs_dict.keys():
        g_box_list = gt_bboxs_dict[img_name]
        for g_box in g_box_list:
            g_box_id = g_box["box_id"]
            g_box_id_to_info[g_box_id] = g_box

    # gt_box按照fault_type分组
    group_dict = defaultdict(list)
    for g_id in g_box_id_to_metric.keys():
        metric_dict = g_box_id_to_metric[g_id]
        conf_list = metric_dict["conf_list"]
        iou_list = metric_dict["iou_list"]
        box_info = g_box_id_to_info[g_id]
        fault_type = box_info["fault_type"]
        item = {
            "g_id":g_id,
            "img_name":box_info["img_name"],
            "cls":box_info["cls"],
            "bbox":box_info["gt_bbox"],
            "conf_list":conf_list,
            "iou_list":iou_list,
            "fault_type":fault_type
        }
        group_dict[fault_type].append(item)

    fault_to_metric_list = {}
    for fault_type in group_dict.keys():
        item_list = group_dict[fault_type]
        conf_list_list = []
        iou_list_list = []
        for item in item_list:
            conf_list_list.append(item["conf_list"])
            iou_list_list.append(item["iou_list"])
        conf_2darray = np.array(conf_list_list)
        iou_2darray = np.array(iou_list_list)
        conf_avg = np.mean(conf_2darray,axis = 0)
        iou_avg = np.mean(iou_2darray,axis = 0)
        fault_to_metric_list[fault_type] = {
            "conf_avg":conf_avg.tolist(),
            "iou_avg":iou_avg.tolist(),
        }
    # 准备 x 轴 epoch
    no_fault_list = fault_to_metric_list[0]["iou_avg"]
    cls_fault_list = fault_to_metric_list[1]["iou_avg"]
    loc_fault_list = fault_to_metric_list[2]["iou_avg"]
    redundancy_fault_list = fault_to_metric_list[3]["iou_avg"]

    epochs = range(1, 51)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, no_fault_list, label="no fault", marker='o', color = "green")
    plt.plot(epochs, cls_fault_list, label="cls fault", marker='o', color = "red")
    plt.plot(epochs, loc_fault_list, label="loc fault", marker='o', color = "blue")
    plt.plot(epochs, redundancy_fault_list, label="redundancy fault", marker='o', color = "black")

    plt.xlabel("Epoch")
    plt.ylabel(f"Mean IOU")
    plt.title(f"Mean IOU Over 50 Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_dir = os.path.join(exp_root_dir,"imgs","correct_vs_error_box","iou_avg")
    os.makedirs(save_dir,exist_ok=True)
    save_path = os.path.join(save_dir,f"{dataset_name}_{model_name}.png")
    plt.savefig(save_path)
    print("correct_vs_error is saved in",save_path)


def add_path_value(d, keys, value):
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur.setdefault(keys[-1], []).append(value)

def find(x,parent):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b, parent, rank):
    ra, rb = find(a,parent), find(b,parent)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1

def clusing(box_list,thre):
    N = len(box_list)
    parent = list(range(N))
    rank = [0]*N
    for i in range(N):
        for j in range(i+1,N):
            box_i = box_list[i]
            box_j = box_list[j]
            i_bbox = box_i["bbox"]
            j_bbox = box_j["bbox"]
            if calu_iou(i_bbox,j_bbox) > thre:
                union(i,j,parent,rank)

    clusters = defaultdict(list)
    for i in range(N):
        r = find(i,parent)
        clusters[r].append(i)
    cluster_list = list(clusters.values())
    return cluster_list


def get_epoch_to_matched_p_boxs(gt_match_dict):
    # 每个epoch中所有被匹配上的p_box
    epoch_to_match_info = {}
    # 遍历所有的g_box
    for g_box_id in gt_match_dict.keys():
        # 当前g_box的匹配信息
        match_info_list = gt_match_dict[g_box_id]
        for match_info in match_info_list:
            epoch = match_info["epoch"]
            p_box = match_info["p_box"]
            p_box_id = p_box["predicted_box_id"]
            if epoch in epoch_to_match_info:
                epoch_to_match_info[epoch][p_box_id] = p_box
            else:
                epoch_to_match_info[epoch] = {p_box_id:p_box}
    return epoch_to_match_info


def get_img_to_p_box_list(img_name_to_no_match_p):
    img_to_p_list = defaultdict(list)
    for img_name in img_name_to_no_match_p.keys():
        for epoch in img_name_to_no_match_p[img_name].keys():
            for p_box in img_name_to_no_match_p[img_name][epoch]:
                p_box["epoch"] = epoch
                img_to_p_list[img_name].append(p_box)
    return img_to_p_list

def get_img_to_clusters(img_to_p_box_list):
    img_to_clusters = defaultdict(list)
    for img_name in img_to_p_box_list.keys():
        p_box_list = img_to_p_box_list[img_name]
        cluster_list = clusing(p_box_list,thre=0.8)
        for cluster in cluster_list:
            cur_cluster_p_box_list = []
            for id in cluster:
                p_box = p_box_list[id]
                cur_cluster_p_box_list.append(p_box)
            img_to_clusters[img_name].append(cur_cluster_p_box_list)
    return img_to_clusters


def stability_pairwise_mean_iou(boxes):
    n = len(boxes)
    if n <= 1:
        return 1.0
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            i_bbox = boxes[i]["bbox"]
            j_bbox =  boxes[j]["bbox"]
            total += calu_iou(i_bbox,j_bbox)
            cnt += 1
    return total / max(1, cnt)

def conf_score(boxes):
    conf_sum = 0
    for p_box in boxes:
        conf_sum += p_box["conf"]
    return conf_sum / len(boxes)

def cls_consis_score(boxes):
    counter = defaultdict(int)
    
    for p_box in boxes:
        counter[p_box["predicted_cls"]] += 1
        
    max_count = -1
    max_cls = -1
    for cls,count in counter.items():
        if count > max_count:
            max_count = count
            max_cls = cls
    return max_count/len(boxes)

def epoch_freq(boxes):
    epoch_cover = set()
    for p_box in boxes:
        epoch_cover.add(p_box["epoch"])
    return len(epoch_cover) / 5



def total_score(freq, conf, stab, cls_cons,
                   w_conf=1.0, w_freq=2.0, w_stab=1.0, w_cls=1.0,
                   eps=1e-12):
    """
    加权几何平均，保证输出在[0,1]（前提：输入在[0,1]）
    S = (conf^w_conf * freq^w_freq * stab^w_stab * cls^w_cls)^(1/sum_w)
    """
    # clamp to [0,1]
    freq = max(0.0, min(1.0, float(freq)))
    conf = max(0.0, min(1.0, float(conf)))
    stab = max(0.0, min(1.0, float(stab)))
    cls_cons = max(0.0, min(1.0, float(cls_cons)))

    sum_w = w_conf + w_freq + w_stab + w_cls
    if sum_w <= 0:
        return 0.0

    # 用 log 防止下溢
    val = (w_conf * math.log(conf + eps) +
           w_freq * math.log(freq + eps) +
           w_stab * math.log(stab + eps) +
           w_cls  * math.log(cls_cons + eps))

    return math.exp(val / sum_w)

def caclu_cluster_score(cluster):
    
    conf = conf_score(cluster)
    stab = stability_pairwise_mean_iou(cluster)  
    cls_consis = cls_consis_score(cluster)
    e_freq = epoch_freq(cluster)
    s = total_score(e_freq,conf,stab,cls_consis)
    return s



def sort_cluster(img_to_clusters):
    cluster_list = []
    for img_name,clusters in img_to_clusters.items():
        for cluster in clusters:
            s = caclu_cluster_score(cluster)
            cluster_list.append({
                "cluster":cluster,
                "img_name":img_name,
                "score":s
            })
    sorted_cluster_list = sorted(cluster_list, key=lambda x: x['score'], reverse=True)
    return sorted_cluster_list

    
    


def mis_detect():
    # 读取所有gt_box的匹配信息
    gt_match_json_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,"gt_box_match.json")
    with open(gt_match_json_path,mode="r") as f:
        gt_match_dict = json.load(f)
    epoch_to_matched_p_boxs = get_epoch_to_matched_p_boxs(gt_match_dict)

    img_name_to_no_match_p = {}
    for epoch in range(epochs-5,epochs):
        predicted_epoch_json_path = os.path.join(exp_root_dir,"collection_indicator_bbox_level",dataset_name,model_name,f"epoch_{epoch}_predicted_bboxs.json")
        with open(predicted_epoch_json_path,mode="r") as f:
            predicted_epoch_dict = json.load(f)
        for img_name in predicted_epoch_dict.keys():
            # img_name在该epoch下的所有预测框
            p_box_list = predicted_epoch_dict[img_name]["predicted_bboxs"]
            for p_box in p_box_list:
                p_id = p_box["predicted_box_id"]
                if p_id not in epoch_to_matched_p_boxs[epoch] and p_box["conf"] > 0.6:
                    add_path_value(img_name_to_no_match_p,keys=[img_name,epoch],value=p_box)
    img_to_p_box_list  = get_img_to_p_box_list(img_name_to_no_match_p)
    img_to_clusters = get_img_to_clusters(img_to_p_box_list)

    sorted_clusters = sort_cluster(img_to_clusters)
    img_name_to_score = defaultdict(float)

    for cluster in sorted_clusters:
        img_name = cluster['img_name']
        score = cluster["score"]
        if score > img_name_to_score[img_name]:
            img_name_to_score[img_name] = score

    sorted_img_name_to_score = sorted(img_name_to_score.items(), key=lambda item: item[1], reverse=True)
    img_name_list = []
    for img_name,score in sorted_img_name_to_score:
        if score > 0.2:
            img_name_list.append(img_name)

    fault_df = pd.read_csv(os.path.join(exp_root_dir,"error_anno",dataset_name,"fault_records.csv")) 
    mis_df = fault_df[fault_df['fault_type'] == 4]
    mis_img_name_set = set(mis_df["img_name"])

    fn = len(mis_img_name_set - set(img_name_list))
    tp = len(mis_img_name_set & set(img_name_list))
    fp = len(set(img_name_list) - mis_img_name_set)

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2*precision*recall / (precision+recall)

    print()


    





if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VOC2012"
    model_name = "YOLOv7"
    epochs = 50
    # match()
    # gt_box_metric_collection()
    # correct_vs_fault()
    mis_detect()

    
    
