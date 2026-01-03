
import os
import joblib
import json
from collections import defaultdict
from ours.base_data_manager import (get_ours_rank_res_path,get_collected_gt_box_json_path,
                                    get_error_ann_file_path,get_correct_ann_file_path)

def get_gid_to_img_and_line():
    res = {}
    with open(gt_json_path,mode='r') as f:
        gt_box = json.load(f)
    for img_name,g_box_list in gt_box.items():
        line_no = 0
        for g_box in g_box_list:
            gid = g_box["box_id"]
            res[gid] = {
                "img_name":img_name,
                "line_no":line_no
            }
            line_no += 1
    return res

def get_img_name_to_ann_ids():

    res = defaultdict(list)

    with open(anno_error_path, 'r') as f:
        anno_error = json.load(f)
    annos = anno_error["annotations"]

    images = anno_error["images"]
    img_id_to_img_name = {}
    for image in images:
        img_id = image["id"]
        img_name = image["file_name"]
        img_id_to_img_name[img_id] = img_name

    for anno in annos:
        anno_id = anno["id"]
        image_id = anno["image_id"]
        img_name = img_id_to_img_name[image_id]
        res[img_name].append(anno_id)

    return res



def get_anno_correct_id_to_anno():
    res = {}
    with open(anno_correct_path,mode="r") as f:
        anno_correct_dict = json.load(f)
    annos = anno_correct_dict["annotations"]
    for anno in annos:
        anno_id = anno["id"]
        res[anno_id] = anno
    return res

def get_anno_correct_img_name_to_annos():
    res = defaultdict(list)

    with open(anno_correct_path,mode="r") as f:
        anno_correct_dict = json.load(f)
    
    image_id_to_img_name = {}
    images = anno_correct_dict["images"]
    for image in images:
        image_id_to_img_name[image["id"]] = image["file_name"]

    annos = anno_correct_dict["annotations"]

    for anno in annos:
        image_name = image_id_to_img_name[anno["image_id"]]
        res[image_name].append(anno)

    return res


def get_gid_to_anno_id():
    gid_to_anno_id = {}
    gid_to_img_and_line =get_gid_to_img_and_line()
    img_name_to_ann_ids = get_img_name_to_ann_ids()
    for gid in gid_to_img_and_line.keys():
        img_name = gid_to_img_and_line[gid]["img_name"]
        line_no = gid_to_img_and_line[gid]["line_no"]
        anno_id = img_name_to_ann_ids[img_name][line_no]
        gid_to_anno_id[gid] = anno_id
    return gid_to_anno_id

def get_repair_info():
    repair_info = {}
    # cur img_name to ann ids
    img_name_to_anno_ids = get_img_name_to_ann_ids()
    gid_to_anno_id = get_gid_to_anno_id()
    # {corrected_anno_id:anno}
    correct_anno_id_to_anno = get_anno_correct_id_to_anno()
    # {corrected_image_name:[anno_list]}
    correct_image_name_to_annos = get_anno_correct_img_name_to_annos()

    cut_off = int(0.4*len(rank_res))
    top_ranked_idd_list = rank_res[:cut_off]

    repair_info["miss"] = {}
    repair_info["cls_loc"] = {}
    repair_info["red"] = []
    
    for idd in top_ranked_idd_list:
        if type(idd) is str:
            missd_annos = []
            # 说明是img_name，可能是missing_fault
            image_name = idd
            # 这张图像所有的正确的annos
            correct_annos = correct_image_name_to_annos[image_name]
            # 这张图像所有的正确的anno ids，有肯能被mis,被red,被cls,被loc
            correct_anno_ids = [anno["id"] for anno in correct_annos]
            # 这张图像现在的anno_ids, 可能是red的，可能是
            cur_anno_ids = img_name_to_anno_ids[image_name]
            # 正确的有，当前没有
            missed_anno_id_set = set(correct_anno_ids) - set(cur_anno_ids)
            missed_anno_id_list = list(missed_anno_id_set)
            if len(missed_anno_id_list) > 0:
                # 该图像(idd)真的有missed annos
                for missed_anno_id in missed_anno_id_list:
                    missed_anno = correct_anno_id_to_anno[missed_anno_id]
                    missd_annos.append(missed_anno)
                repair_info["miss"][image_name] = missd_annos

        else:
            # 说明是gbox_id
            gid = idd
            anno_id = gid_to_anno_id[gid]
            if anno_id in correct_anno_id_to_anno:
                # 可能是cls_fault|loc_fault
                correct_anno = correct_anno_id_to_anno[anno_id]
                correct_cls = correct_anno["category_id"]
                correct_box = correct_anno["bbox"] # xcycwh
                repair_info["cls_loc"][anno_id] = correct_anno
            else:
                # 可能是redundancy_fault
                repair_info["red"].append(anno_id)
    return repair_info
    

def repair_anno_json(cur_anno_json,recify_info):
    annos = cur_anno_json["annotations"]
    cls_loc_xiufu_dict = recify_info["cls_loc"]

    # 修复cls_loc
    for anno in annos:
        anno_id = anno["id"]

        if anno_id in cls_loc_xiufu_dict:
            correct_anno = cls_loc_xiufu_dict[anno_id]
            anno["category_id"] = correct_anno["category_id"]
            anno["bbox"] = correct_anno["bbox"]
    # 修复miss_fault
    miss_dict = recify_info["miss"]
    for img_name, missed_anns in miss_dict.items():
        annos.extend(missed_anns)
    # 修复可能是redundancy_fault
    redundancy_idd_list = recify_info["red"]
    new_annos = []
    for anno in annos:
        if anno["id"] not in redundancy_idd_list:
            new_annos.append(anno)
    cur_anno_json["annotations"] = new_annos
    return cur_anno_json


def main():
    # 得到修复信息
    repair_info = get_repair_info()
    # 修复anno
    with open(anno_error_path,"r") as f:
        anno_error_json = json.load(f)
    new_annos = repair_anno_json(anno_error_json,repair_info)
    with open(repair_anno_save_path,"w") as f:
        json.dump(new_annos,f)
    print(f"修复的anno json保存在:{repair_anno_save_path}")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone" # VOC2012|KITTI_8|VisDrone
    model_name = "FRCNN" # YOLOv7|FRCNN|SSD
    rank_res = joblib.load(get_ours_rank_res_path(dataset_name,model_name,istopsis=True))
    gt_json_path = get_collected_gt_box_json_path(dataset_name)
    anno_error_path = get_error_ann_file_path(dataset_name)
    anno_correct_path = get_correct_ann_file_path(dataset_name,"train")
    repair_anno_save_path = os.path.join(exp_root_dir,"datasets",f"{dataset_name}-coco","train",f"_annotations.coco_repair_ours_{model_name}.json")
    main()