
from collections import defaultdict

def get_g_id_to_g_box(g_boxes_json:dict) -> dict:
    g_id_to_g_box = {}
    for img_name, g_boxes in g_boxes_json.items():
        for g_box in g_boxes:
            g_id_to_g_box[g_box["box_id"]] = g_box
    return g_id_to_g_box

def get_imgid_to_imgname(anno:dict) -> dict:
    images = anno["images"]
    img_id_to_img_name = {}
    for image in images:
        img_id = image["id"]
        img_name = image["file_name"]
        img_id_to_img_name[img_id] = img_name
    return img_id_to_img_name

def get_imgname_to_imgid(anno:dict) -> dict:
    images = anno["images"]
    img_name_to_img_id = {}
    for image in images:
        img_id = image["id"]
        img_name = image["file_name"]
        img_name_to_img_id[img_name] = img_id
    return img_name_to_img_id

def get_cls_id_to_name(anno:dict) -> dict:
    categories = anno["categories"]
    cls_id_to_name = {}
    for category in categories:
        cls_id_to_name[category["id"]] = category["name"]
    return cls_id_to_name

def get_img_name_to_ann_ids(anno:dict) -> dict:
    imgname_to_annids = defaultdict(list)
    img_id_to_img_name = get_imgid_to_imgname(anno)
    annos = anno["annotations"]
    for anno in annos:
        anno_id = anno["id"]
        image_id = anno["image_id"]
        img_name = img_id_to_img_name[image_id]
        imgname_to_annids[img_name].append(anno_id)
    return imgname_to_annids

def get_gid_to_img_and_line(g_boxes_json:dict):
    res = {}
    for img_name,g_box_list in g_boxes_json.items():
        line_no = 0
        for g_box in g_box_list:
            gid = g_box["box_id"]
            res[gid] = {
                "img_name":img_name,
                "line_no":line_no
            }
            line_no += 1
    return res

def get_gid_to_anno_id(g_boxes_json:dict,anno:dict):
    gid_to_anno_id = {}
    gid_to_img_and_line =get_gid_to_img_and_line(g_boxes_json)
    img_name_to_ann_ids = get_img_name_to_ann_ids(anno)
    for gid in gid_to_img_and_line.keys():
        img_name = gid_to_img_and_line[gid]["img_name"]
        line_no = gid_to_img_and_line[gid]["line_no"]
        anno_id = img_name_to_ann_ids[img_name][line_no]
        gid_to_anno_id[gid] = anno_id
    return gid_to_anno_id

def get_annoid_to_imgname(anno_json:dict) -> dict:
    imgid_to_imgname = get_imgid_to_imgname(anno_json)
    annoid_to_imgname = {}
    annos = anno_json["annotations"]
    for anno in annos:
        img_name = imgid_to_imgname[anno["image_id"]]
        annoid_to_imgname[anno["id"]] = img_name
    return annoid_to_imgname

def get_all_miss_img_name_list(anno_with_miss_json:dict) -> list:
    imgid_to_imgname = get_imgid_to_imgname(anno_with_miss_json)
    miss_img_name_list = []
    annos = anno_with_miss_json["annotations"]
    for anno in annos:
        if anno["fault_type"] == 4:
            img_name = imgid_to_imgname[anno["image_id"]]
            miss_img_name_list.append(img_name)
    return miss_img_name_list


