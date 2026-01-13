
import os
from ours.base_data_manager import exp_data_root_dir
from ours.small_utils import read_json
from ours.data_organization_tools import get_gid_to_anno_id,get_annoId_to_anno
from ours.base_data_manager import get_imgs_dir
from PIL import Image

def get_anno_ids(anno_json:dict)->list:
    anno_ids = []
    annos = anno_json["annotations"]
    for anno in annos:
        annoId = anno["id"]
        anno_ids.append(annoId)
    return anno_ids

def yolo_box_2_coco_box(yolo_box, width, height)->list:
    coco_box = []
    center_x = yolo_box[0]
    center_y = yolo_box[1]
    width_norm = yolo_box[2]
    height_norm = yolo_box[3]

    x_center = center_x * width
    y_center = center_y * height
    box_width = width_norm * width
    box_height = height_norm * height
    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2
    return [x_min, y_min, box_width, box_height]

def main():
    anno_with_mis_json = read_json(anno_with_mis_path)
    anno_no_mis_json = read_json(anno_no_mis_path)
    anno_error_json = read_json(anno_error_path)
    g_boxes_json = read_json(g_boxes_json_path)

    with_mis_anno_ids = get_anno_ids(anno_with_mis_json)
    no_mis_anno_ids = get_anno_ids(anno_no_mis_json)
    anno_ids = get_anno_ids(anno_error_json)

    with_mis_anno_id_set = set(with_mis_anno_ids)
    no_mis_anno_id_set = set(no_mis_anno_ids)
    used_anno_ids = set(anno_ids)

    assert no_mis_anno_id_set == used_anno_ids, "数据不对"
    
    intersection = used_anno_ids & with_mis_anno_id_set

    assert intersection == set(used_anno_ids), "数据不对"


    gid_to_annoId =get_gid_to_anno_id(g_boxes_json, anno_error_json)
    annoId_to_anno = get_annoId_to_anno(anno_with_mis_json)
    g_box_nums = 0
    imgs_dir = get_imgs_dir(dataset_name,"train","coco")
    for img_name,g_boxes in g_boxes_json.items():
        img_path = os.path.join(imgs_dir,img_name)
        # 图像的width,height
        image = Image.open(img_path)
        width, height = image.size
        for g_box in g_boxes:
            g_id = g_box["box_id"]
            anno_id = gid_to_annoId[g_id]
            yolo_box = g_box["gt_bbox"]
            anno = annoId_to_anno[anno_id]
            coco_box = anno["bbox"]
            converted_coco_box = yolo_box_2_coco_box(yolo_box,width,height)
            if anno_id == 129285 or anno_id == 61916:
                print()

            

    print("检查通过")



if __name__ == "__main__":
    dataset_name = "VisDrone"
    anno_with_mis_path = os.path.join(exp_data_root_dir,"error_anno",dataset_name, "coco_format","annotations_with_miss.json")
    anno_no_mis_path = os.path.join(exp_data_root_dir,"error_anno",dataset_name, "coco_format","annotations_no_miss.json")
    anno_error_path = os.path.join(exp_data_root_dir,"datasets",f"{dataset_name}-coco","train","_annotations.coco_error.json")
    
    g_boxes_json_path = "/data/mml/data_debugging_data/collection_indicator_bbox_level/VisDrone/YOLOv7/gt_bboxs.json"

    main()