
from pathlib import Path
from labelformat.formats import (YOLOv7ObjectDetectionInput, COCOObjectDetectionOutput, 
                                 COCOObjectDetectionInput, PascalVOCObjectDetectionOutput, 
                                 KittiObjectDetectionInput, YOLOv7ObjectDetectionOutput,
                                 )
# coco -> yolov7
exp_data_root = "/data/mml/data_debugging_data"
dataset_name = "KITTI_8" # VOC2012|KITTI_8|VisDrone
coco_input_path = Path("/data/mml/data_debugging_data/datasets/VOC2012-coco/train/_annotations.coco_repair_ours_FRCNN.json")
yolo_output_path = Path("/data/mml/data_debugging_data/error_anno/VOC2012/yolo_format/repair_ours/with_topsis/data.yaml")
coco_input = COCOObjectDetectionInput(input_file=coco_input_path)
yolo_output = YOLOv7ObjectDetectionOutput(
    output_file=yolo_output_path,
    output_split="train"
)
yolo_output.save(label_input=coco_input)
print("Conversion from COCO to YOLOv7 completed successfully!")


# Load KITTI labels
'''
split = "val"
kitti_input = KittiObjectDetectionInput(
    input_folder=Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_kitti_format/{split}/labels"),
    category_names="Car,Van,Truck,Pedestrian,Person_sitting,Cyclist,Tram,Misc,DontCare", # 9 个 categories
    images_rel_path=f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_kitti_format/{split}/images"
)
coco_output = COCOObjectDetectionOutput(output_file = 
            Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_coco_format/{split}/_annotations.coco.json")
            )
coco_output.save(label_input=kitti_input)
print(f"Conversion from KITTI to COCO completed successfully! split:{split}.")


# COCO Convert to YOLOv7 and save
split = "val"
coco_input_path = Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_coco_format/{split}/_annotations.coco_noDonCare.json")
yolo_output_path = Path(f"/data/mml/data_debugging_data/datasets/no_needed_datasets/KITTI/dataset_yolo_format/{split}/data.yaml")

coco_input = COCOObjectDetectionInput(input_file=coco_input_path)
yolo_output = YOLOv7ObjectDetectionOutput(
    output_file=yolo_output_path,
    output_split=split
)
yolo_output.save(label_input=coco_input)
print(f"Conversion from COCO to YOLOv7 completed successfully! split:{split}.")
'''


'''
exp_data_root = "/data/mml/data_debugging_data"
dataset_name = "KITTI" # VOC2012|VisDrone|KITTI
# Initialize input and output classes
split_flag = "train" # train|val
correct_or_error = "error"

# yolo -> coco
yolo_input = YOLOv7ObjectDetectionInput(input_file = Path(f"{exp_data_root}/datasets/{dataset_name}-yolo/data.yaml"), input_split=split_flag)
coco_output = COCOObjectDetectionOutput(output_file = Path(f"{exp_data_root}/datasets/{dataset_name}-coco/{split_flag}/_annotations.coco_{correct_or_error}.json"))
coco_output.save(label_input=yolo_input)
print(f"Conversion from YOLOv7 to COCO completed successfully! split:{split_flag}. isError:{correct_or_error}")
'''

'''
# coco -> xml
coco_input = COCOObjectDetectionInput(input_file=Path(f"{exp_data_root}/datasets/{dataset_name}-coco/{split_flag}/_annotations.coco_{correct_or_error}.json"))
pascal_output = PascalVOCObjectDetectionOutput(output_folder=Path(f"{exp_data_root}/datasets/{dataset_name}-xml/datasets_error/train"))
pascal_output.save(label_input=coco_input)
print(f"Conversion from COCO to XML completed successfully! split:{split_flag}. isError:{correct_or_error}.")
'''






