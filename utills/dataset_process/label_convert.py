
from pathlib import Path
from labelformat.formats import (YOLOv7ObjectDetectionInput, COCOObjectDetectionOutput, 
                                 COCOObjectDetectionInput, PascalVOCObjectDetectionOutput, 
                                 KittiObjectDetectionInput, YOLOv7ObjectDetectionOutput)

'''
# Load KITTI labels
label_input = KittiObjectDetectionInput(
    input_folder=Path("/data/mml/data_debugging_data/datasets/KITTI-yolo/train/labels"),
    category_names="Car,Van,Truck,Pedestrian,Person_sitting,Cyclist,Tram,Misc,DontCare",
    images_rel_path="/data/mml/data_debugging_data/datasets/KITTI-yolo/train/images"
)

# Convert to YOLOv8 and save
YOLOv7ObjectDetectionOutput(
    output_file=Path("/data/mml/data_debugging_data/datasets/KITTI-yolo/yolo_format/data.yaml"),
    output_split="train"
).save(label_input=label_input)
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
# coco -> xml
coco_input = COCOObjectDetectionInput(input_file=Path(f"{exp_data_root}/datasets/{dataset_name}-coco/{split_flag}/_annotations.coco_{correct_or_error}.json"))
pascal_output = PascalVOCObjectDetectionOutput(output_folder=Path(f"{exp_data_root}/datasets/{dataset_name}-xml/datasets_error/train"))
pascal_output.save(label_input=coco_input)
print(f"Conversion from COCO to XML completed successfully! split:{split_flag}. isError:{correct_or_error}.")
'''






