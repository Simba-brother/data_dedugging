
from pathlib import Path
from labelformat.formats import (YOLOv7ObjectDetectionInput, COCOObjectDetectionOutput, 
                                 COCOObjectDetectionInput, PascalVOCObjectDetectionOutput)


# Initialize input and output classes
split_flag = "train" # train|val
correct_or_error = "correct"

# yolo -> coco
yolo_input = YOLOv7ObjectDetectionInput(input_file = Path("/data/mml/data_debugging/datasets/VOC2012-yolo/data.yaml"), input_split=split_flag)
coco_output = COCOObjectDetectionOutput(output_file = Path(f"/data/mml/data_debugging/datasets/VOC2012-coco/{split_flag}/_annotations.coco_{correct_or_error}.json"))
coco_output.save(label_input=yolo_input)
print(f"Conversion from YOLOv7 to COCO completed successfully! split:{split_flag}. isError:{correct_or_error}")

'''
# coco -> xml
coco_input = COCOObjectDetectionInput(input_file=Path(f"/data/mml/data_debugging/datasets/VOC2012-coco/{split_flag}/_annotations.coco_{correct_or_error}.json"))
pascal_output = PascalVOCObjectDetectionOutput(output_folder=Path(f"/data/mml/data_debugging/datasets/VOC2012-xml/datasets_error/train"))
pascal_output.save(label_input=coco_input)
print(f"Conversion from COCO to XML completed successfully! split:{split_flag}. isError:{correct_or_error}.")
'''






