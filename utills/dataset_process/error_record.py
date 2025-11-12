import os
import pandas as pd

def simple_error_record():
    df_record = pd.read_csv("/data/mml/data_debugging/datasets/VOC2012_error_record/error_record.csv")
    item_list = []
    for row_id, row in df_record.iterrows():
        txt_file_name = row["label_file_path"].split("/")[-1]
        img_file_name = txt_file_name.replace(".txt", ".jpg")
        item = {
            "img_file_name":img_file_name,
            "error_type":row["error_type"]
        }
        item_list.append(item)
    df_yolo = pd.DataFrame(item_list)
    df_yolo.to_csv("/data/mml/data_debugging/datasets/VOC2012_error_record/error_record_simple.csv", index=False)


def error_record_yolo():
    df_record = pd.read_csv("/data/mml/data_debugging/datasets/error_record.csv")
    item_list = []
    for row_id, row in df_record.iterrows():
        txt_file_name = row["label_file_path"].split("/")[-1]
        img_file_name = txt_file_name.replace(".txt", ".jpg")
        img_file_path = os.path.join(yolo_dataset_abs_dir,"train","images",img_file_name)
        item = {
            "img_file_path":img_file_path,
            "error_type":row["error_type"]
        }
        item_list.append(item)
    df_yolo = pd.DataFrame(item_list)
    df_yolo.to_csv("/data/mml/data_debugging/datasets/error_record-yolo.csv", index=False)

def error_record_coco():
    df_record = pd.read_csv("/data/mml/data_debugging/datasets/error_record.csv")
    item_list = []
    for row_id, row in df_record.iterrows():
        txt_file_name = row["label_file_path"].split("/")[-1]
        img_file_name = txt_file_name.replace(".txt", ".jpg")
        img_file_path = os.path.join(coco_dataset_abs_dir,"train",img_file_name)
        item = {
            "img_file_path":img_file_path,
            "error_type":row["error_type"]
        }
        item_list.append(item)
    df_coco = pd.DataFrame(item_list)
    df_coco.to_csv("/data/mml/data_debugging/datasets/error_record-coco.csv", index=False)

def error_record_xml():
    df_record = pd.read_csv("/data/mml/data_debugging/datasets/error_record.csv")
    item_list = []
    for row_id, row in df_record.iterrows():
        txt_file_name = row["label_file_path"].split("/")[-1]
        img_file_name = txt_file_name.replace(".txt", ".jpg")
        img_file_path = os.path.join(xml_dataset_abs_dir,"train",img_file_name)
        item = {
            "img_file_path":img_file_path,
            "error_type":row["error_type"]
        }
        item_list.append(item)
    df_xml = pd.DataFrame(item_list)
    df_xml.to_csv("/data/mml/data_debugging/datasets/error_record-xml.csv", index=False)




if __name__ == "__main__":
    
    yolo_dataset_abs_dir = "/data/mml/data_debugging/datasets/VOC2012-yolo"
    coco_dataset_abs_dir = "/data/mml/data_debugging/datasets/VOC2012-coco"
    xml_dataset_abs_dir = "/data/mml/data_debugging/datasets/VOC2012-xml/datasets_error"

    simple_error_record()
    # error_record_xml()
    