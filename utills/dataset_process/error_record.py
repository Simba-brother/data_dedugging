import os
import pandas as pd

'''
先行脚本是label_tamper.py
'''
def simple_error_record():
    df_record = pd.read_csv(f"{exp_data_dir}/datasets/{dataset_name}_error_record/error_record.csv")
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
    df_yolo.to_csv(f"{exp_data_dir}/datasets/{dataset_name}_error_record/error_record_simple.csv", index=False)


def error_record_yolo(yolo_dataset_abs_dir):
    df_record = pd.read_csv(f"{exp_data_dir}/datasets/error_record.csv")
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
    df_yolo.to_csv(f"{exp_data_dir}/datasets/error_record-yolo.csv", index=False)

def error_record_coco(coco_dataset_abs_dir):
    df_record = pd.read_csv(f"{exp_data_dir}/datasets/error_record.csv")
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
    df_coco.to_csv(f"{exp_data_dir}/datasets/error_record-coco.csv", index=False)

def error_record_xml(xml_dataset_abs_dir):
    df_record = pd.read_csv(f"{exp_data_dir}/datasets/error_record.csv")
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
    df_xml.to_csv(f"{exp_data_dir}/datasets/error_record-xml.csv", index=False)




if __name__ == "__main__":
    
    exp_data_dir = "/data/mml/data_debugging_data"
    dataset_name = "KITTI" # VOC2012|VisDrone|KITTI

    '''
    yolo_dataset_abs_dir = f"{exp_data_dir}/datasets/{dataset_name}-yolo"
    coco_dataset_abs_dir = f"{exp_data_dir}/datasets/{dataset_name}-coco"
    xml_dataset_abs_dir = f"{exp_data_dir}/datasets/{dataset_name}-xml/datasets_error"
    '''

    simple_error_record()
    # error_record_xml()
    