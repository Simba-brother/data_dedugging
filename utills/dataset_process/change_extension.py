import os
def change_extension_to_jpg(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 检查是否为文件
        if os.path.isfile(file_path):
            # 分割文件名和扩展名
            name, ext = os.path.splitext(filename)
            # 如果不是.jpg，则更改扩展名为.jpg
            if ext != '.jpg':
                new_filename = name + '.jpg'
                new_file_path = os.path.join(folder_path, new_filename)
                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f'Renamed "{filename}" to "{new_filename}"')

def check_extension(folder_path):
    ext_set = set()
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 检查是否为文件
        if os.path.isfile(file_path):
            # 分割文件名和扩展名
            name, ext = os.path.splitext(filename)
            ext_set.add(ext)
    print(ext_set)

if __name__ == "__main__":
    check_extension("/data/mml/data_debugging_data/datasets/VisDrone/images/val")