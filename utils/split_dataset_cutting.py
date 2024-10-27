import os
import random
import shutil

def split_dataset(sub_img_folder, train_txt_path, test_txt_path, extra_save_dir=None, test_ratio=0.1):
    """
    将数据集分割为训练集和测试集,并可选择保存到额外的路径
    
    :param sub_img_folder: 子图文件夹路径
    :param train_txt_path: 训练集txt文件保存路径
    :param test_txt_path: 测试集txt文件保存路径
    :param extra_save_dir: 额外的保存目录路径(可选)
    :param test_ratio: 测试集占总数据集的比例,默认为0.1
    """
    
    def ensure_file_exists(file_path):
        if not os.path.exists(file_path):
            open(file_path, 'w').close()
            print(f"已创建文件：{file_path}")

    ensure_file_exists(train_txt_path)
    ensure_file_exists(test_txt_path)

    sub_img_files = os.listdir(sub_img_folder)

    sub_img_dict = {}
    for sub_img in sub_img_files:
        full_img_name = '_'.join(sub_img.split('_')[:-1])
        if full_img_name not in sub_img_dict:
            sub_img_dict[full_img_name] = []
        sub_img_dict[full_img_name].append(sub_img)

    full_img_names = list(sub_img_dict.keys())
    test_full_imgs = random.sample(full_img_names, k=int(len(full_img_names) * test_ratio))

    train_list = []
    test_list = []

    for full_img_name, sub_imgs in sub_img_dict.items():
        if full_img_name in test_full_imgs:
            test_list.extend(sub_imgs)
        else:
            train_list.extend(sub_imgs)

    def remove_extension(file_name):
        return os.path.splitext(file_name)[0]

    def write_to_file(file_path, img_list):
        with open(file_path, 'w') as file:
            for img in img_list:
                file.write(remove_extension(img) + '\n')

    write_to_file(train_txt_path, train_list)
    write_to_file(test_txt_path, test_list)

    print("训练集和测试集已成功写入原始 txt 文件！")

    # 如果提供了额外的保存目录,则复制文件到该目录
    if extra_save_dir:
        os.makedirs(extra_save_dir, exist_ok=True)
        extra_train_path = os.path.join(extra_save_dir, 'train.txt')
        extra_test_path = os.path.join(extra_save_dir, 'test.txt')
        
        shutil.copy(train_txt_path, extra_train_path)
        shutil.copy(test_txt_path, extra_test_path)
        
        print(f"训练集和测试集已额外保存到: {extra_save_dir}")

# 如果直接运行此脚本,则使用默认参数
if __name__ == "__main__":
    sub_img_folder = r'/home/czh/Sigma-main/datasets/Soil/Cutting'
    train_txt_path = r'/home/czh/Sigma-main/datasets/Soil/train.txt'
    test_txt_path = r'/home/czh/Sigma-main/datasets/Soil/test.txt'
    extra_save_dir = r'/home/czh/Sigma-main/models/encoders/selective_scan/datasets/Soil'
    
    split_dataset(sub_img_folder, train_txt_path, test_txt_path, extra_save_dir)
