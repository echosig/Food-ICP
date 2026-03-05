import os
import cv2
import numpy as np
from tqdm import tqdm

# 遍历包含类别子文件夹的主文件夹
train_data_folder = r'D:\PyCharm_project\Awesome-Backbones-main\train_data\Eggtart\train'
class_folders = os.listdir(train_data_folder)

# 初始化各通道的累计和
mean_r, mean_g, mean_b = 0, 0, 0
std_r, std_g, std_b = 0, 0, 0
total_pixels = 0

# 遍历每个类别文件夹
for class_folder in class_folders:
    class_folder_path = os.path.join(train_data_folder, class_folder)

    for file_name in tqdm(os.listdir(class_folder_path), desc=f"Calculating for {class_folder}"):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(class_folder_path, file_name)
            image = cv2.imread(file_path)
            image = image / 255.0  # 将像素值标准化到[0, 1]范围
            total_pixels += image.size // 3  # 除以3是因为有3个通道

            mean_r += np.mean(image[:, :, 0])
            mean_g += np.mean(image[:, :, 1])
            mean_b += np.mean(image[:, :, 2])

# 计算均值并保留4位有效数字
# mean_r = round(mean_r / total_pixels, 4)
# mean_g = round(mean_g / total_pixels, 4)
# mean_b = round(mean_b / total_pixels, 4)
mean_r = mean_r / total_pixels
mean_g = mean_g / total_pixels
mean_b = mean_b / total_pixels

# 重新遍历数据集以计算标准差并保留2位有效数字
for class_folder in class_folders:
    class_folder_path = os.path.join(train_data_folder, class_folder)

    for file_name in tqdm(os.listdir(class_folder_path), desc=f"Calculating for {class_folder}"):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(class_folder_path, file_name)
            image = cv2.imread(file_path)
            image = image / 255.0

            std_r += np.sum((image[:, :, 0] - mean_r) ** 2)
            std_g += np.sum((image[:, :, 1] - mean_g) ** 2)
            std_b += np.sum((image[:, :, 2] - mean_b) ** 2)

# 计算标准差并保留2位有效数字
std_r = round(np.sqrt(std_r / total_pixels), 2)
std_g = round(np.sqrt(std_g / total_pixels), 2)
std_b = round(np.sqrt(std_b / total_pixels), 2)

# 创建img_norm_cfg
img_norm_cfg = dict(
    mean=[mean_r * 255.0, mean_g * 255.0, mean_b * 255.0],  # 还原到[0, 255]范围
    std=[std_r * 255.0, std_g * 255.0, std_b * 255.0],  # 还原到[0, 255]范围
    to_rgb=True
)

print("img_norm_cfg:", img_norm_cfg)
