import os
from shutil import copy, rmtree
import random
from tqdm import tqdm


def main():
    '''
    split_rates: 划分比例 [训练集, 验证集, 测试集]
    init_dataset: 未划分前的数据集路径
    new_dataset : 划分后的数据集路径

    '''

    def makedir(path):
        if os.path.exists(path):
            rmtree(path)
        os.makedirs(path)

    #split_rates = [0.8, 0, 0.2]  # 训练集: 80%, 验证集: 10%, 测试集: 10%
    split_rates = [0.6, 0.2, 0.2]  # 训练集: 80%, 验证集: 10%, 测试集: 10%
    # split_rates = [0.0, 0.0, 1]
    init_dataset = r"/home/j/CVer/HZX/Awesome-Backbones-main（复件）/dataset"
    # new_dataset = 'D:/ybj/Awesome-Backbones-main/train_data/merge/all_a12345'
    new_dataset = r'/home/j/CVer/HZX/Awesome-Backbones-main（复件）/datasets'
    random.seed(0)

    classes_name = [name for name in os.listdir(init_dataset)]

    makedir(new_dataset)
    training_set = os.path.join(new_dataset, "train")
    validation_set = os.path.join(new_dataset, "validation")
    test_set = os.path.join(new_dataset, "test")
    makedir(training_set)
    makedir(validation_set)
    makedir(test_set)

    for cla in classes_name:
        makedir(os.path.join(training_set, cla))
        makedir(os.path.join(validation_set, cla))
        makedir(os.path.join(test_set, cla))

    for cla in classes_name:
        class_path = os.path.join(init_dataset, cla)
        img_set = os.listdir(class_path)
        num = len(img_set)

        # 计算划分的数量
        num_train = int(num * split_rates[0])
        num_validation = int(num * split_rates[1])
        num_test = num - num_train - num_validation

        # 随机打乱图像列表
        random.shuffle(img_set)

        # 根据划分数量将图像复制到相应的目录
        with tqdm(total=num, desc=f'Class : ' + cla, mininterval=0.3) as pbar:
            for i, img in enumerate(img_set):
                init_img = os.path.join(class_path, img)
                if i < num_train:
                    new_img = os.path.join(training_set, cla)
                elif i < num_train + num_validation:
                    new_img = os.path.join(validation_set, cla)
                else:
                    new_img = os.path.join(test_set, cla)
                copy(init_img, new_img)
                pbar.update(1)
        print()


if __name__ == '__main__':
    main()
