import os
import sys
sys.path.insert(0,os.getcwd())
from utils.train_utils import get_info

def main():
    # classes_path = 'D:/ybj/Awesome-Backbones-main/datas/annotations.txt'
    # datasets_path = 'D:/ybj/Awesome-Backbones-main/train_data/34_44bear_vs_A2_A3_img'

    classes_path = r'/home/j/CVer/HZX/Awesome-Backbones-main（复件）/datas/annotations.txt'
    # datasets_path = r'E:\PyCharm_project\Awesome-Backbones-main\train_data\3D_photos'
    datasets_path = r'/home/j/CVer/HZX/Awesome-Backbones-main（复件）/datasets'

    datasets = ["train",'validation', "test"]  # 添加验证集
    classes, indexs = get_info(classes_path)

    for dataset in datasets:
        txt_file = open('/home/j/CVer/HZX/Awesome-Backbones-main（复件）/datas/' + dataset + '.txt', 'w')
        datasets_path_ = os.path.join(datasets_path, dataset)
        classes_name = os.listdir(datasets_path_)
        for name in classes_name:
            if name not in classes:
                continue
            cls_id = indexs[classes.index(name)]
            images_path = os.path.join(datasets_path_, name)
            images_name = os.listdir(images_path)
            for photo_name in images_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    continue
                txt_file.write('%s' % (os.path.join(images_path, photo_name)) + ' ' + str(cls_id))
                txt_file.write('\n')
        txt_file.close()

if __name__ == "__main__":
    main()

