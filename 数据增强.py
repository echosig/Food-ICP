from PIL import Image
import copy
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from core.datasets.compose import Compose




class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        return len(self.gt_labels)

    def __getitem__(self, index):
        results = self.pipeline(copy.deepcopy(self.data_infos[index]))
        return results['img'], int(results['gt_label']), results['filename']

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if len(self.gt_labels) == 0:
            raise TypeError('ann_file is None')
        samples = [x.strip().rsplit(' ', 1) for x in self.gt_labels]

        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos


total_annotations = "datas/train.txt"
with open(total_annotations, encoding='utf-8') as f:
    total_datas = f.readlines()

img_lighting_cfg = dict(
    eigval=[55.4625, 4.7940, 1.1475],
    eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]],
    alphastd=0.1,
    to_rgb=True)

policies = [
    dict(type='AutoContrast', prob=0.5),
    dict(type='Equalize', prob=0.5),
    dict(type='Invert', prob=0.5),
    dict(
        type='Rotate',
        magnitude_key='angle',
        magnitude_range=(0, 30),
        pad_val=0,
        prob=0.5,
        random_negative_prob=0.5),
    dict(
        type='Sharpness',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        prob=0.3,
        random_negative_prob=0.),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=0,
        prob=0.2,
        direction='horizontal',
        random_negative_prob=0.2,
        interpolation='bicubic')
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomDoubleCrop', crop_fraction=0.4, min_crop_area=0.2),  # 添加自定义裁剪操作
    # dict(
    #     type='RandAugment',
    #     policies=policies,
    #     num_policies=8,
    #     magnitude_level=12),
    dict(
        type='RandomResizedCrop',
        size=224,
        efficientnet_style=True,
        interpolation='bicubic',
        backend='pillow'),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Lighting', **img_lighting_cfg),
    dict(
        type='Normalize',
        mean=[8.15e-05, 8.40e-05, 8.86e-05],
        std=[79.05, 86.7, 91.8],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test = Mydataset(total_datas, train_pipeline)
print(test[2])

# 获取样本
img_tensor, gt_label, filename = test[1000]

# 将图像张量转换为 NumPy 数组
img_tensor = img_tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

# 如果进行了标准化，则需要反标准化
mean = [8.15e-05, 8.40e-05, 8.86e-05]
std = [79.05, 86.7, 91.8]
img_tensor = img_tensor * torch.tensor(std) + torch.tensor(mean)

# 转换为 NumPy 数组并展示
img_np = img_tensor.numpy().astype('uint8')

plt.imshow(img_np)
plt.title(f"Label: {gt_label}, Filename: {filename}")
plt.axis('off')  # 关闭坐标轴
plt.show()
