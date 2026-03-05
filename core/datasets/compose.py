# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from .build import build_from_cfg, PIPELINES
from PIL import Image
import copy
import numpy as np
import random

@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform_cfg = copy.deepcopy(transform)
                # transform = eval(transform_cfg.pop('type'))(**transform_cfg)
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class RandomDoubleCrop:
    def __init__(self, crop_fraction=0.5, min_crop_area=0.2, probability=0.5):
        self.crop_fraction = crop_fraction
        self.min_crop_area = min_crop_area
        self.probability = probability

    def __call__(self, results):
        img = results['img']

        # 根据概率决定是否裁剪
        if random.random() > self.probability:
            return results

        if isinstance(img, Image.Image):  # 处理 PIL 图像对象
            width, height = img.size
        elif isinstance(img, np.ndarray):  # 处理 numpy 数组
            height, width = img.shape[:2]
        else:
            raise TypeError(f"Expected 'img' to be either PIL Image or ndarray, but got {type(img).__name__}.")

        # 裁剪下半区域
        crop_height = int(height * self.crop_fraction)
        if crop_height >= height:
            raise ValueError("Crop height must be less than the image height.")

        if isinstance(img, Image.Image):
            box = (0, crop_height, width, height)
            img = img.crop(box)
        elif isinstance(img, np.ndarray):
            img = img[crop_height:, :, :]
        else:
            raise TypeError(f"Unsupported image type: {type(img).__name__}")

        results['img'] = img
        return results