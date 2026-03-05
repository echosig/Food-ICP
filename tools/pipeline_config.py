from PIL import Image
import numpy as np
from core.datasets.build import PIPELINES

@PIPELINES.register_module()
class RandomDoubleCrop:
    def __init__(self, crop_fraction=0.5, min_crop_area=0.2):
        self.crop_fraction = crop_fraction
        self.min_crop_area = min_crop_area

    def __call__(self, results):
        img = results['img']

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

        new_height = height - crop_height
        if isinstance(img, Image.Image):
            box = (0, crop_height, width, height)
            img_cropped = img.crop(box)
        elif isinstance(img, np.ndarray):
            img_cropped = img[crop_height:, :].copy()  # 使用 .copy() 创建新的数组

        # 对下半区域进行随机面积裁剪
        img_cropped_height, img_cropped_width = img_cropped.shape[:2] if isinstance(img_cropped, np.ndarray) else img_cropped.size
        min_crop_height = int(img_cropped_height * self.min_crop_area)
        min_crop_width = int(img_cropped_width * self.min_crop_area)

        crop_w = np.random.randint(min_crop_width, img_cropped_width)
        crop_h = np.random.randint(min_crop_height, img_cropped_height)

        if isinstance(img_cropped, Image.Image):
            x = np.random.randint(0, img_cropped_width - crop_w)
            y = np.random.randint(0, img_cropped_height - crop_h)
            img_cropped = img_cropped.crop((x, y, x + crop_w, y + crop_h))
        elif isinstance(img_cropped, np.ndarray):
            x = np.random.randint(0, img_cropped_width - crop_w)
            y = np.random.randint(0, img_cropped_height - crop_h)
            img_cropped = img_cropped[y:y + crop_h, x:x + crop_w].copy()  # 使用 .copy() 创建新的数组

        results['img'] = img_cropped
        return results