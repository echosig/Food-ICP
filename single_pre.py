from utils.checkpoint import load_checkpoint
import torch
import argparse
import os
from models.build import BuildNet
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
from torchvision import transforms
from PIL import Image

classes = []
with open('E:/PyCharm_project/Awesome-Backbones-main/datas/annotations_all.txt', 'r', encoding='utf-8') as f:
    for k in f.readlines():
        if k.strip():
            classes.append(k.strip())
classes = tuple(classes)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--split-validation',
        action='store_true',
        help='whether to split validation set from training set.')
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='the proportion of the validation set to the training set.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def init_model(model, data_cfg, device='cuda:0', mode='eval'):
    """Initialize a classifier from config file.

    Returns:
        nn.Module: The constructed classifier.
    """
    if mode == 'train':
        if data_cfg.get('train').get('pretrained_flag') and data_cfg.get('train').get('pretrained_weights'):
            print('Loading {}'.format(data_cfg.get('train').get('pretrained_weights').split('/')[-1]))
            load_checkpoint(model, data_cfg.get('train').get('pretrained_weights'), device, False)


    elif mode == 'eval':
        print('Loading {}'.format(data_cfg.get('test').get('ckpt').split('/')[-1]))
        model.eval()
        load_checkpoint(model, data_cfg.get('test').get('ckpt'), device, False)

    model.to(device)

    return model

args = parse_args()
model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
model = BuildNet(model_cfg)
if args.device is not None:
    device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_model(model, data_cfg, device=device, mode='eval')
model.eval()

image_path = 'datasets_stft/train/32_22_Missing_tooth/32_22_3_5.jpg'  # 图像文件路径
outputs = model(images.to(device),return_loss=False)

# predicted_label = classes[predicted.item()]
# print('预测标签：', predicted_label)
