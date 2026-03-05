from utils.checkpoint import load_checkpoint
import torch
import argparse
import os
from models.build import BuildNet
from utils.train_utils import train, validation, print_info, file2dict, init_random_seed, set_random_seed, resume_model
import torchsummary as summary
from thop import profile

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='E:/PyCharm_project/Awesome-Backbones-main/models/edgenext/edgenext_xxssmall.py', help='Config file')
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
model = init_model(model, data_cfg, device=device, mode='train')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_parameters = count_parameters(model)
print(f"Total number of trainable parameters: {model_parameters}")


print(model)