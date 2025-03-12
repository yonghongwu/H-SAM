import argparse
import logging
import os
import pytz
import datetime
# 设置该文件所在文件夹为工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from PIL import Image
from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer_synapse
from icecream import ic

import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data2/zhcheng/train_npz_224', help='root dir for data')
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--split', type=str,
                    default='train', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=2345, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint; 只对于 hsam 有效')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint; 只对 hsam 有效')
parser.add_argument('--rank', type=int, default=5, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.9)

parser.add_argument('--debug', '-d', action='store_true', help='If activated, debug mode is activated')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
parser.add_argument('--interval_epoch', type=int, default=50, help='interval epoch for saving')

parser.add_argument('--model', type=str, default='sam2', choices=['hsam', 'sam2'], help='模型选择')
parser.add_argument('--is_grpo', action='store_true', help='是否使用grPO优化')
parser.add_argument('--is_dpo', action='store_true', help='是否使用DPO优化')
parser.add_argument('--rw_dispered', action='store_true', help='是否使用离散的奖励机制')
parser.add_argument('--rw_temp', type=float, default=1., help='奖励的温度')
parser.add_argument('--grpo_KL_weight', action='store_true', help='')
parser.add_argument('--weight_temp', type=float, default=1., help='在grpo中使用权重进行调节KL的惩罚力度, 温度越小则惩罚越大(类别之间的惩罚力度差距会变大)')

parser.add_argument('--num_prompts_per_class', type=int, default=3, help='对于一张图像, 会采样出多少个prompt, 等于GRPO组的大小')
parser.add_argument('--kl_beta', type=float, default=0.05, help='调控KL散度')
parser.add_argument('--ours_use_lora', action='store_true', help='我们的方法是否使用lora')

# parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='混合精度训练类型: no=禁用, fp16=float16, bf16=bfloat16')
parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], help='训练精度: float32=全精度, float16=半精度, bfloat16=BF16精度')
parser.add_argument('--desc', type=str, default='none', help='实验说明')
args = parser.parse_args()

if args.debug:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args.n_gpu = int(torch.cuda.device_count())
    if args.model == 'sam2':
        args.root_path = '/new_wyh/Synapse-multi-organ-CT-dataset/train_npz_new_224_with_foreground/'
        args.is_grpo = True
        args.rw_dispered = False
        args.rw_temp = 3
        args.grpo_KL_weight = True
        args.weight_temp = 0.5
    elif args.model == 'hsam':
        args.root_path = '/new_wyh/Synapse-multi-organ-CT-dataset/train_npz_new_224/'
    args.split = 'train'
    args.batch_size = 2
    args.base_lr = 0.0026
    args.img_size = 224
    args.warmup = True
    args.AdamW = True
    args.max_epochs = 300
    args.stop_epoch = 300
    args.vit_name = 'vit_b'
    args.num_workers = 0
    args.ckpt = 'checkpoints/sam_vit_b_01ec64.pth'

    args.interval_epoch = 2


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    tz = pytz.timezone('Asia/Shanghai')  # 东八区对应的时区
    current_time = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    
    if args.is_grpo:
        po_type = 'grpo_dispered' if args.rw_dispered else 'grpo'
    elif args.is_dpo:
        po_type = 'dpo'
    else:
        po_type = 'nopo'
    snapshot_path = os.path.join(args.output, args.model, f"{po_type}", "{}_{}".format(current_time, args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    code_dir = os.path.join(snapshot_path, 'code')
    if not os.path.exists(code_dir):
        ignore_patterns = ['__pycache__', '.pytest_cache', '.git', '.vscode', 'data', 'checkpoints', 'env_hsam', 'env_hsam_copy', 'figure', 'output', 'sam2-main',
                        'segment_anything', 'test_outputs', 'testset', 'vis_imgs', '*.pth', '*.npy']
        shutil.copytree('../hsam_code', code_dir, ignore=shutil.ignore_patterns(*ignore_patterns))

    # register model
    if args.model == 'hsam':
        sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])

        pkg = import_module(args.module)
        net = pkg.LoRA_Sam(sam, args.rank).cuda()

        # net = LoRA_Sam(sam, args.rank).cuda()
        if args.lora_ckpt is not None:
            net.load_lora_parameters(args.lora_ckpt)
    
    elif args.model == 'sam2':
        # 需要进入sam2-main文件夹下, 然后 pip install -e .; 或者按照官网的教程安装
        # 使用: /database/wuyonghuang/hsam_code/sam2-main/test.ipynb
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if args.vit_name == 'vit_b':
            checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"    # 这个是安装的时候写的, 不是相对路径
        elif args.vit_name == 'vit_l':
            checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else: raise ValueError
        sam2 = build_sam2(model_cfg, checkpoint)
        image_size = sam2.image_size

        if args.ours_use_lora:  # Todo: 需要修改代码
            pkg = import_module(args.module)
            sam2 = pkg.LoRA_Sam(sam2, args.rank).cuda()
            sam2.image_size = image_size
            sam2.device = 'cuda'

        net = SAM2ImagePredictor(sam2)

        # 沿用 hsam 的设置
        img_embedding_size = 14

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
