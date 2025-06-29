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

from trainer import trainer_synapse, trainer_pannuke, trainer_pannuke_batch
from icecream import ic
from sende import let_me_know

import shutil

def parse_str(s):
    if isinstance(s, bool): return s
    try:
        return int(s)
    except:
        return eval(s)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data2/zhcheng/train_npz_224', help='root dir for data')
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='[Synapse, PanNuke]')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--split', type=str,
                    default='train', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output channel of network')    # 这个应该是前景类别
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', '-b', type=int,
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
parser.add_argument('--prompt_type', type=str, default='point', choices=['point', 'box', 'both'], help='prompt 类型')
parser.add_argument('--is_strict', action='store_true', help='是否要求 point 在 box 内')
parser.add_argument('--pos_point_num', type=parse_str, default='1', help='str(1), int(1), str("(1, 3)"))')    # note: 目前只在 train_with_seg_batch 中 使用
parser.add_argument('--neg_point_num', type=parse_str, default='0', help='str(1), int(1), str("(1, 3)"))')

parser.add_argument('--kl_prompt_type', type=str, default='box', choices=['point', 'box', 'both'], help='教师模型的 prompt 类型')
parser.add_argument('--kl_is_strict', action='store_true', help='是否要求 point 在 box 内')
parser.add_argument('--kl_pos_point_num', type=parse_str, default=False, help='')    # note: 目前只在 train_with_seg_batch 中 使用
parser.add_argument('--kl_neg_point_num', type=parse_str, default=False, help='')

parser.add_argument('--is_grpo', action='store_true', help='是否使用grPO优化')
parser.add_argument('--rw_dispered', action='store_true', help='是否使用离散的奖励机制')
parser.add_argument('--rw_func', type=str, default='all', choices=['f1', 'f2', 'all'], help='离散的奖励函数类型')
parser.add_argument('--rw_temp', type=float, default=1., help='奖励的温度')
parser.add_argument('--grpo_KL_weight', action='store_true', help='?')
parser.add_argument('--weight_temp', type=float, default=1., help='在grpo中使用权重进行调节KL的惩罚力度, 温度越小则惩罚越大(类别之间的惩罚力度差距会变大)')

parser.add_argument('--is_dpo', action='store_true', help='是否使用DPO优化')
parser.add_argument('--dpo_weight', type=float, default=1., help='')
parser.add_argument('--abla_kl', action='store_true', help='把kl消融掉')
parser.add_argument('--abla_dpo', action='store_true', help='把dpo消融掉')

parser.add_argument('--dev', action='store_true', help='来自 fine-tune-train_segment_anything_2_in_60_lines_of_code, 未调试成功')

parser.add_argument('--num_prompts_per_class', type=int, default=3, help='对于一张图像, 会采样出多少个prompt, 等于GRPO组的大小')
parser.add_argument('--kl_beta', type=float, default=0.05, help='调控KL散度')
parser.add_argument('--ours_use_lora', action='store_true', help='我们的方法是否使用lora')

# parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='混合精度训练类型: no=禁用, fp16=float16, bf16=bfloat16')
parser.add_argument('--precision', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'], help='训练精度: float32=全精度, float16=半精度, bfloat16=BF16精度')
parser.add_argument('--exp_series', type=str, default='EXPS1', help='实验系列名称')
parser.add_argument('--exp_name', type=str, default='exp1', help='实验名称')
parser.add_argument('--desc', type=str, default='none', help='实验说明')

parser.add_argument('--onlybest_in_multimask_output', action='store_true', help='If activated, only the best mask will be used in multimask_output')

parser.add_argument('--pred_score_task', action='store_true', help='是否要对 iou 的预测进行训练')

parser.add_argument('--semi', action='store_true', help='是否开启半监督任务')
parser.add_argument('--semi_ratio', type=float, default=0.1, help='半监督任务中有标注数据的比例')
parser.add_argument('--use_unet', action='store_true', help='是否使用 unet 来支持半监督或者其他算法')

parser.add_argument('--only_train_unet', action='store_true', help='是否要进行unet的监督训练; 级别>grpo=dpo')
args = parser.parse_args()

if args.debug:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.n_gpu = int(torch.cuda.device_count())
    if args.model == 'sam2':
        args.root_path = '/database/wuyonghuang/hsam_code/data/multi-organ-CT/train_npz_512/'

        args.dataset = 'PanNuke'

        args.pred_score_task = True

        args.pos_point_num = (1, 5)
        args.neg_point_num = (0, 5)

        # GRPO 训练的设置
        # args.is_grpo = True
        # args.rw_dispered = False
        # args.rw_temp = 3
        # args.grpo_KL_weight = True
        # args.weight_temp = 0.5

        # DPO 训练的设置
        args.is_dpo = True
        # args.abla_dpo = True

        # args.dev = True

        args.onlybest_in_multimask_output = False
        args.ours_use_lora = True

        args.semi = False    # Todo: 半监督方法, 开发中
        args.semi_ratio = 0.1
        args.use_unet = True   # 默认使用 unet 模型

        args.only_train_unet = False

    elif args.model == 'hsam':
        args.root_path = '/new_wyh/Synapse-multi-organ-CT-dataset/train_npz_new_224/'
    
    if args.model == 'sam2' and args.semi and args.semi_ratio == 0.1:
        args.split = 'train_220_clean'  # 剔除了只有背景的slice(数据)
    elif args.model == 'sam2' and not args.semi:
        args.split = 'train_clean'  # 剔除了只有背景的slice(数据)
    else:
        args.split = 'train'

    args.batch_size = 4
    args.base_lr = 0.0026
    args.img_size = 224
    args.warmup = True
    args.AdamW = True
    args.max_epochs = 300
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
    
    if not args.only_train_unet:
        if args.is_grpo:
            po_type = 'grpo_dispered' if args.rw_dispered else 'grpo'
        elif args.is_dpo:
            po_type = 'dpo'
        elif args.dev:
            po_type = 'dev'
        else:
            po_type = 'nopo'
    else:
        # only_train_unet 级别更高
        args.is_grpo, args.is_dpo, args.dev = False, False, False
        po_type = 'nopo-unet'
    snapshot_path = os.path.join(args.output, args.exp_series, args.exp_name, f"{args.model}-{po_type}", "{}_{}".format(current_time, args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if args.debug: snapshot_path = '/database/wuyonghuang/hsam_code/output/debug'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    code_dir = os.path.join(snapshot_path, 'code')
    if not os.path.exists(code_dir):
        ignore_patterns = ['__pycache__', '.pytest_cache', '.git', '.vscode', 'data', 'checkpoints', '*env_hsam', '*env_hsam_copy', 'figure', 'output', 'sam2-main',
                        'segment_anything', 'test_outputs', 'testset', 'vis_imgs', '*.pth', '*.npy', 'python*', 'uv*']
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

        # 测试lora_sam2
        pkg = import_module(args.module)

        if args.ours_use_lora:
            from sam2_lora import add_lora_to_sam2, LoRA_Adapter
            # 第二种, 梯度没问题, 但是应用的是 HSAM 中的 注意力 lora
            # sam2 = pkg.LoRA_Sam2(sam2, args.rank).sam
            # net = SAM2ImagePredictor(sam2)
            # 保存 lora 参数

            # 第四种, 梯度没问题, 对所有线性层使用 lora
            sam2 = pkg.LoRA_Sam3(sam2, rank=4, target_modules=["Linear"])
            net = SAM2ImagePredictor(sam2.model)
            save_func = pkg.LoRA_Sam3.save_lora_weights
            load_func = pkg.LoRA_Sam3.load_lora_weights # sum(p.numel() for n, p in torch.load(pth).items())
            args.utils = {'save_func': save_func, 'load_func': load_func}
        
        else:
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

    trainer = {'Synapse': trainer_synapse, 'PanNuke': trainer_pannuke_batch}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
    let_me_know(f'Finish', 'SAM-Exps')