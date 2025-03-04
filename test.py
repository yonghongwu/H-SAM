import os
import sys
import pytz, datetime
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset

from icecream import ic


class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}


def inference(args, multimask_output, db_config, model, test_save_path=None, save_nii=False):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print(f'{len(testloader)} test iterations per epoch')
    # model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'], mode='test',stage=args.stage, 
                                      model=args.model, save_nii=args.is_savenii)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        break
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes + 1):
        try:
            print('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
        except:
            print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    print("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='./test_outputs')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,default=2345, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default=None, help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='outputs/299.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=5, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--mode',  type=str, default='test')

    parser.add_argument('--model', type=str, default='hsam', choices=['hsam', 'sam2'], help='模型选择')
    parser.add_argument('--debug', '-d', action='store_true', help='If activated, debug mode is activated')

    args = parser.parse_args()

    if args.debug:
        args.lora_ckpt = None# '/database/wuyonghuang/hsam_code/220_epoch_299.pth'
        args.vit_name = 'vit_b'
        args.ckpt = None# 'checkpoints/sam_vit_b_01ec64.pth'
        args.stage = 3
        args.img_size = 512
        args.model = 'sam2'

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

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
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    tz = pytz.timezone('Asia/Shanghai')  # 东八区对应的时区
    current_time = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    
    if args.model == 'hsam':
        lora_ckpt_name = os.path.basename(args.lora_ckpt)
        if args.ckpt is None:
            args.ckpt = 'checkpoints/sam_vit_b_01ec64.pth'
        args.output_dir = os.path.join(args.output_dir, f"{args.model}", f"{current_time}-{args.vit_name}-{lora_ckpt_name}")

        # register model
        sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                        num_classes=args.num_classes,
                                                                        checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                        pixel_std=[1, 1, 1])
        
        pkg = import_module(args.module)
        net = pkg.LoRA_Sam(sam, args.rank).cuda()

        assert args.lora_ckpt is not None
        net.load_lora_parameters(args.lora_ckpt)

    elif args.model == 'sam2':
        if args.ckpt is not None:
            ckpt_name = os.path.basename(args.ckpt)
        else:
            ckpt_name = 'noneckpt'
        args.output_dir = os.path.join(args.output_dir, f"{args.model}", f"{current_time}-{args.vit_name}-{ckpt_name}")

        # 需要进入sam2-main文件夹下, 然后 pip install -e .; 或者按照官网的教程安装
        # 使用: /database/wuyonghuang/hsam_code/sam2-main/test.ipynb
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        # checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"    # 这个是安装的时候写的, 不是相对路径

        if args.vit_name == 'vit_b':
            checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"    # 这个是安装的时候写的, 不是相对路径
        elif args.vit_name == 'vit_l':
            checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else: raise ValueError

        sam2 = build_sam2(model_cfg, checkpoint)

        if args.ckpt is not None:
            sam2.load_state_dict(torch.load(args.ckpt))
            print('成功加载参数')
        net = SAM2ImagePredictor(sam2)
        img_embedding_size = 14

    else: raise ValueError

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=print,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    print(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = args.output_dir
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path, save_nii=args.is_savenii)
