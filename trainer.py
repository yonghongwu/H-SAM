import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_dataset
from load_pannuke_v3 import demo_usage

from contextlib import nullcontext
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic
from PIL import Image

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from opt_utils import vanilla_opt, train_with_po, train_with_seg_single, train_with_seg_batch, val_with_seg_batch, vanilla_eva

def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]


# def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8, ssl=False):
#     low_res_logits = outputs['low_res_logits']
#     # print(low_res_logits.size())
#     # print(low_res_label_batch.size())
#     loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
#     loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
#     loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
#     return loss, loss_ce, loss_dice
def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = ((1 - dice_weight) * loss_ce + dice_weight * loss_dice)
    return loss, loss_ce, loss_dice

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
#椒盐噪声
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img

def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    from dataset.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=tqdm.write,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    tqdm.write(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split=args.split,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
                                   ]))
    db_test = Synapse_dataset(base_dir='testset/test_vol_h5/', list_dir='./lists/lists_Synapse/', split='test_vol', transform=None)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.semi:   # note: 目前只支持 数据集一, 尚不支持 pannuke
        from semi_utils import semi_split, LabeledDataset, UnlabeledDataset
        labeled_indices, unlabeled_indices = semi_split(db_train, labeled_ratio=0.1)

        print(f"Actual labeled indices count: {len(labeled_indices)}")
        print(f"Actual unlabeled indices count: {len(unlabeled_indices)}")

        # --- 5. 实例化新的 Dataset ---
        labeled_dataset, unlabeled_dataset = LabeledDataset(db_train, labeled_indices), UnlabeledDataset(db_train, unlabeled_indices)
        trainloader = DataLoader(labeled_dataset, batch_size=batch_size//2, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
        trainloader_un = DataLoader(unlabeled_dataset, batch_size=batch_size//2, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
    else:
        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
        trainloader_un = None

    if args.model == 'hsam':
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
            # model = model.module
        model.train()
        
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

        ce_loss = CrossEntropyLoss()
        cos_loss = CosineEmbeddingLoss()
        l1_loss = nn.L1Loss()
        dice_loss = DiceLoss(num_classes + 1)
        # dice_loss = Focal_loss(num_classes=num_classes + 1)
        if args.warmup:
            b_lr = base_lr / args.warmup_period
        else:
            b_lr = base_lr
        if args.AdamW:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    
    elif args.model == 'sam2':
        if args.n_gpu > 1:
            model.model = nn.DataParallel(model.model)
            # model = model.module
        model.model.train()
        model.model.cuda()
        
        model_total_params = sum(p.numel() for p in model.model.parameters())
        model_grad_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(num_classes + 1)
        # dice_loss = Focal_loss(num_classes=num_classes + 1)
        if args.warmup:
            b_lr = base_lr / args.warmup_period
        else:
            b_lr = base_lr
        if args.AdamW:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-5)
            # optimizer = optim.AdamW([list(model.model.parameters())[-3]], lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-5)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.model.parameters()), lr=b_lr, momentum=0.9, weight_decay=args.weight_decay*0.001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
        
        if (args.semi and args.use_unet) or args.only_train_unet:   # 两种情况: ①半监督下要是否使用unet; 2 是否只训练unet
            import segmentation_models_pytorch as smp
            net_semi = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=9) # note: 这个是小模型, 用来支持半监督或者其他算法
            optimizer_semi = optim.AdamW(filter(lambda p: p.requires_grad, net_semi.parameters()), lr=1e-4)

            net_semi.train()
            net_semi.cuda()
        elif args.semi and not args.use_unet:
            raise ValueError
        else:
            net_semi = None
            optimizer_semi = None

    else: raise ValueError

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    tqdm.write("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kwargs = {}
    for epoch_num in iterator:
        for i_batch, sampled_batch in tqdm(enumerate(trainloader)):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            # 加载半监督未标注数据
            if trainloader_un is not None:
                unlabeled_iter = iter(trainloader_un)
                batch_un = next(unlabeled_iter)
                image_batch_un, label_batch_un = batch_un['image'].cuda(), batch_un['label'].cuda() # shape: (1, 3, 224, 224), (1, 224, 224)
                kwargs = {'unlabel_data': True, 'image_un': image_batch_un, 'label_un': label_batch_un, 'net_semi': net_semi, 'optimizer_semi': optimizer_semi}
                # pred = net_semi(image_batch_un)
            elif args.only_train_unet:  # 如果要对unet进行监督训练, 那么这个要准备模型和优化器, 但是不用准备无标注数据
                kwargs = {'net_semi': net_semi, 'optimizer_semi': optimizer_semi}

            num_prompts_per_class = args.num_prompts_per_class
            if args.model == 'hsam':
                outputs1,outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=low_res_label_batch)
                loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
                loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
                weight = 0.6**(0.990**epoch_num)
                weight_self = 0.2
                loss = (1-weight)*loss1 + (weight)*loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif args.model == 'sam2':
                dtype = None if args.precision == 'float32' else getattr(torch, args.precision) # 设置精度和上下文
                amp_context = nullcontext() if dtype is None else torch.autocast(device_type="cuda", dtype=dtype)
                scaler = torch.amp.GradScaler() if args.precision in ['float16', 'bfloat16'] else None    # # 只有float16需要梯度缩放

                with amp_context:  # torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)
                    # 对于不同的 prompt 导致的不同的预测结果, 计算出奖励, 然后使用 GRPO 进行优化: 奖励是由其他模型提供的, 那么怎么对本模型进行优化呢？
                    # 奖励的方式: 训练一个 prompt奖励回归器、基于 iou_prediction、基于预测结果跟 ground truth 的 iou、基于更大的模型
                    if args.is_grpo or args.is_dpo: # TODO: 在DPO中加入半监督的数据处理
                        loss, _ = train_with_po(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class,
                                                        beta=args.kl_beta, iteration=iter_num, writer=writer, args=args, **kwargs)
                    elif args.dev:
                        loss, _ = train_with_seg_batch(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class, 
                                                       beta=args.kl_beta, iteration=iter_num, writer=writer, args=args, **kwargs)
                    else:
                        loss, _ = vanilla_opt(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class, args=args, 
                                              only_train_unet=args.only_train_unet, **kwargs)

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            if args.model == 'hsam':
                writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
                writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
                writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
                writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)
            elif args.model == 'sam2' and args.is_grpo:
                pass # writer.add_scalar('info/avg_reward', avg_reward, iter_num)
            else: pass
            # writer.add_scalar('info/loss_self2', loss_self, iter_num)

            # tqdm.write('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item()))
            if args.model == 'hsam':
                tqdm.write('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f, loss_ce2: %f, loss_dice2: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item(), loss_ce2.item(), loss_dice2.item()))
            else:
                tqdm.write('iteration %d : loss : %f' % (iter_num, loss))
            
            if args.debug and i_batch == 0:
                break

        save_interval = args.interval_epoch # int(max_epoch/6) Todo: 增加参数
        if args.debug or (epoch_num + 1) % save_interval == 0:
            infer_losses, infer_ious, infer_netsemi_iou, infer_netsemi_dice = [], [], [], []
            for test_case in db_test:
                images, labels, name = test_case['image'], test_case['label'], test_case['case_name']   # (slices, 512, 512), (slices, 512, 512), string
                infer_loss, infer_iou = 0, 0
                infer_loss, infer_iou, net_semi_res = vanilla_eva(model, 
                                                                  torch.from_numpy(images).unsqueeze(1).repeat(1, 3, 1, 1), # .astype(np.float32)
                                                                  torch.from_numpy(labels), num_prompts_per_class=1, args=args, **kwargs)
                infer_losses.append(infer_loss), infer_ious.append(infer_iou)
                infer_netsemi_iou.append(net_semi_res[0]), infer_netsemi_dice.append(net_semi_res[1])

                if args.debug:
                    break
            
            writer.add_scalar('eva/infer_losses', np.mean(infer_losses), epoch_num)
            writer.add_scalar('eva/infer_ious', np.mean(infer_ious), epoch_num)

            save_path_suffix = '_lora' if args.ours_use_lora else ''
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}-loss_{np.mean(infer_losses):.04f}-iou_{np.mean(infer_ious):.04f}{save_path_suffix}.pth')

            if kwargs.get('net_semi', None):
                # torch.save(kwargs['net_semi'].state_dict(), f'unet_{epoch_num:02d}.pth')
                # infer_netsemi_iou是一个列表, 每一个元素是一个字典, 字典中按照类别存储iou指标的值, 现在需要根据这个列表计算出整体的iou值
                netsemi_iou_macro = np.mean([np.mean(list(iou_dict.values())) for iou_dict in infer_netsemi_iou])
                netsemi_dice_macro = np.mean([np.mean(list(dice_dict.values())) for dice_dict in infer_netsemi_dice])
                save_netsemi_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}-netsemi-iou_{np.mean(netsemi_iou_macro):.04f}.pth')

            if args.model == 'hsam':
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            elif args.model == 'sam2':
                if args.ours_use_lora:
                    args.utils['save_func'](model.model, save_mode_path)
                else:
                    try:
                        torch.save(model.model.state_dict(), save_mode_path)
                    except:
                        torch.save(model.model.module.state_dict(), save_mode_path)
                if kwargs.get('net_semi', None):
                    torch.save(kwargs['net_semi'].state_dict(), save_netsemi_path)
            tqdm.write("save model to {}".format(save_mode_path))

        if args.debug or epoch_num >= max_epoch - 1:
            save_path_suffix = '_lora' if args.ours_use_lora else ''
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}{save_path_suffix}.pth')
            if args.model == 'hsam':
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            elif args.model == 'sam2':
                if args.ours_use_lora:
                    args.utils['save_func'](model.model, save_mode_path)
                else:
                    try:
                        torch.save(model.model.state_dict(), save_mode_path)
                    except:
                        torch.save(model.model.module.state_dict(), save_mode_path)
                if kwargs.get('net_semi', None):
                    torch.save(kwargs['net_semi'].state_dict(), save_netsemi_path)
            tqdm.write("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_pannuke(args, model, snapshot_path, multimask_output, low_res):
    from dataset.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=tqdm.write,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    tqdm.write(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    # dataset = load_dataset("/database/wuyonghuang/PanNuke/")
    # db_train = dataset['fold1']
    # db_test = dataset['fold2']

    # from load_pannuke_v3 import split_train_val_test
    # split_train_val_test(dataset=dataset, val_ratio=0.1, test_ratio=0.1)

    from datasets import load_from_disk
    dataset = load_from_disk("./data/split_pannuke")
    db_train = dataset['train']
    db_val = dataset['validation']
    db_test = dataset['test']

    # result1 = demo_usage(sample, mode='all_nuclei') # 'all_nuclei' or 'specific_category'

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.semi:
        raise ValueError("Semi-supervised learning is not supported for pannuke dataset.")
    else:
        # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
        #                          worker_init_fn=worker_init_fn)
        # trainloader_un = None
        trainloader = None

    if args.model == 'hsam':
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
            # model = model.module
        model.train()
        
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

        ce_loss = CrossEntropyLoss()
        cos_loss = CosineEmbeddingLoss()
        l1_loss = nn.L1Loss()
        dice_loss = DiceLoss(num_classes + 1)
        # dice_loss = Focal_loss(num_classes=num_classes + 1)
        if args.warmup:
            b_lr = base_lr / args.warmup_period
        else:
            b_lr = base_lr
        if args.AdamW:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    
    elif args.model == 'sam2':
        if args.n_gpu > 1:
            model.model = nn.DataParallel(model.model)
            # model = model.module
        model.model.train()
        model.model.cuda()
        
        model_total_params = sum(p.numel() for p in model.model.parameters())
        model_grad_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(num_classes + 1)
        # dice_loss = Focal_loss(num_classes=num_classes + 1)
        if args.warmup:
            b_lr = base_lr / args.warmup_period
        else:
            b_lr = base_lr
        if args.AdamW:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-5)
            # optimizer = optim.AdamW([list(model.model.parameters())[-3]], lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-5)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.model.parameters()), lr=b_lr, momentum=0.9, weight_decay=args.weight_decay*0.001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
        
        if (args.semi and args.use_unet) or args.only_train_unet:   # 两种情况: ①半监督下要是否使用unet; 2 是否只训练unet
            import segmentation_models_pytorch as smp
            net_semi = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=9) # note: 这个是小模型, 用来支持半监督或者其他算法
            optimizer_semi = optim.AdamW(filter(lambda p: p.requires_grad, net_semi.parameters()), lr=1e-4)

            net_semi.train()
            net_semi.cuda()
        elif args.semi and not args.use_unet:
            raise ValueError
        else:
            net_semi = None
            optimizer_semi = None

    else: raise ValueError

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(db_train)  # max_epoch = max_iterations // len(trainloader) + 1
    num_prompts_per_class = args.num_prompts_per_class
    tqdm.write("{} iterations per epoch. {} max iterations ".format(len(db_train), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kwargs = {}
    for epoch_num in iterator:
        for i_batch, sample in tqdm(enumerate(db_train)):
            # point 的数量以及分布; prompts2(教师用的prompt应该覆盖所有的实例)
            if isinstance(args.pos_point_num, tuple):
                num_pos_points = np.random.randint(*args.pos_point_num, size=(1))[0]
            else: num_pos_points = args.pos_point_num
            if isinstance(args.neg_point_num, tuple):
                num_neg_points = np.random.randint(*args.neg_point_num, size=(1))[0]
            else: num_neg_points = args.neg_point_num

            prompts = demo_usage(sample, mode='all_nuclei', num_positive_points=num_pos_points, num_negative_points=num_neg_points, num_batches=num_prompts_per_class, all_instance_mode=True) # 'all_nuclei' or 'specific_category'
            prompts2 = demo_usage(sample, mode='all_nuclei', num_positive_points=100, num_negative_points=100, num_batches=num_prompts_per_class, all_instance_mode=True) # 'all_nuclei' or 'specific_category'
            if prompts is None or prompts2 is None: continue
            new_prompts = {'class_prompts':
                           {1: prompts['prompts']},
                            'decoded_mask': {1: prompts['target_mask']}}
            new_prompts2 = {'class_prompts':
                           {1: prompts2['prompts']},
                            'decoded_mask': {1: prompts2['target_mask']}}
            kwargs.update({'prompts': prompts, 'new_prompts': new_prompts, 
                           'prompts2': prompts2, 'new_prompts2': new_prompts2})

            image_batch = prompts['image']   # numpy: (256, 256, 3)
            label_batch = prompts['target_mask'] # numpy: (256, 256)

            image_batch, label_batch = torch.from_numpy(image_batch).permute(2, 0, 1)[None].cuda(), torch.from_numpy(label_batch)[None].cuda()
            # low_res_label_batch = sampled_batch['low_res_label']
            # low_res_label_batch = low_res_label_batch.cuda()
            low_res_label_batch = None
            # assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            # 加载半监督未标注数据, 暂停开发

            if args.model == 'hsam':
                # outputs1,outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=low_res_label_batch)
                # loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
                # loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
                # weight = 0.6**(0.990**epoch_num)
                # weight_self = 0.2
                # loss = (1-weight)*loss1 + (weight)*loss2
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                pass
            elif args.model == 'sam2':
                dtype = None if args.precision == 'float32' else getattr(torch, args.precision) # 设置精度和上下文
                amp_context = nullcontext() if dtype is None else torch.autocast(device_type="cuda", dtype=dtype)
                scaler = torch.amp.GradScaler() if args.precision in ['float16', 'bfloat16'] else None    # # 只有float16需要梯度缩放

                with amp_context:  # torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)
                    # 对于不同的 prompt 导致的不同的预测结果, 计算出奖励, 然后使用 GRPO 进行优化: 奖励是由其他模型提供的, 那么怎么对本模型进行优化呢？
                    # 奖励的方式: 训练一个 prompt奖励回归器、基于 iou_prediction、基于预测结果跟 ground truth 的 iou、基于更大的模型
                    if args.is_grpo or args.is_dpo: # TODO: 在DPO中加入半监督的数据处理
                        loss, _ = train_with_po(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class,
                                                        beta=args.kl_beta, iteration=iter_num, writer=writer, args=args, **kwargs)
                    elif args.dev:
                        loss, _ = train_with_seg_batch(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class, 
                                                       beta=args.kl_beta, iteration=iter_num, writer=writer, args=args, **kwargs)
                    else:
                        loss, _ = vanilla_opt(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class, args=args, 
                                              only_train_unet=args.only_train_unet, **kwargs)

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            if args.model == 'hsam':
                writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
                writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
                writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
                writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)
            elif args.model == 'sam2' and args.is_grpo:
                pass # writer.add_scalar('info/avg_reward', avg_reward, iter_num)
            else: pass
            # writer.add_scalar('info/loss_self2', loss_self, iter_num)

            # tqdm.write('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item()))
            if args.model == 'hsam':
                tqdm.write('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f, loss_ce2: %f, loss_dice2: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item(), loss_ce2.item(), loss_dice2.item()))
            else:
                tqdm.write('iteration %d : loss : %f' % (iter_num, loss))
            
            if args.debug and i_batch == 0:
                break

        save_interval = args.interval_epoch # int(max_epoch/6)
        if args.debug or (epoch_num + 1) % save_interval == 0:
            infer_losses, infer_ious, infer_netsemi_iou, infer_netsemi_dice = [], [], [], []
            for test_case in db_test:
                # images, labels, name = test_case['image'], test_case['label'], test_case['case_name']   # (slices, 512, 512), (slices, 512, 512), string
                test_prompts = demo_usage(test_case, mode='all_nuclei', num_positive_points=2, num_negative_points=0, num_batches=1, all_instance_mode=True) # 'all_nuclei' or 'specific_category'
                if test_prompts is None: continue
                new_test_prompts = {'class_prompts': {1: test_prompts['prompts']},
                               'decoded_mask': {1: test_prompts['target_mask']}}
                kwargs.update({'test_prompts': test_prompts, 'new_test_prompts': new_test_prompts})

                images = test_prompts['image'] # (256, 256, 3)
                labels = test_prompts['target_mask'] # (256, 256)

                infer_loss, infer_iou = 0, 0
                infer_loss, infer_iou, net_semi_res = vanilla_eva(model,
                                                                  # torch.from_numpy(images.astype(np.float32)).unsqueeze(1).repeat(1, 3, 1, 1), # Synapse的处理方式
                                                                  torch.from_numpy(images)[None].permute(0, 3, 1, 2),
                                                                  torch.from_numpy(labels)[None], num_prompts_per_class=1, args=args, **kwargs)
                infer_losses.append(infer_loss), infer_ious.append(infer_iou)
                infer_netsemi_iou.append(net_semi_res[0]), infer_netsemi_dice.append(net_semi_res[1])

                if args.debug:
                    break
            
            writer.add_scalar('eva/infer_losses', np.mean(infer_losses), epoch_num)
            writer.add_scalar('eva/infer_ious', np.mean(infer_ious), epoch_num)

            save_path_suffix = '_lora' if args.ours_use_lora else ''
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}-loss_{np.mean(infer_losses):.04f}-iou_{np.mean(infer_ious):.04f}{save_path_suffix}.pth')

            if kwargs.get('net_semi', None):
                # torch.save(kwargs['net_semi'].state_dict(), f'unet_{epoch_num:02d}.pth')
                # infer_netsemi_iou是一个列表, 每一个元素是一个字典, 字典中按照类别存储iou指标的值, 现在需要根据这个列表计算出整体的iou值
                netsemi_iou_macro = np.mean([np.mean(list(iou_dict.values())) for iou_dict in infer_netsemi_iou])
                netsemi_dice_macro = np.mean([np.mean(list(dice_dict.values())) for dice_dict in infer_netsemi_dice])
                save_netsemi_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}-netsemi-iou_{np.mean(netsemi_iou_macro):.04f}.pth')

            if args.model == 'hsam':
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            elif args.model == 'sam2':
                if args.ours_use_lora:
                    args.utils['save_func'](model.model, save_mode_path)
                else:
                    try:
                        torch.save(model.model.state_dict(), save_mode_path)
                    except:
                        torch.save(model.model.module.state_dict(), save_mode_path)
                if kwargs.get('net_semi', None):
                    torch.save(kwargs['net_semi'].state_dict(), save_netsemi_path)
            tqdm.write("save model to {}".format(save_mode_path))

        if args.debug or epoch_num >= max_epoch - 1:
            save_path_suffix = '_lora' if args.ours_use_lora else ''
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}{save_path_suffix}.pth')
            if args.model == 'hsam':
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            elif args.model == 'sam2':
                if args.ours_use_lora:
                    args.utils['save_func'](model.model, save_mode_path)
                else:
                    try:
                        torch.save(model.model.state_dict(), save_mode_path)
                    except:
                        torch.save(model.model.module.state_dict(), save_mode_path)
                if kwargs.get('net_semi', None):
                    torch.save(kwargs['net_semi'].state_dict(), save_netsemi_path)
            tqdm.write("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_pannuke_batch(args, model, snapshot_path, multimask_output, low_res):
    from dataset.dataset_synapse import Synapse_dataset, RandomGenerator
    from opt_utils_batch import train_with_po_batched
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=tqdm.write,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    tqdm.write(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    # dataset = load_dataset("/database/wuyonghuang/PanNuke/")
    # db_train = dataset['fold1']
    # db_test = dataset['fold2']

    # from load_pannuke_v3 import split_train_val_test
    # split_train_val_test(dataset=dataset, val_ratio=0.1, test_ratio=0.1)

    from datasets import load_from_disk
    dataset = load_from_disk("./data/split_pannuke")
    db_train = dataset['train']
    db_val = dataset['validation']
    db_test = dataset['test']

    # result1 = demo_usage(sample, mode='all_nuclei') # 'all_nuclei' or 'specific_category'

    print("The length of train set is: {}".format(len(db_train)))

    db_train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: batch)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.semi:
        raise ValueError("Semi-supervised learning is not supported for pannuke dataset.")
    else:
        # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
        #                          worker_init_fn=worker_init_fn)
        # trainloader_un = None
        trainloader = None

    if args.model == 'hsam':
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
            # model = model.module
        model.train()
        
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

        ce_loss = CrossEntropyLoss()
        cos_loss = CosineEmbeddingLoss()
        l1_loss = nn.L1Loss()
        dice_loss = DiceLoss(num_classes + 1)
        # dice_loss = Focal_loss(num_classes=num_classes + 1)
        if args.warmup:
            b_lr = base_lr / args.warmup_period
        else:
            b_lr = base_lr
        if args.AdamW:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    
    elif args.model == 'sam2':
        if args.n_gpu > 1:
            model.model = nn.DataParallel(model.model)
            # model = model.module
        model.model.train()
        model.model.cuda()
        
        model_total_params = sum(p.numel() for p in model.model.parameters())
        model_grad_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(num_classes + 1)
        # dice_loss = Focal_loss(num_classes=num_classes + 1)
        if args.warmup:
            b_lr = base_lr / args.warmup_period
        else:
            b_lr = base_lr
        if args.AdamW:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-5)
            # optimizer = optim.AdamW([list(model.model.parameters())[-3]], lr=b_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-5)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.model.parameters()), lr=b_lr, momentum=0.9, weight_decay=args.weight_decay*0.001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
        
        if (args.semi and args.use_unet) or args.only_train_unet:   # 两种情况: ①半监督下要是否使用unet; 2 是否只训练unet
            import segmentation_models_pytorch as smp
            net_semi = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=9) # note: 这个是小模型, 用来支持半监督或者其他算法
            optimizer_semi = optim.AdamW(filter(lambda p: p.requires_grad, net_semi.parameters()), lr=1e-4)

            net_semi.train()
            net_semi.cuda()
        elif args.semi and not args.use_unet:
            raise ValueError
        else:
            net_semi = None
            optimizer_semi = None

    else: raise ValueError

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(db_train)  # max_epoch = max_iterations // len(trainloader) + 1
    num_prompts_per_class = args.num_prompts_per_class
    tqdm.write("{} iterations per epoch. {} max iterations ".format(len(db_train), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kwargs = {}

    def collect_batch_fn(samples_batch, mode='all_nuclei', kwargs1={}, kwargs2={}, return_list=[]):
        for sample in samples_batch:
            prompts = demo_usage(sample, **kwargs1) # 'all_nuclei' or 'specific_category'
            prompts2 = demo_usage(sample, **kwargs2) # 'all_nuclei' or 'specific_category'
            if prompts is None or prompts2 is None: continue
            new_prompts = {'class_prompts':
                        {1: prompts['prompts']},
                            'decoded_mask': {1: prompts['target_mask']}}
            new_prompts2 = {'class_prompts':
                        {1: prompts2['prompts']},
                            'decoded_mask': {1: prompts2['target_mask']}}
            return_list.append({'prompts': prompts, 'new_prompts': new_prompts, 
                        'prompts2': prompts2, 'new_prompts2': new_prompts2})
        return return_list
    
    for epoch_num in iterator:
        for i_batch, samples_batch in tqdm(enumerate(db_train_loader)):
            # point 的数量以及分布; prompts2(教师用的prompt应该覆盖所有的实例)
            if isinstance(args.pos_point_num, tuple):
                num_pos_points = np.random.randint(*args.pos_point_num, size=(1))[0]
            else: num_pos_points = args.pos_point_num
            if isinstance(args.neg_point_num, tuple):
                num_neg_points = np.random.randint(*args.neg_point_num, size=(1))[0]
            else: num_neg_points = args.neg_point_num

            # image_batch = prompts['image']   # numpy: (256, 256, 3)
            # label_batch = prompts['target_mask'] # numpy: (256, 256)
            # image_batch, label_batch = torch.from_numpy(image_batch).permute(2, 0, 1)[None].cuda(), torch.from_numpy(label_batch)[None].cuda()

            collect_batch = collect_batch_fn(samples_batch, return_list=[],
                          kwargs1={'mode': 'all_nuclei', 'num_positive_points': num_pos_points, 'num_negative_points': num_neg_points, 'num_batches': num_prompts_per_class, 'all_instance_mode': True},
                          kwargs2={'mode': 'all_nuclei', 'num_positive_points': 100, 'num_negative_points': 100, 'num_batches': num_prompts_per_class, 'all_instance_mode': True})
            if len(collect_batch) < 1: continue

            # low_res_label_batch = sampled_batch['low_res_label']
            # low_res_label_batch = low_res_label_batch.cuda()
            low_res_label_batch = None
            # assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            # 加载半监督未标注数据, 暂停开发

            if args.model == 'hsam':
                # outputs1,outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=low_res_label_batch)
                # loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
                # loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
                # weight = 0.6**(0.990**epoch_num)
                # weight_self = 0.2
                # loss = (1-weight)*loss1 + (weight)*loss2
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                pass
            elif args.model == 'sam2':
                dtype = None if args.precision == 'float32' else getattr(torch, args.precision) # 设置精度和上下文
                amp_context = nullcontext() if dtype is None else torch.autocast(device_type="cuda", dtype=dtype)
                scaler = torch.amp.GradScaler() if args.precision in ['float16', 'bfloat16'] else None    # # 只有float16需要梯度缩放

                with amp_context:  # torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)
                    # 对于不同的 prompt 导致的不同的预测结果, 计算出奖励, 然后使用 GRPO 进行优化: 奖励是由其他模型提供的, 那么怎么对本模型进行优化呢？
                    # 奖励的方式: 训练一个 prompt奖励回归器、基于 iou_prediction、基于预测结果跟 ground truth 的 iou、基于更大的模型
                    if args.is_grpo or args.is_dpo: # Todo: 在DPO中加入半监督的数据处理
                        loss, _ = train_with_po_batched(model, optimizer, scaler, collect_batch, num_prompts_per_class=num_prompts_per_class,
                                                        beta=args.kl_beta, iteration=iter_num, writer=writer, args=args, **kwargs)
                    elif args.dev:
                        loss, _ = None  # train_with_seg_batch(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class, 
                                           #              beta=args.kl_beta, iteration=iter_num, writer=writer, args=args, **kwargs)
                    else:
                        loss, _ = None  # vanilla_opt(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=num_prompts_per_class, args=args, 
                                          #     only_train_unet=args.only_train_unet, **kwargs)

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            if args.model == 'hsam':
                writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
                writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
                writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
                writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)
            elif args.model == 'sam2' and args.is_grpo:
                pass # writer.add_scalar('info/avg_reward', avg_reward, iter_num)
            else: pass
            # writer.add_scalar('info/loss_self2', loss_self, iter_num)

            # tqdm.write('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item()))
            if args.model == 'hsam':
                tqdm.write('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f, loss_ce2: %f, loss_dice2: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item(), loss_ce2.item(), loss_dice2.item()))
            else:
                tqdm.write('iteration %d : loss : %f' % (iter_num, loss))
            
            if args.debug and i_batch == 0:
                break

        save_interval = args.interval_epoch # int(max_epoch/6)
        if args.debug or (epoch_num + 1) % save_interval == 0:
            infer_losses, infer_ious, infer_netsemi_iou, infer_netsemi_dice = [], [], [], []
            for test_case in db_test:
                # images, labels, name = test_case['image'], test_case['label'], test_case['case_name']   # (slices, 512, 512), (slices, 512, 512), string
                test_prompts = demo_usage(test_case, mode='all_nuclei', num_positive_points=2, num_negative_points=0, num_batches=1, all_instance_mode=True) # 'all_nuclei' or 'specific_category'
                if test_prompts is None: continue
                new_test_prompts = {'class_prompts': {1: test_prompts['prompts']},
                               'decoded_mask': {1: test_prompts['target_mask']}}
                kwargs.update({'test_prompts': test_prompts, 'new_test_prompts': new_test_prompts})

                images = test_prompts['image'] # (256, 256, 3)
                labels = test_prompts['target_mask'] # (256, 256)

                infer_loss, infer_iou = 0, 0
                infer_loss, infer_iou, net_semi_res = vanilla_eva(model,
                                                                  # torch.from_numpy(images.astype(np.float32)).unsqueeze(1).repeat(1, 3, 1, 1), # Synapse的处理方式
                                                                  torch.from_numpy(images)[None].permute(0, 3, 1, 2),
                                                                  torch.from_numpy(labels)[None], num_prompts_per_class=1, args=args, **kwargs)
                infer_losses.append(infer_loss), infer_ious.append(infer_iou)
                infer_netsemi_iou.append(net_semi_res[0]), infer_netsemi_dice.append(net_semi_res[1])

                if args.debug:
                    break
            
            writer.add_scalar('eva/infer_losses', np.mean(infer_losses), epoch_num)
            writer.add_scalar('eva/infer_ious', np.mean(infer_ious), epoch_num)

            save_path_suffix = '_lora' if args.ours_use_lora else ''
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}-loss_{np.mean(infer_losses):.04f}-iou_{np.mean(infer_ious):.04f}{save_path_suffix}.pth')

            if kwargs.get('net_semi', None):
                # torch.save(kwargs['net_semi'].state_dict(), f'unet_{epoch_num:02d}.pth')
                # infer_netsemi_iou是一个列表, 每一个元素是一个字典, 字典中按照类别存储iou指标的值, 现在需要根据这个列表计算出整体的iou值
                netsemi_iou_macro = np.mean([np.mean(list(iou_dict.values())) for iou_dict in infer_netsemi_iou])
                netsemi_dice_macro = np.mean([np.mean(list(dice_dict.values())) for dice_dict in infer_netsemi_dice])
                save_netsemi_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}-netsemi-iou_{np.mean(netsemi_iou_macro):.04f}.pth')

            if args.model == 'hsam':
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            elif args.model == 'sam2':
                if args.ours_use_lora:
                    args.utils['save_func'](model.model, save_mode_path)
                else:
                    try:
                        torch.save(model.model.state_dict(), save_mode_path)
                    except:
                        torch.save(model.model.module.state_dict(), save_mode_path)
                if kwargs.get('net_semi', None):
                    torch.save(kwargs['net_semi'].state_dict(), save_netsemi_path)
            tqdm.write("save model to {}".format(save_mode_path))

        if args.debug or epoch_num >= max_epoch - 1:
            save_path_suffix = '_lora' if args.ours_use_lora else ''
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num:03d}{save_path_suffix}.pth')
            if args.model == 'hsam':
                try:
                    model.save_lora_parameters(save_mode_path)
                except:
                    model.module.save_lora_parameters(save_mode_path)
            elif args.model == 'sam2':
                if args.ours_use_lora:
                    args.utils['save_func'](model.model, save_mode_path)
                else:
                    try:
                        torch.save(model.model.state_dict(), save_mode_path)
                    except:
                        torch.save(model.model.module.state_dict(), save_mode_path)
                if kwargs.get('net_semi', None):
                    torch.save(kwargs['net_semi'].state_dict(), save_netsemi_path)
            tqdm.write("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"