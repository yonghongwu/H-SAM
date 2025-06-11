import os, copy
import cv2
import torch 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from einops import rearrange
from collections import defaultdict
from utils import get_decoded_mask
from gen_prompt import generate_prompts_from_semantic_mask, get_prompt_preds, concatenate_masks_and_scores_v2, plot_results


def vanilla_opt(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=3, args=None, only_train_unet=False, **kwargs):
    batch_losses = []

    if only_train_unet:
        assert len(kwargs) != 0

    for image_idx, label_idx in zip(range(image_batch.shape[0]), range(label_batch.shape[0])):
        if not only_train_unet:
            image, label = image_batch[image_idx].permute(1, 2, 0).cpu().numpy(), label_batch[label_idx].cpu().numpy()   # low_res_label_batch[0].cpu().numpy()

            if label.mean() == 0: print("无前景; Skip"); continue

            # 获取 prompts、二值化的多类别掩码
            prompts = generate_prompts_from_semantic_mask(
                label,
                class_ids=None,  # 处理所有类别
                num_positive_points=(2, 3),
                num_negative_points=(4, 6),
                num_prompts_per_class=num_prompts_per_class,  # 每个类别生成3组prompt
                point_sampling_strategy="center_weighted",
                box_noise_level=0.1,
                generate_box=True,
                generate_points=True
            )

            decoded_mask = prompts['decoded_mask']
            # 按照decoded_mask的键的顺序进行排序, 然后将值重复三次再全部拼接起来。
            exist_keys = sorted(decoded_mask.keys())
            for key in exist_keys:
                # decoded_mask[key] = np.repeat(decoded_mask[key], 3, axis=0)
                decoded_mask[key] = torch.from_numpy(decoded_mask[key])[None].long().repeat(num_prompts_per_class, 1, 1)
            decoded_mask = torch.concat(list(decoded_mask.values()), dim=0)

            # 函数: 接受 image、point、model, 输出 prediction
            model.set_image(image)
            results = get_prompt_preds(model, prompts, prompt_mode=args.prompt_type, multimask_output=True, only_best_score_pred=True, only_save_best_prompt_pred=False)
            all_logits, all_scores, category_indices = concatenate_masks_and_scores_v2(results['prompts_preds'], sort_keys=True)

            criterion = nn.BCEWithLogitsLoss()
            adaptive_pool = nn.AdaptiveAvgPool2d(decoded_mask.shape[-2:])

            sample_loss = criterion(adaptive_pool(all_logits), decoded_mask.float().cuda())
            batch_losses.append(sample_loss)

            # 展示结果
            # plot_results(results, image, label, plot_prompts_preds=True, save_path_lst=None)

            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad()

            # 根据是否使用scaler选择不同的反向传播和优化步骤
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            return loss.item(), None
        
        else:
            # 这里就是直接用unet进行训练
            image = image_batch[image_idx:image_idx+1]
            label = label_batch[label_idx:label_idx+1]
            pred = kwargs['net_semi'](image[:, 0:1])  # 只使用2D slice的第一个通道(其实三个通道的信息是一样的, 因为这是医学数据); shape: (B, cls, H, W)
            loss = F.cross_entropy(pred, label.long().cuda())
            batch_losses.append(loss)

            loss = torch.stack(batch_losses).mean()
            kwargs['optimizer_semi'].zero_grad()

            # 根据是否使用scaler选择不同的反向传播和优化步骤
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(kwargs['optimizer_semi'])
                scaler.update()
            else:
                loss.backward()
                kwargs['optimizer_semi'].step()
            return loss.item(), None


@torch.no_grad()
def vanilla_eva(model, image_batch, label_batch, num_prompts_per_class=3, args=None, **kwargs):
    device = image_batch.device
    batch_losses, batch_ious = [], []
    all_iou_scores_semi = {} # Stores list of IoUs for each class label
    all_dice_scores_semi = {} # Stores list of Dice scores for each class label

    for image_idx, label_idx in tqdm(zip(range(image_batch.shape[0]), range(label_batch.shape[0]))):
        
        if kwargs.get('net_semi', None):    # 传入了 unet 模型或者其他 net_semi 模型, 则说明要对unet进行评估
            image = image_batch[image_idx:image_idx+1][:, 0:1].cuda()
            label = label_batch[label_idx:label_idx+1].cuda()
            # label_name = list(set(torch.unique(label).cpu().numpy())-set([0]))  # 只关注前景

            current_label_numpy = label[0].cpu().numpy().astype(np.int64)
            present_foreground_labels = sorted(list(set(range(args.num_classes+1)) - set([0])))

            with torch.no_grad():
                pred = kwargs['net_semi'](image)  # 只使用2D slice的第一个通道(其实三个通道的信息是一样的, 因为这是医学数据); shape: (B, cls, H, W)
            pred_classes_numpy = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.int64)
            
            label_flat = current_label_numpy.flatten()
            pred_flat = pred_classes_numpy.flatten()

            from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
            iou_per_class = jaccard_score(label_flat, pred_flat, labels=present_foreground_labels, average=None, zero_division=0)
            dice_per_class = f1_score(label_flat, pred_flat, labels=present_foreground_labels, average=None, zero_division=0)

            for i, class_label in enumerate(present_foreground_labels):
                if class_label not in all_iou_scores_semi:
                    all_iou_scores_semi[class_label] = []
                    all_dice_scores_semi[class_label] = []
                all_iou_scores_semi[class_label].append(iou_per_class[i])
                all_dice_scores_semi[class_label].append(dice_per_class[i])

            # plt.imshow(torch.argmax(pred, dim=1)[0].cpu().numpy()); plt.savefig(f'test-idx_{label_idx}-pred.jpg')
            # plt.imshow(label[0].cpu().numpy()); plt.savefig(f'test-idx_{label_idx}-lab.jpg')
        
        if args.only_train_unet:
            continue

        image, label = image_batch[image_idx].permute(1, 2, 0).cpu().numpy(), label_batch[label_idx].cpu().numpy()   # low_res_label_batch[0].cpu().numpy()

        if label.mean() == 0: continue

        # 获取 prompts、二值化的多类别掩码
        if args.dataset  == 'Synapse':
            prompts = generate_prompts_from_semantic_mask(
                label,
                class_ids=None,  # 处理所有类别
                num_positive_points=1,
                num_negative_points=0,
                num_prompts_per_class=num_prompts_per_class,  # 每个类别生成3组prompt
                point_sampling_strategy="center_weighted",
                box_noise_level=0.1,
                generate_box=True,
                generate_points=True
            )
            decoded_mask = get_decoded_mask(prompts['decoded_mask'], num_prompts_per_class=num_prompts_per_class)

        elif args.dataset == 'PanNuke':
            prompts = kwargs['new_test_prompts']
            decoded_mask = torch.from_numpy(prompts['decoded_mask'][1])[None].to(device)
        
        else: raise ValueError

        # 函数: 接受 image、point、model, 输出 prediction
        model.set_image(image)
        results = get_prompt_preds(model, prompts, prompt_mode=args.prompt_type, multimask_output=True, only_best_score_pred=args.onlybest_in_multimask_output, only_save_best_prompt_pred=False)
        all_logits, all_scores, category_indices = concatenate_masks_and_scores_v2(results['prompts_preds'], sort_keys=True)

        criterion = nn.BCEWithLogitsLoss()
        adaptive_pool = nn.AdaptiveAvgPool2d(decoded_mask.shape[-2:])

        sample_loss = criterion(adaptive_pool(all_logits), decoded_mask.float().cuda())
        batch_losses.append(sample_loss)

        sample_iou = compute_iou(adaptive_pool(all_logits), decoded_mask.float().cuda())
        batch_ious.append(sample_iou)

        # 展示结果
        # plot_results(results, image, label, plot_prompts_preds=True, save_path_lst=None)
        coupled_mask = np.zeros_like(label, dtype=np.uint8) # size: (H, W)
        for idx, i in enumerate(sorted(set(np.unique(label).astype(np.uint8)) - set([0]))):
            coupled_mask += (adaptive_pool(all_logits) > 0).cpu().numpy().astype(np.uint8)[idx] * i

    mean_iou_per_class_net_semi = {cls: np.mean(scores) for cls, scores in all_iou_scores_semi.items()}
    mean_dice_per_class_net_semi = {cls: np.mean(scores) for cls, scores in all_dice_scores_semi.items()}

    if args.only_train_unet:
        return 0, 0, (mean_iou_per_class_net_semi, mean_dice_per_class_net_semi)
    
    loss = torch.stack(batch_losses).mean()
    iou = np.stack(batch_ious).mean()

    return loss.item(), iou, (mean_iou_per_class_net_semi, mean_dice_per_class_net_semi)


# 1. 首先创建参考模型（冻结的SAM副本）
def create_reference_model(model):
    ref_model = copy.deepcopy(model)
    for param in ref_model.model.parameters():
        param.requires_grad = False
    ref_model.model.eval()
    return ref_model


# 每3个元素为一组进行标准化
def normalize_in_chunks(x, chunk_size, temperature=1.0):
    # 复制结果张量
    result = x.clone()
    
    # 遍历每个chunk
    for i in range(0, len(x), chunk_size):
        chunk = x[i:i+chunk_size]
        mean = chunk.mean()
        std = chunk.std(unbiased=False)  # 使用无偏估计
        result[i:i+chunk_size] = (chunk - mean) / (std + 1e-8) / temperature

    return result

def normalize_in_chunks_v2(x, chunk_size, temperature=1.0):
    # Vectorized Example
    N = x.shape[0]
    num_chunks = N // chunk_size
    if N % chunk_size != 0:
        # Handle cases where length is not a multiple of chunk_size if necessary
        print("Warning: length of x is not a multiple of chunk_size")

    # Reshape into chunks
    x_reshaped = x.view(num_chunks, chunk_size) # Or -1 instead of num_chunks

    # Calculate mean and std per chunk (along dim=1)
    mean = x_reshaped.mean(dim=1, keepdim=True)
    std = x_reshaped.std(dim=1, keepdim=True, unbiased=False) # Or unbiased=True

    # Normalize and apply temperature
    normalized_x = (x_reshaped - mean) / (std + 1e-8) / temperature

    # Reshape back to original shape
    result = normalized_x.view(-1) 
    return result


# 2. 实现奖励函数计算
def compute_segmentation_rewards(pred_masks, gt_masks, images):
    """计算分割质量奖励
        # rewards = compute_segmentation_rewards(resized_masks, decoded_mask, torch.from_numpy(image).permute(2, 0, 1).cuda()\
        #                                         if len(image.shape) == 3 else torch.from_numpy(image).cuda())
    """
    rewards = []
    
    for pred, gt, img in zip(pred_masks, gt_masks, images):
        # 1. IoU奖励 - 基础分割质量度量
        iou_score = compute_iou(pred, gt)
        
        # 2. 边缘平滑度奖励 - 使用索贝尔算子检测边缘并与图像梯度比较
        edge_smoothness = compute_edge_smoothness(pred, img)
        
        # 3. 区域连通性奖励 - 鼓励生成连贯的分割区域
        connectivity_score = compute_connectivity_score(pred)
        
        # 综合奖励计算
        reward = (0.6 * iou_score + 0.2 * edge_smoothness + 0.2 * connectivity_score)
        rewards.append(reward)
    
    # 转换为tensor并归一化
    rewards = torch.tensor(rewards, device=pred_masks.device)
    if len(rewards) > 1:  # 只有在batch size > 1时才归一化
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    return rewards


def compute_iou(pred, gt):
    """计算IoU分数"""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_binary * gt).sum()
    union = pred_binary.sum() + gt.sum() - intersection
    return (intersection / (union + 1e-8)).item()


def compute_edge_smoothness(pred, img):
    """计算边缘平滑度分数"""
    # 简化实现 - 实际应用中可使用更复杂的边缘检测
    pred_sigmoid = torch.sigmoid(pred)
    pred_edges = F.max_pool2d(pred_sigmoid, 3, stride=1, padding=1) - F.avg_pool2d(pred_sigmoid, 3, stride=1, padding=1)
    
    # 计算图像梯度
    img_gray = img.mean(dim=0) if img.dim() == 3 else img
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device)
    
    img_grad_x = F.conv2d(img_gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    img_grad_y = F.conv2d(img_gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    img_grad = torch.sqrt(img_grad_x**2 + img_grad_y**2).squeeze()
    
    # 边缘与图像梯度的一致性
    alignment = (pred_edges.squeeze() * img_grad).sum() / (pred_edges.sum() + 1e-8)
    return alignment.item()


def compute_connectivity_score(pred):
    """计算区域连通性分数"""
    # 使用简化的连通性度量 - 实际应用中可使用scipy的label函数
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    
    # 使用池化操作近似评估连通性
    dilated = F.max_pool2d(pred_binary, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-pred_binary, 3, stride=1, padding=1)
    
    # 连通区域应该在膨胀和腐蚀后保持相似
    connectivity = (eroded.sum() / (dilated.sum() + 1e-8)).item()
    return connectivity


# 3. 实现获取模型预测概率的函数
def get_mask_log_probs(model, image, prompts, pred_masks=None):
    """获取模型对掩码的预测对数概率"""
    # 已设置图像的情况下, 直接使用prompts获取logits
    results = get_prompt_preds(
        model, prompts, prompt_mode='point', 
        multimask_output=True, 
        only_best_score_pred=True, 
        only_save_best_prompt_pred=False,
        # 确保返回logits而不仅是最终掩码
    )
    
    # 提取logits
    # all_logits = results.get('logits', None)
    all_logits, all_scores, category_indices = concatenate_masks_and_scores_v2(results['prompts_preds'], sort_keys=True)
    if all_logits is None:
        raise ValueError("模型预测未返回logits, 请确保get_prompt_preds函数支持return_logits参数")
    
    # 如果提供了pred_masks, 计算这些掩码的对数概率
    if pred_masks is not None:
        batch_log_probs = []
        for logits, mask in zip(all_logits, pred_masks):
            # 将logits转换为对数概率
            log_probs = F.logsigmoid(logits)  # 正类的对数概率
            neg_log_probs = F.logsigmoid(-logits)  # 负类的对数概率
            
            # 根据mask选择相应的对数概率
            mask_binary = (mask > 0.5).float()
            pixel_log_probs = mask_binary * log_probs + (1 - mask_binary) * neg_log_probs
            
            # 计算整个掩码的平均对数概率
            mask_log_prob = pixel_log_probs.mean()
            batch_log_probs.append(mask_log_prob)
        
        return torch.stack(batch_log_probs)
    
    return all_logits


def continuous_to_dispersed(tensor_2d):
    """
    处理二维张量, 将每行最大值置 1, 最小值置 -1, 其余置 0.

    Args:
        tensor_2d (np.ndarray): 一个二维 NumPy 数组.

    Returns:
        np.ndarray: 处理后的二维 NumPy 数组.
    """
    processed_tensor = torch.zeros_like(tensor_2d)  # 初始化一个与输入张量形状相同的零张量

    for i in range(tensor_2d.shape[0]): # 遍历每一行
        row = tensor_2d[i]
        max_index = torch.argmax(row)      # 找到最大值索引
        min_index = torch.argmin(row)      # 找到最小值索引

        processed_tensor[i, max_index] = 1   # 将最大值位置置为 1
        # processed_tensor[i, min_index] = -1  # 将最小值位置置为 -1

    return processed_tensor


def continuous_to_dispersed_v2(policy_values, scheme=1):
    """
    计算不同方案下的策略奖励值
    
    参数:
        policy_values: 形状为(N, L)的张量, N是数据量, L是策略数量
        scheme: 使用的方案, 1表示第一个方案, 2表示第二个方案
        
    返回:
        rewards: 形状为(N, L)的张量, 表示每个策略的奖励值
    """
    # 确保输入是张量
    if not isinstance(policy_values, torch.Tensor):
        policy_values = torch.tensor(policy_values, dtype=torch.float32)
    
    # 创建与输入相同形状的奖励张量
    rewards = torch.zeros_like(policy_values)
    
    if scheme == 1:
        # 第一个方案：基于固定阈值的奖励计算
        # 当策略值大于0.85, 则奖励为1
        rewards[policy_values > 0.85] = 1.0
        # 当小于0.85而大于0.5时, 奖励为0.5
        rewards[(policy_values <= 0.85) & (policy_values > 0.5)] = 0.5
        # 当奖励小于0.5而大于0.2时, 奖励为0（默认已经为0, 不需要额外设置）
        rewards[(policy_values <= 0.2)] = -1
    
    elif scheme == 2:
        # 第二个方案：基于每条数据中策略最小值的相对差值
        N, L = policy_values.shape
        
        # 对每条数据单独处理
        for n in range(N):
            # 获取当前数据的所有策略值
            current_policies = policy_values[n]
            # 找到最小策略值
            min_value = torch.min(current_policies)
            # 计算每个策略值与最小值的差值
            differences = current_policies - min_value
            
            # 根据差值计算奖励
            # 差值大于0.2, 奖励为2
            rewards[n][differences > 0.2] = 2.0
            # 差值小于0.2但大于0.02, 奖励为1
            rewards[n][(differences <= 0.2) & (differences > 0.02)] = 1.0
            # 差值小于0.02, 奖励为0（默认已经为0, 不需要额外设置）
            # 最小值对应的位置, 奖励恒为0（由于差值为0, 所以默认已经为0）
    
    else:
        raise ValueError("方案参数必须为1或2")
    
    return rewards


# 4. 实现GRPO损失函数
def grpo_loss(current_logits, ref_logits, pred_masks, gt_masks, rewards, beta=0.05, clip_param=0.2, gen_log_probs=None):
    """计算GRPO损失"""
    # 计算当前模型的对数概率
    current_log_probs = F.logsigmoid(current_logits)
    neg_current_log_probs = F.logsigmoid(-current_logits)
    
    # 计算参考模型的对数概率
    ref_log_probs = F.logsigmoid(ref_logits)
    neg_ref_log_probs = F.logsigmoid(-ref_logits)
    
    # 获取掩码二值化版本
    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
    
    # 为每个像素选择正确的对数概率
    current_pixel_log_probs = pred_binary * current_log_probs + (1 - pred_binary) * neg_current_log_probs
    ref_pixel_log_probs = pred_binary * ref_log_probs + (1 - pred_binary) * neg_ref_log_probs
    
    # 计算KL散度 (像素级别)
    per_pixel_kl = torch.exp(ref_pixel_log_probs - current_pixel_log_probs) - (ref_pixel_log_probs - current_pixel_log_probs) - 1
    
    # 准备优势函数 (扩展维度以匹配像素数)
    advantages = rewards.view(-1, 1, 1, 1).expand_as(current_logits)
    
    # 如果有生成时的对数概率, 使用PPO的比率裁剪
    if gen_log_probs is not None:
        gen_pixel_log_probs = pred_binary * F.logsigmoid(gen_log_probs) + (1 - pred_binary) * F.logsigmoid(-gen_log_probs)
        ratio = torch.exp(current_pixel_log_probs - gen_pixel_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_pixel_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # 简化版本, 直接使用当前对数概率
        per_pixel_loss = -current_pixel_log_probs * advantages
    
    # 添加KL惩罚项
    per_pixel_loss = per_pixel_loss + beta * per_pixel_kl
    
    # 计算有效像素的掩码 (忽略填充区域)
    valid_mask = (gt_masks >= 0).float()  # 假设-1表示填充区域
    
    # 应用掩码并计算平均损失
    masked_loss = (per_pixel_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    
    return masked_loss


# 5. 主要训练循环修改
def train_with_po(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=3, beta=0.05, clip_param=0.2, iteration=0, writer=None, args=False, **kwargs):
    # 创建参考模型 (如果尚未创建)
    if not hasattr(train_with_po, 'ref_model'):
        train_with_po.ref_model = create_reference_model(model)
    
    ref_model = train_with_po.ref_model
    
    batch_losses = []
    all_rewards = []
    idx_count = 0

    if len(kwargs) != 0 and kwargs.get('semi', None):
        assert args.semi == True
        image_batch_un = kwargs['image_un']
        label_batch_un = kwargs['label_un']
    
    for idx, (image_idx, label_idx) in enumerate(zip(range(image_batch.shape[0]), range(label_batch.shape[0]))):    # 假设 有标注数据和未标注数据的比例是 1:1
        device = image_batch.device
        image_idx_in_all_trainset = iteration * image_batch.shape[0] + idx_count
        image, label = image_batch[image_idx].permute(1, 2, 0).cpu().numpy(), label_batch[label_idx].cpu().numpy()
        if label.sum()  == 0: continue
        
        idx_count += 1

        if isinstance(args.pos_point_num, tuple):
            num_pos_points = np.random.randint(*args.pos_point_num, size=(1))[0]
        else: num_pos_points = args.pos_point_num
        if isinstance(args.neg_point_num, tuple):
            num_neg_points = np.random.randint(*args.neg_point_num, size=(1))[0]
        else: num_neg_points = args.neg_point_num

        if args.semi:   # note: 设置在一定的iteration之后再进行半监督,提供一定的稳定性
            image_un_npy = image_batch_un[image_idx].permute(1, 2, 0).cpu().numpy()
            image_un = image_batch_un[image_idx:image_idx+1]
            pred_un = kwargs['net_semi'](image_un[:, 0:1])  # 只使用2D slice的第一个通道(其实三个通道的信息是一样的, 因为这是医学数据); shape: (B, cls, H, W)
            # note: 半监督算法在视觉上的常见算法: MT\Fixmatch, 这里要新的半监督算法: 基于DPO来实现半监督, 可以使用正则化来提供额外的监督
            # 1. 处理 pred_un 成二分类的集合, 找到每个类别的最大连通区域作为mask; 2. 生成point 和 box prompt; 3. 分别给 actor 和 ref_model 使用, 对于 DPO, ref_model 为最优
            from semi_utils import process_segmentation_output
            mask_un = process_segmentation_output(output_tensor=pred_un)

            pseudo_prompts = generate_prompts_from_semantic_mask(
                mask_un,
                class_ids=None,
                num_positive_points=num_pos_points,
                num_negative_points=num_neg_points,
                num_prompts_per_class=num_prompts_per_class,
                point_sampling_strategy="center_weighted",
                box_noise_level=0.1,
                generate_box=True,
                generate_points=True,
                is_strict=args.is_strict
            )

            # 1. 将 prompt 输入给 actor 以利用 未标注数据
            model.set_image(image_un_npy)
            pseudo_results = get_prompt_preds(
                model, pseudo_prompts, prompt_mode='box',     # TODO: 这里的低质量 掩码应该生成 point 还是 box prompt ? 先使用 box prompt
                multimask_output=True, 
                only_best_score_pred=args.onlybest_in_multimask_output, 
                only_save_best_prompt_pred=False,
            )
            
            pseudo_logits, pseudo_scores, pseudo_category_indices = concatenate_masks_and_scores_v2(
                pseudo_results['prompts_preds'], sort_keys=True
            )

            # 2. TODO: actor 输出分割结果作为 GT 来监督 unet 模型; unet 用 mask_un 监督 actor 模型
            pseudo_logits = rearrange(pseudo_logits, '(b n) h w -> b n h w', n=args.num_prompts_per_class)
            pseudo_logprobs = F.logsigmoid(pseudo_logits)

            # policy_chosen_logps = torch.gather(policy_logprobs, dim=1, index=chosen_indices) # Shape: (mbatch_size, 1, height, width)
            # criterion(policy_logprobs.squeeze(1), gt_masks.float().cuda()[::args.num_prompts_per_class]) 
        
        if args.dataset == 'Synapse':
            # 获取prompts和GT掩码 (与原代码相同)
            prompts = generate_prompts_from_semantic_mask(
                label,
                class_ids=None,
                num_positive_points=num_pos_points,
                num_negative_points=num_neg_points,
                num_prompts_per_class=num_prompts_per_class,
                point_sampling_strategy="center_weighted",
                box_noise_level=0.1,
                generate_box=True,
                generate_points=True,
                is_strict=args.is_strict
            )   # {class_prompts: {cls1: [ {box_prompt: [x1, x2, y1, y2], point_prompts: { (x1, y1, pos), (x2, y2, neg), (...), ... } } , ..., ..., ] } , ...}
        
            decoded_mask = prompts['decoded_mask']
            class_keys = sorted(decoded_mask.keys())
            for key in class_keys:
                decoded_mask[key] = torch.from_numpy(decoded_mask[key])[None].long().repeat(num_prompts_per_class, 1, 1)
            decoded_mask = torch.concat(list(decoded_mask.values()), dim=0).cuda()
        elif args.dataset == 'PanNuke':
            prompts, prompts2 = kwargs['new_prompts'], kwargs['new_prompts2']
            decoded_mask = torch.from_numpy(prompts['decoded_mask'][1])[None].to(device)
        
        # 设置图像
        model.set_image(image)
        ref_model.set_image(image)
        
        # 1. 获取当前模型预测
        with torch.enable_grad():  # 确保启用梯度
            results = get_prompt_preds(
                model, prompts, prompt_mode=args.prompt_type,     
                multimask_output=True, 
                only_best_score_pred=args.onlybest_in_multimask_output, 
                only_save_best_prompt_pred=False,
            )
            
            current_logits, current_scores, current_category_indices = concatenate_masks_and_scores_v2(
                results['prompts_preds'], sort_keys=True
            )
        
        # 2. 获取参考模型预测 (无梯度)
        with torch.no_grad():
            if args.kl_pos_point_num is not False and args.kl_neg_point_num is not False:
                if isinstance(args.kl_pos_point_num, tuple):
                    kl_num_pos_points = np.random.randint(*args.kl_pos_point_num, size=(1))[0]
                else: kl_num_pos_points = args.kl_pos_point_num
                if isinstance(args.kl_neg_point_num, tuple):
                    kl_num_neg_points = np.random.randint(*args.kl_neg_point_num, size=(1))[0]
                else: kl_num_neg_points = args.kl_neg_point_num

                if args.dataset == 'Synapse':
                    prompts = generate_prompts_from_semantic_mask(
                        label,
                        class_ids=None,
                        num_positive_points=kl_num_pos_points,
                        num_negative_points=kl_num_neg_points,
                        num_prompts_per_class=num_prompts_per_class,
                        point_sampling_strategy="center_weighted",
                        box_noise_level=0.01,
                        generate_box=True,
                        generate_points=True,
                        is_strict=args.kl_is_strict
                    )
                elif args.dataset == 'PanNuke':
                    prompts = prompts2
                else: raise ValueError
            else: pass

            ref_results = get_prompt_preds(
                ref_model, prompts, prompt_mode=args.kl_prompt_type, 
                multimask_output=True, 
                only_best_score_pred=True, 
                only_save_best_prompt_pred=False,
                # return_logits=True
            )
            ref_logits, ref_scores, ref_category_indices = concatenate_masks_and_scores_v2(
                ref_results['prompts_preds'], sort_keys=True
            )
            
            # 保存生成时的logits (用于PPO比率计算)
            gen_logits = current_logits.detach()
        
        adaptive_pool = nn.AdaptiveAvgPool2d(decoded_mask.shape[-2:])
        
        # note: grpo 会出现 loss为负数的情况: https://mp.weixin.qq.com/s/IsPIpsemqtJXNlcb3hJi-A
        if args.rw_dispered:
            # rewards = continuous_to_dispersed(compute_reward(adaptive_pool(current_logits), decoded_mask.float()).reshape(-1, num_prompts_per_class)).flatten()
            rewards1 = continuous_to_dispersed_v2(compute_reward_v2(adaptive_pool(current_logits), decoded_mask.float()).reshape(-1, num_prompts_per_class), scheme=1)
            rewards2 = continuous_to_dispersed_v2(compute_reward_v2(adaptive_pool(current_logits), decoded_mask.float()).reshape(-1, num_prompts_per_class), scheme=2)
            
            if args.rw_func == 'f1':
                rewards = rewards1.flatten()
            elif args.rw_func == 'f2':
                rewards = rewards2.flatten()
            elif args.rw_func == 'all':
                rewards = (rewards1 + rewards2).flatten()
            else:
                raise NotImplementedError

            log_structured_rewards_stats(rewards1, num_prompts_per_class, step=image_idx_in_all_trainset, logger=writer, class_keys=class_keys, name='rewards1')
            log_structured_rewards_stats(rewards2, num_prompts_per_class, step=image_idx_in_all_trainset, logger=writer, class_keys=class_keys, name='rewards2')

        else:   # require: compute_reward( torch.randn(-1, H, w), torch.randn(-1, H, w))
            rewards = normalize_in_chunks_v2(compute_reward_v2(adaptive_pool(current_logits), decoded_mask.float()), chunk_size=num_prompts_per_class, temperature=args.rw_temp)   # 向量
        
        if args.grpo_KL_weight:
            cls_weights = torch.softmax(
                1 / normalize_in_chunks_v2(
                    compute_reward_v2(adaptive_pool(current_logits), decoded_mask.float()), 
                    chunk_size=num_prompts_per_class, temperature=2).reshape(-1, num_prompts_per_class).max(dim=1)[0] / args.weight_temp, 
                dim=0)
            cls_weights = cls_weights[None].repeat(num_prompts_per_class, 1).transpose(0, 1).flatten().cuda()

        if args.is_grpo:
            advantages = rewards.unsqueeze(-1).unsqueeze(-1).expand_as(adaptive_pool(current_logits))   # 扩展维度以匹配logits (N, 1, 1) -> (N, H, W)
            loss, kl_div, sample_loss = grpo_loss_v5(
                adaptive_pool(current_logits), 
                adaptive_pool(ref_logits), 
                decoded_mask, 
                advantages, 
                beta=beta, 
                weights=cls_weights if args.grpo_KL_weight else None,
                clip_param=clip_param, 
                gen_logits=adaptive_pool(gen_logits),
                num_prompts_per_class = num_prompts_per_class
            )
        elif args.is_dpo:
            rewards = rewards.reshape(-1, num_prompts_per_class)
            loss, cls_loss = dpo_loss_segmentation(policy_logits=adaptive_pool(current_logits), ref_logits=adaptive_pool(ref_logits), gt_masks=decoded_mask, 
                                                   rewards=rewards, beta=beta, abla_kl=args.abla_kl, abla_dpo=args.abla_dpo, dpo_weight=args.dpo_weight, args=args)
        else: raise ValueError

        # 绘图
        if args is not None and args.debug:
            from plot_utils import plot_results_np, draw_prompts_on_image, organize_prompts
            organized_prompts, (class_box_prompts, class_point_prompts)  = organize_prompts(prompts)
            all_prompts_vis_imgs = []
            for i_cls in sorted(list(organized_prompts.keys())):
                result_images = draw_prompts_on_image(image, organized_prompts, class_id=i_cls)
                all_prompts_vis_imgs.extend(result_images)
                # for idx in range(len(result_images)):
                    # plt.figure(); plt.imshow(result_images[idx]); plt.title(f'cls{i_cls}-{idx}'); plt.savefig(f'./cls{i_cls}-{idx}.jpg'); plt.close()
            save_paths= [os.path.join('./vis_imgs', f'./{image_idx_in_all_trainset}-cls{i_cls}-{idx}.jpg') for i_cls in organized_prompts.keys() for idx in range(len(result_images))]
            # plot_results_np([image] * len(current_logits), all_prompts_vis_imgs, (current_logits > 0).cpu().numpy(), decoded_mask.cpu().numpy(), rewards=rewards.cpu().numpy(), save_paths=save_paths)

        if writer is not None:
            if args.is_grpo:
                # writer.add_scalar('detail/per_pixel_loss', per_pixel_loss.cpu().item(), image_idx_in_all_trainset)
                writer.add_scalar('detail/kl_div', kl_div.cpu().item(), image_idx_in_all_trainset)
                # writer.add_scalar('detail/ppo_obj', ppo_obj.cpu().item(), image_idx_in_all_trainset)
                writer.add_scalar('detail/sample_loss', sample_loss.cpu().item(), image_idx_in_all_trainset)
                writer.add_scalar('detail/rewards', rewards.mean().cpu().item(), image_idx_in_all_trainset)
            elif args.is_dpo:
                writer.add_scalar('detail/cls_loss', cls_loss.cpu().item(), image_idx_in_all_trainset)
                writer.add_scalar('detail/rewards', rewards.mean().cpu().item(), image_idx_in_all_trainset)
                writer.add_scalar('detail/loss', loss.cpu().item(), image_idx_in_all_trainset)
        batch_losses.append(loss)
        all_rewards.append(rewards.mean().cpu().item())
    
    # 如果批次中有有效样本, 计算平均损失并优化
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()
        
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)  # 添加梯度裁剪
        optimizer.step()
        
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        print(f"GRPO Loss: {total_loss.item():.4f}, Avg Reward: {avg_reward:.4f}")
        
        return total_loss.item(), avg_reward
    
    return 0, 0


def read_single(img=np.random.randint(0, 255, size=(512, 512, 3)), lab=np.random.randint(0, 8, size=(512, 512)), target_size=512): # read random image and its annotaion from  the dataset (LabPics)
        Img = img
        ann_map = lab

        r = np.min([target_size / Img.shape[1], target_size / Img.shape[0]]) # scalling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        mat_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

        inds = np.unique(mat_map)[1:] # load all indices
        points= []
        masks = []
        for ind in inds:
            mask=(mat_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
            masks.append(mask)
            coords = np.argwhere(mask > 0) # get all coordinates in mask
            yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
            points.append([[yx[1], yx[0]]])
        return Img, np.array(masks), np.array(points), np.ones([len(masks),1])


def read_batch_single_v1(img=np.random.randint(0, 255, size=(512, 512, 3)), lab=np.random.randint(0, 8, size=(512, 512)), target_size=512): # read random image and single mask from  the dataset (LabPics)
    Img = img
    ann_map = lab

    r = np.min([target_size / Img.shape[1], target_size / Img.shape[0]]) # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    inds = np.unique(ann_map)[1:] # load all indices
    if inds.__len__()>0:
            ind = inds[np.random.randint(inds.__len__())]  # pick single segment
    else:
            return None  # return read_batch_single(data)

    mask = (ann_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
    coords = np.argwhere(mask > 0) # get all coordinates in mask
    yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
    return Img, mask, [[yx[1], yx[0]]]


def read_batch_single_v2(img=np.random.randint(0, 255, size=(512, 512, 3)), lab=np.random.randint(0, 8, size=(512, 512)), target_size=512, num_pos_points=1, num_neg_points=0):
    """
    读取并处理单个图像和分割掩码, 提取正负样本点和边界框
    
    参数:
        img: 输入图像, 默认为随机生成的RGB图像
        lab: 分割标签图, 默认为随机生成的标签图
        target_size: 目标尺寸, 默认为512
        num_pos_points: 负样本点的数量, 默认为1
        num_neg_points: 负样本点的数量, 默认为0
    
    返回:
        Img: 调整大小后的图像
        mask: 选定分割区域的二值掩码
        pos_points: 正样本点坐标列表 [[x,y]]
        neg_points: 负样本点坐标列表 [[x1,y1], [x2,y2], ...]
        bbox: 分割区域的边界框 [x_min, y_min, x_max, y_max]
    """
    Img = img
    ann_map = lab

    r = np.min([target_size / Img.shape[1], target_size / Img.shape[0]])
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # 获取所有非背景标签
    inds = np.unique(ann_map)[1:]
    if inds.__len__() > 0:
        ind = inds[np.random.randint(inds.__len__())]  # 随机选择一个分割区域
    else:
        return None  # 如果没有分割区域, 返回None

    # 创建二值掩码
    mask = (ann_map == ind).astype(np.uint8)
    
    # 提取正样本点
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None  # 如果掩码为空, 返回None

    pos_points = []
    pos_indices = np.random.choice(len(coords), min(num_pos_points, len(coords)), replace=False)
    for idx in pos_indices:
        ny, nx = coords[idx]
        pos_points.append([nx, ny])  # 转换为[x,y]格式
    
    # 提取负样本点
    neg_mask = (mask == 0).astype(np.uint8)  # 掩码外的区域
    neg_coords = np.argwhere(neg_mask > 0)
    
    neg_points = []
    if len(neg_coords) > 0:
        # 随机选择指定数量的负样本点
        neg_indices = np.random.choice(len(neg_coords), min(num_neg_points, len(neg_coords)), replace=False)
        for idx in neg_indices:
            ny, nx = neg_coords[idx]
            neg_points.append([nx, ny])  # 转换为[x,y]格式
    
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox = [x_min, y_min, x_max, y_max]  # [x_min, y_min, x_max, y_max]格式
    else:
        bbox = [0, 0, 0, 0]
    
    return Img, mask, pos_points, neg_points, bbox


def read_batch(imgs=np.random.randint(0, 255, (16, 3, 512, 512)), labs=np.random.randint(0, 255, (16, 512, 512)), target_size=512, num_pos_points=1, num_neg_points=0):
    limage = []
    lmask = []
    linput_pos_point = []
    linput_neg_point = []
    linput_box = []
    for i in range(len(imgs)):
        img = np.transpose(imgs[i], (1, 2, 0))
        lab = labs[i]
        read_return = read_batch_single_v2(img, lab, target_size, num_pos_points=num_pos_points, num_neg_points=num_neg_points)
        if read_return is not None:
            image, mask, input_pos_point, input_neg_point, input_box = read_return
        else:
            continue
        limage.append(image)
        lmask.append(mask)
        linput_pos_point.append(input_pos_point)
        linput_neg_point.append(input_neg_point) if num_neg_points != 0 else []
        linput_box.append(input_box)
    
    if len(limage) != 0:
        return limage, np.array(lmask), np.array(linput_pos_point), np.array(linput_neg_point), np.array(linput_box), np.ones([len(limage),1])
    else:
        return None


def train_with_seg_single(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=3, beta=0.05, clip_param=0.2, iteration=0, writer=None, args=False):
    # https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/
    if label_batch.sum() == 0:
        return 0, 0
    
    batch_losses = []
    
    for idx, (image_idx, label_idx) in enumerate(zip(range(image_batch.shape[0]), range(label_batch.shape[0]))):
        image_idx_in_all_trainset = iteration*image_batch.shape[0]+idx
        image, label = image_batch[image_idx].permute(1, 2, 0).cpu().numpy(), label_batch[label_idx].cpu().numpy()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): # cast to mix precision
            image, mask, input_point, input_label = read_single(image, label)
            model.set_image(image) # apply SAM image encoder to the image
            
            if label.mean() == 0:
                continue

            mask_input, unnorm_coords, labels, unnorm_box = model._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = model.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in model._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = model.model.sam_mask_decoder(image_embeddings=model._features["image_embed"][-1].unsqueeze(0),
                                                                            image_pe=model.model.sam_prompt_encoder.get_dense_pe(),
                                                                            sparse_prompt_embeddings=sparse_embeddings,
                                                                            dense_prompt_embeddings=dense_embeddings,
                                                                            multimask_output=True,
                                                                            repeat_image=batched_mode,
                                                                            high_res_features=high_res_features,)
            prd_masks = model._transforms.postprocess_masks(low_res_masks, model._orig_hw[-1])# Upscale the masks to the original image resolution

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map

            # plt.imshow(gt_mask.permute(1, 2, 0).cpu().numpy().sum(axis=2)); plt.show()
            # plt.imshow(prd_mask.permute(1, 2, 0).detach().cpu().numpy().sum(axis=2)); plt.show()

            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05  # mix losses

            model.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision

            if image_idx_in_all_trainset % 1000==0: torch.save(model.model.state_dict(), "model.torch");print("save model")

            if image_idx_in_all_trainset == 0: mean_iou = 0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step)",image_idx_in_all_trainset, "Accuracy(IOU)=",mean_iou)
            batch_losses.append(loss)
            writer.add_scalar('detail/loss', loss.cpu().item(), image_idx_in_all_trainset)
            writer.add_scalar('detail/mean_iou', mean_iou, image_idx_in_all_trainset)

    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()
        
        return total_loss.item(), 0
    
    return 0, 0


def train_with_seg_batch(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=3, beta=0.05, clip_param=0.2, iteration=0, writer=None, args=False, **kwargs):
    if label_batch.sum() == 0:
        return 0, 0
    
    if isinstance(args.pos_point_num, tuple):
        num_pos_points = np.random.randint(*args.pos_point_num, size=(1))
    else: num_pos_points = args.pos_point_num
    if isinstance(args.neg_point_num, tuple):
        num_neg_points = np.random.randint(*args.neg_point_num, size=(1))
    else: num_neg_points = args.neg_point_num

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): # cast to mix precision
        read_batch_return = read_batch(image_batch.cpu().numpy(), label_batch.cpu().numpy(), target_size=label_batch.shape[-1], 
                                       num_pos_points=num_pos_points, num_neg_points=num_neg_points)
        if read_batch_return is None:
            return 0, 0
        else:
            image, mask, input_point, input_neg_point, input_box, input_label = read_batch_return
        
        if len(input_neg_point) != 0:
            input_point = np.concatenate([input_point, input_neg_point], axis=1)
            input_label = np.concatenate([input_label, np.zeros((input_label.shape[0], input_neg_point.shape[1]))], axis=1)

        model.set_image_batch(image) # apply SAM image encoder to the image

        if args.prompt_type == 'point':
            mask_input, unnorm_coords, labels, unnorm_box = model._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        elif args.prompt_type == 'box':
            mask_input, unnorm_coords, labels, unnorm_box = model._prep_prompts(None, None, box=input_box, mask_logits=None, normalize_coords=True)
        elif args.prompt_type == 'both':
            mask_input, unnorm_coords, labels, unnorm_box = model._prep_prompts(input_point, input_label, box=input_box, mask_logits=None, normalize_coords=True)
        else: raise ValueError
        
        if unnorm_coords is not None:
            concat_points = (unnorm_coords, labels)
        else:
            concat_points = None

        if unnorm_box is not None:
            box_coords = unnorm_box.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device='cuda')
            box_labels = box_labels.repeat(unnorm_box.size(0), 1)
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)
        sparse_embeddings, dense_embeddings = model.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None,)

        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in model._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = model.model.sam_mask_decoder(image_embeddings=model._features["image_embed"],
                                                                        image_pe=model.model.sam_prompt_encoder.get_dense_pe(),
                                                                        sparse_prompt_embeddings=sparse_embeddings,
                                                                        dense_prompt_embeddings=dense_embeddings,
                                                                        multimask_output=True,
                                                                        repeat_image=False,
                                                                        high_res_features=high_res_features)
        prd_masks = model._transforms.postprocess_masks(low_res_masks, model._orig_hw[-1])# Upscale the masks to the original image resolution

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        multimask_len = prd_masks.shape[1]
        multimask_idx = 0   # Todo: 这个要怎么使用呢？
        prd_mask = torch.sigmoid(prd_masks[:, multimask_idx])# Turn logit map to probability map    # 为什么这是第一个索引
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, multimask_idx] - iou).mean()
        loss = seg_loss + score_loss * 0.05  # mix losses

        model.model.zero_grad() # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update() # Mix precision

        # if iteration % 1000==0: torch.save(model.model.state_dict(), "model.torch");print("save model")
        if iteration == 0: args.mean_iou = 0
        args.mean_iou = args.mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        writer.add_scalar('detail/loss', loss.cpu().item(), iteration)
        writer.add_scalar('detail/mean_iou', args.mean_iou, iteration)
        print("step)",iteration, "Accuracy(IOU)=", args.mean_iou)

    return loss.item(), 0


def val_with_seg_batch():
    pass


def grpo_loss_v2(current_logits, ref_logits, pred_binary, gen_logits, rewards, beta=0.05, clip_param=0.2, gen_probs=None, num_prompts_per_class=3, rw_temp=1):
    """适用于单通道二值分割的GRPO损失函数"""
    # 输入维度验证
    assert current_logits.dim() == 3 and ref_logits.dim() == 3
    assert pred_binary.shape == current_logits.shape
    assert gen_logits.shape == current_logits.shape
    assert rewards.dim() == 1 and rewards.shape[0] == current_logits.shape[0]

    current_logits = rearrange(current_logits, '(b n) h w -> b n h w', n=num_prompts_per_class)
    ref_logits = rearrange(ref_logits, '(b n) h w -> b n h w', n=num_prompts_per_class)
    pred_binary = rearrange(pred_binary, '(b n) h w -> b n h w', n=num_prompts_per_class)
    gen_logits = rearrange(gen_logits, '(b n) h w -> b n h w', n=num_prompts_per_class)
    gen_probs = rearrange(gen_probs, '(b n) h w -> b n h w', n=num_prompts_per_class) if gen_probs is not None else None

    # 在模型输出层添加数值约束（例如限制logits在[-10,10]之间）
    current_logits = torch.clamp(current_logits, -10, 10)
    ref_logits = torch.clamp(ref_logits, -10, 10)

    # rewards = rearrange(normalize_in_chunks(rewards, chunk_size=num_prompts_per_class), '(b n) -> b n', n=num_prompts_per_class)
    rewards = torch.softmax(rewards.view(-1, num_prompts_per_class)/rw_temp, dim=1) # 这个是错误的

    batch_size, _, h, w = current_logits.shape
    device = current_logits.device

    # 转换logits到概率空间 (二值分割使用sigmoid)
    current_probs = torch.sigmoid(current_logits)
    ref_probs = torch.sigmoid(ref_logits)

    # 计算当前策略的对数概率 (二元交叉熵形式)
    current_log_probs = F.logsigmoid(current_logits)          # 正类对数概率
    current_neg_log_probs = F.logsigmoid(-current_logits)      # 负类对数概率

    # 计算参考策略的对数概率
    ref_log_probs = F.logsigmoid(ref_logits)
    ref_neg_log_probs = F.logsigmoid(-ref_logits)

    # 根据预测掩码选择对应的对数概率
    current_p_log = pred_binary * current_log_probs + (1 - pred_binary) * current_neg_log_probs
    ref_p_log = pred_binary * ref_log_probs + (1 - pred_binary) * ref_neg_log_probs

    ## 计算每个像素的KL散度 (exp(ref - cur) - (ref - cur) - 1)
    # per_pixel_kl = torch.exp(ref_p_log - current_p_log) - (ref_p_log - current_p_log) - 1
    # 添加数值稳定措施
    diff = torch.clamp(ref_p_log - current_p_log, max=10)  # 限制最大差值
    per_pixel_kl = torch.exp(diff) - diff - 1

    # 准备优势函数 (扩展奖励到每个像素)
    advantages = rewards[..., None, None].expand(batch_size, num_prompts_per_class, h, w)  # [15, 3,224,224]

    # 生成时的对数概率处理 (如果提供)
    if gen_probs is not None:
        advantages = advantages.to(device)
        gen_probs = gen_probs.to(device)
        # ratio = torch.exp(current_p_log - gen_probs)
        # 在计算ratio时添加保护
        ratio = torch.exp(torch.clamp(current_p_log - gen_probs, min=-5, max=5))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # 基础策略梯度形式
        policy_loss = -current_p_log * advantages

    # 组合损失项 (策略损失 + KL约束)
    total_loss =  per_pixel_kl * beta + policy_loss

    # 计算有效区域掩码 (忽略填充区域)
    valid_mask = (gen_logits >= 0).float()  # 假设-1表示无效区域

    # 计算加权平均损失
    masked_loss = (total_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    return masked_loss, total_loss.mean(), per_pixel_kl.mean(), policy_loss.mean(), rewards.mean()


def grpo_loss_v3(current_logits, ref_logits, gt_masks, advantages, beta=0.05, clip_param=0.2, gen_logits=None, num_prompts_per_class=3):  
    # 计算像素级概率
    current_probs = torch.sigmoid(current_logits)
    ref_probs = torch.sigmoid(ref_logits)
    
    # note: https://github.com/lsdefine/simple_GRPO/blob/a77bfb43abc4297d3d5a6b221863411da837c578/simple_grpo_v1/grpo_ref_split.py#L187
    gen_probs = torch.sigmoid(gen_logits)  # 生成时的概率
    
    # 计算PPO比率 (当前策略/行为策略)
    ratio = current_probs / (gen_probs + 1e-8)
    clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
    
    # PPO目标函数
    ppo_obj = torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # KL散度惩罚项 (用于约束与参考模型的距离)
    kl_div = ref_probs * torch.log(ref_probs / (current_probs + 1e-8) + 1e-8) + \
             (1 - ref_probs) * torch.log((1 - ref_probs) / (1 - current_probs + 1e-8) + 1e-8)
    
    # 组合损失 (最大化奖励的同时最小化KL散度)
    per_pixel_loss = -(ppo_obj - beta * kl_div)   # origin

    criterion = nn.BCEWithLogitsLoss()

    sample_loss = criterion(current_logits, gt_masks.float().cuda())
    
    # # 只关注有效区域 (可能需要一个mask来排除填充区域)
    # valid_mask = (gt_masks != -1).float()  # 假设-1代表忽略区域
    # loss = (per_pixel_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    loss = per_pixel_loss.mean() + sample_loss

    return loss, per_pixel_loss.mean(), kl_div.mean(), ppo_obj.mean(), sample_loss


def grpo_loss_v4(current_logits, ref_logits, gt_masks, advantages, beta=0.05, clip_param=0.2, gen_logits=None, num_prompts_per_class=3, weights=None, criterion=nn.BCEWithLogitsLoss()):
    # 数值截断防止 logits 过大
    current_logits = torch.clamp(current_logits, -5, 5)
    ref_logits = torch.clamp(ref_logits, -5, 5)
    gen_logits = torch.clamp(gen_logits, -5, 5)

    # 计算概率并截断
    eps = 1e-8
    current_probs = torch.clamp(torch.sigmoid(current_logits), min=eps, max=1-eps)
    ref_probs = torch.clamp(torch.sigmoid(ref_logits), min=eps, max=1-eps)
    gen_probs = torch.clamp(torch.sigmoid(gen_logits), min=eps, max=1-eps)

    # 计算 PPO 比率
    ratio = current_probs / gen_probs
    clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
    ppo_obj = torch.min(ratio * advantages, clipped_ratio * advantages)

    # 修正 KL 散度计算
    kl_ref_current = ref_probs * (torch.log(ref_probs + eps) - torch.log(current_probs + eps))
    kl_background = (1 - ref_probs) * (torch.log(1 - ref_probs + eps) - torch.log(1 - current_probs + eps))
    if weights is not None:
        kl_div = (kl_ref_current + kl_background) * beta * weights.unsqueeze(-1).unsqueeze(-1)
    else:
        kl_div = (kl_ref_current + kl_background) * beta

    # 组合损失
    # per_pixel_loss = -(ppo_obj - beta * kl_div)
    per_pixel_loss = ((-(ppo_obj - kl_div))*gt_masks).sum() / (gt_masks.sum() + eps)
    sample_loss = criterion(current_logits, gt_masks.float().cuda())
    loss = per_pixel_loss.mean() + sample_loss

    return loss, kl_div.mean(), sample_loss


def stable_log_sigmoid(logits):
    """Numerically stable log sigmoid."""
    # log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x)) = -softplus(-x)
    return -F.softplus(-logits)

def stable_log_one_minus_sigmoid(logits):
    """Numerically stable log(1 - sigmoid(x))."""
    # log(1 - sigmoid(x)) = log(exp(-x) / (1 + exp(-x))) = -x - log(1 + exp(-x)) = -x - softplus(-x) = -softplus(x)
    return -F.softplus(logits)

def grpo_loss_v5(
    current_logits,  # 当前模型输出的 logits (经过 adaptive_pool)
    ref_logits,      # 参考模型输出的 logits (经过 adaptive_pool)
    decoded_mask,    # GT mask (可能不需要，因为 advantage 已基于此计算)
    advantages,      # 预先计算的优势值 (通常基于奖励)，已扩展到 logits 形状
    beta,            # KL 散度惩罚系数
    weights=None,    # 可选的样本权重 (例如，基于类别的 KL 权重)
    clip_param=0.2,  # PPO 裁剪参数 epsilon
    gen_logits=None, # 生成数据时（前向传播时）的 logits (detached, 经过 adaptive_pool)
    num_prompts_per_class=1 # (可能不需要，除非权重需要特殊处理)
):
    """
    计算 GRPO (PPO-style) 损失。

    Args:
        current_logits (torch.Tensor): 当前策略网络的 logits (N, H, W) 或 (N, C, H, W)。
                                        假设是 (N, H, W) 代表二分类分割的 logits。
        ref_logits (torch.Tensor): 参考策略网络的 logits (N, H, W)。
        decoded_mask (torch.Tensor): GT mask (N, H, W)。 (在此实现中未使用，优势已包含奖励信息)
        advantages (torch.Tensor): 优势函数值，已扩展到 logits 形状 (N, H, W)。
        beta (float): KL 散度惩罚项的系数。
        weights (torch.Tensor, optional): 每个样本的权重 (N,)。默认为 None。
        clip_param (float): PPO 裁剪范围 (epsilon)。
        gen_logits (torch.Tensor): 用于计算 PPO 比率的生成时 logits (N, H, W)。
        num_prompts_per_class (int): (在此实现中未使用)。

    Returns:
        tuple: (总损失, KL 散度部分, PPO 样本损失部分)
    """
    # 确保 gen_logits 存在
    if gen_logits is None:
        raise ValueError("gen_logits must be provided for PPO ratio calculation.")

    # --- 1. 计算 PPO 裁剪替代目标 (Clipped Surrogate Objective) ---

    # 计算当前策略和生成策略下，每个像素"动作"的对数概率
    # 假设是二分类问题，使用 log sigmoid
    # log π_θ(a|s)  (a 代表像素值，这里用 logits 近似)
    log_probs_current = stable_log_sigmoid(current_logits)
    # log π_θ_old(a|s) (生成数据时的策略)
    log_probs_gen = stable_log_sigmoid(gen_logits) # gen_logits 应该是 detach() 过的

    # 计算重要性采样比率 r(θ) = π_θ / π_θ_old = exp(log π_θ - log π_θ_old)
    # 在实践中，直接使用 log ratio 更稳定
    log_ratio = log_probs_current - log_probs_gen
    ratio = torch.exp(log_ratio)

    # 计算 PPO 目标函数的两个部分
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    # PPO 损失是最小化目标的负值 (因为我们要最大化目标函数)
    # 对每个像素计算损失，然后取平均
    # 注意：这里假设 advantages > 0 对应好的动作，< 0 对应坏的动作
    # PPO 论文是 E[min(r*A, clip(r)*A)], 我们要最小化 -E[...]
    ppo_loss_per_pixel = -torch.min(surr1, surr2)

    # --- 2. 计算 KL 散度惩罚 ---
    # KL(π_ref || π_θ) 或 KL(π_θ || π_ref)
    # 通常计算 KL( 当前策略 || 参考策略 ) 作为惩罚项
    # KL散度 D_KL(P || Q) for Bernoulli distributions parameterized by logits l_p, l_q:
    # p * (log p - log q) + (1-p) * (log(1-p) - log(1-q))
    # where p = sigmoid(l_p), q = sigmoid(l_q)

    # 使用稳定计算的版本
    log_p_current = stable_log_sigmoid(current_logits)
    log_one_minus_p_current = stable_log_one_minus_sigmoid(current_logits)
    log_p_ref = stable_log_sigmoid(ref_logits)
    log_one_minus_p_ref = stable_log_one_minus_sigmoid(ref_logits)

    # 计算 KL( ref || current ) - 稍微简单点，因为 ref_logits 不需要梯度
    # p_ref = torch.sigmoid(ref_logits.detach()) # 不需要梯度
    # kl_div_per_pixel = p_ref * (log_p_ref - log_p_current) + \
    #                    (1 - p_ref) * (log_one_minus_p_ref - log_one_minus_p_current)

    # 或者计算 KL( current || ref ) - 更常见于约束当前策略不要偏离参考策略太远
    p_current = torch.sigmoid(current_logits) # 需要梯度
    kl_div_per_pixel = p_current * (log_p_current - log_p_ref) + \
                       (1 - p_current) * (log_one_minus_p_current - log_one_minus_p_ref)


    # --- 3. 聚合和加权 ---

    # 对空间维度 (H, W) 求平均得到每个样本的损失
    ppo_loss_per_sample = ppo_loss_per_pixel.mean(dim=[1, 2])
    kl_div_per_sample = kl_div_per_pixel.mean(dim=[1, 2])

    # 应用样本权重 (如果提供)
    if weights is not None:
        # 确保权重形状匹配 (N,)
        if weights.shape[0] != ppo_loss_per_sample.shape[0]:
             raise ValueError(f"Weights shape {weights.shape} does not match sample shape {ppo_loss_per_sample.shape}")
        # 加权平均
        weighted_ppo_loss = (ppo_loss_per_sample * weights).mean()
        weighted_kl_div = (kl_div_per_sample * weights).mean()
    else:
        # 直接平均
        weighted_ppo_loss = ppo_loss_per_sample.mean()
        weighted_kl_div = kl_div_per_sample.mean()

    # --- 4. 计算总损失 ---
    total_loss = weighted_ppo_loss + beta * weighted_kl_div

    # 返回总损失以及各部分，方便记录和调试
    # 返回的 sample_loss 应该是 PPO 目标部分，kl_div 是 KL 惩罚部分
    return total_loss, weighted_kl_div, weighted_ppo_loss


def dpo_loss_segmentation(policy_logits, ref_logits, gt_masks, rewards, beta=0.05, criterion=nn.BCEWithLogitsLoss(), abla_kl=False, abla_dpo=False, dpo_weight=1., args=None):
    """
    note: https://github.com/modelscope/ms-swift/blob/main/swift/trainers/rlhf_trainer/dpo_trainer.py
    像素级 DPO Loss for 语义分割 (二分类前景分割).

    Args:
        policy_logits: 策略模型输出的 logits, shape (batch_size, height, width).
        ref_logits: 参考模型输出的 logits, shape (batch_size, height, width).
        gt_masks: 真实标签的二分类掩码, shape (batch_size, height, width).
        rewards: 
        beta: DPO 的 beta 参数.

    Returns:
        loss: DPO loss.
    """
    # 1. 计算 log 概率 (像素级别)
    policy_logits = rearrange(policy_logits, '(b n) h w -> b n h w', n=args.num_prompts_per_class)
    ref_logits = rearrange(ref_logits, '(b n) h w -> b n h w', n=args.num_prompts_per_class)
    mbatch_size, num_prompts, height, width = policy_logits.shape
    chosen_indices = torch.argmax(rewards, dim=1)
    rejected_indices = torch.argmin(rewards, dim=1)
    
    # chosen_indices = chosen_indices.view(15, 1, 1, 1).expand(-1, -1, 64, 64)
    chosen_indices = torch.repeat_interleave(chosen_indices, repeats=height * width, dim=0).reshape(mbatch_size, 1, height, width)
    rejected_indices = torch.repeat_interleave(rejected_indices, repeats=height * width, dim=0).reshape(mbatch_size, 1, height, width)

    policy_logprobs, ref_logprobs = F.logsigmoid(policy_logits), F.logsigmoid(ref_logits)

    # 判断对张量的操作是否正确
    # a = torch.repeat_interleave(chosen_indices, repeats=height * width, dim=0).reshape(mbatch_size, 1, height, width)
    # b = torch.gather(policy_logprobs, dim=1, index=a) 
    # assert sum([(b[i, 0] - policy_logprobs[i, torch.argmax(rewards, dim=1)[i]]).mean().item() for i in range(5)]) == 0

    # 根据选择和拒绝的索引，提取对应的 log probabilities
    policy_chosen_logps = torch.gather(policy_logprobs, dim=1, index=chosen_indices) # Shape: (mbatch_size, 1, height, width)
    policy_rejected_logps = torch.gather(policy_logprobs, dim=1, index=rejected_indices) # Shape: (mbatch_size, 1, height, width)
    ref_chosen_logps = torch.gather(ref_logprobs, dim=1, index=chosen_indices) # Shape: (mbatch_size, 1, height, width)
    ref_rejected_logps = torch.gather(ref_logprobs, dim=1, index=rejected_indices) # Shape: (mbatch_size, 1, height, width)

    # 2. 计算 log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    # 3. DPO Loss (像素级别)
    dpo_loss = -F.logsigmoid(beta * logits)

    if abla_kl:
        loss = dpo_loss.mean()
        sample_loss = torch.tensor(0).cuda()
    elif abla_dpo:
        sample_loss_choosen = criterion(policy_chosen_logps.squeeze(1), gt_masks.float().cuda()[::args.num_prompts_per_class])   # criterion(policy_chosen_logps, gt_masks.float().cuda()[torch.argmax(rewards, dim=1)])
        sample_loss_rejected = criterion(policy_rejected_logps.squeeze(1), gt_masks.float().cuda()[::args.num_prompts_per_class])  # criterion(policy_rejected_logps, gt_masks.float().cuda()[torch.argmin(rewards, dim=1)])
        sample_loss = sample_loss_choosen + sample_loss_rejected * 0.3

        loss = sample_loss
    else:
        # sample_loss_choosen = criterion(policy_chosen_logps, gt_masks.float().cuda()[torch.argmax(rewards, dim=1)])
        # sample_loss_rejected = criterion(policy_rejected_logps, gt_masks.float().cuda()[torch.argmin(rewards, dim=1)])
        sample_loss_choosen = criterion(policy_chosen_logps.squeeze(1), gt_masks.float().cuda()[::args.num_prompts_per_class])   # criterion(policy_chosen_logps, gt_masks.float().cuda()[torch.argmax(rewards, dim=1)])
        sample_loss_rejected = criterion(policy_rejected_logps.squeeze(1), gt_masks.float().cuda()[::args.num_prompts_per_class])  # criterion(policy_rejected_logps, gt_masks.float().cuda()[torch.argmin(rewards, dim=1)])
        sample_loss = sample_loss_choosen + sample_loss_rejected * 0.3

        loss = dpo_weight * dpo_loss.mean() + sample_loss
    return loss, sample_loss


# part of grpo_loss_v3
def f1_score(pred_mask, gt_mask, smooth=1e-6):
    """
    计算两个二值掩码之间的F1分数
    
    参数:
        pred_mask: 预测掩码, 形状为[N, H, W]的张量, 值为0或1
        gt_mask: 真实掩码, 形状为[N, H, W]的张量, 值为0或1
        smooth: 平滑项, 避免除零错误
        
    返回:
        形状为[N]的F1分数张量, 每个样本一个分数
    """
    # 确保输入是二值的
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()
    
    # 计算真阳性(TP)、假阳性(FP)和假阴性(FN)
    true_positive = (pred_mask * gt_mask).sum(dim=[1, 2])
    false_positive = (pred_mask * (1 - gt_mask)).sum(dim=[1, 2])
    false_negative = ((1 - pred_mask) * gt_mask).sum(dim=[1, 2])
    
    # 计算精确率和召回率
    precision = true_positive / (true_positive + false_positive + smooth)
    recall = true_positive / (true_positive + false_negative + smooth)
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    return f1


# part of grpo_loss_v3
def extract_boundary(mask, kernel_size=3):
    """
    从二值掩码中提取边界
    
    参数:
        mask: 形状为[N, H, W]的二值掩码张量, 值为0或1
        kernel_size: 形态学操作的核大小
    
    返回:
        边界掩码, 其中边界像素为1, 其他为0
    """
    # 确保输入是二值掩码
    if not torch.all((mask == 0) | (mask == 1)):
        mask = (mask > 0.5).float()
    
    # 创建形态学操作的核
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    
    # 添加通道维度[N, H, W] -> [N, 1, H, W]
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    
    # 膨胀操作
    dilated = F.conv2d(mask, kernel, padding=kernel_size//2)
    dilated = (dilated > 0).float()
    
    # 腐蚀操作
    eroded = F.conv2d(mask, kernel, padding=kernel_size//2)
    eroded = (eroded == kernel_size*kernel_size).float()
    
    # 边界 = 膨胀 - 腐蚀
    boundary = dilated - eroded
    
    # 如果原始输入是[N, H, W], 则恢复此形状
    if mask.dim() == 4 and mask.size(1) == 1:
        boundary = boundary.squeeze(1)
    
    return boundary


# part of grpo_loss_v3
def compute_boundary_accuracy(pred_mask, gt_mask):
    # 使用形态学操作提取边界
    pred_boundary = extract_boundary(pred_mask)
    gt_boundary = extract_boundary(gt_mask)
    # 计算边界F1分数
    return f1_score(pred_boundary, gt_boundary)


# part of grpo_loss_v3
def compute_reward(pred_logits, gt_mask):
    # 将logits转换为二值掩码
    pred_mask = (pred_logits > 0).float()
    
    # 计算IoU (Intersection over Union)
    intersection = (pred_mask * gt_mask).sum(dim=[1, 2])
    union = pred_mask.sum(dim=[1, 2]) + gt_mask.sum(dim=[1, 2]) - intersection
    iou = intersection / (union + 1e-8)
    
    # 可以添加额外奖励组件
    # 例如边缘准确度、平滑度等
    # boundary_accuracy = compute_boundary_accuracy(pred_mask, gt_mask)
    
    # 组合奖励
    rewards = iou # + 0.2 * boundary_accuracy
    return rewards

def compute_reward_v2(pred_logits, gt_mask, metric='dice', smooth=1.0):
    """
    计算预测 logits 和 GT 掩码之间的奖励。

    Args:
        pred_logits (torch.Tensor): 模型输出的原始 logits (N, H, W)。
        gt_mask (torch.Tensor): 真实掩码 (N, H, W)，通常是 0 或 1 的整数类型。
        metric (str): 使用的奖励指标，可选 'dice' 或 'iou'。默认为 'dice'。
        smooth (float): 用于防止除以零并增加数值稳定性的平滑因子。默认为 1.0。

    Returns:
        torch.Tensor: 每个样本的奖励值 (N,)。
    """
    # 1. 将 logits 转换为概率 (0 到 1 之间)
    pred_probs = torch.sigmoid(pred_logits)

    # 2. 确保 GT 掩码是 float 类型，以便进行乘法运算
    gt_mask = gt_mask.float()

    # 3. 计算交集和各自的总和 (在空间维度 H, W 上求和)
    # 保留 batch 维度 N
    intersection = (pred_probs * gt_mask).sum(dim=[1, 2])
    pred_sum = pred_probs.sum(dim=[1, 2])
    gt_sum = gt_mask.sum(dim=[1, 2])

    # 4. 根据选择的指标计算奖励
    if metric == 'dice':
        # Soft Dice 系数: 2 * |A ∩ B| / (|A| + |B|)
        rewards = (2. * intersection + smooth) / (pred_sum + gt_sum + smooth)
    elif metric == 'iou':
        # Soft IoU (Jaccard) 系数: |A ∩ B| / (|A ∪ B|) = |A ∩ B| / (|A| + |B| - |A ∩ B|)
        union = pred_sum + gt_sum - intersection
        rewards = (intersection + smooth) / (union + smooth)
    else:
        raise ValueError(f"不支持的 metric: {metric}. 请选择 'dice' 或 'iou'.")

    # 5. （可选）添加其他奖励组件
    # boundary_accuracy = compute_boundary_accuracy(pred_probs > 0.5, gt_mask) # 如果需要，可以基于阈值计算
    # rewards = rewards + 0.1 * boundary_accuracy # 示例：组合奖励

    return rewards


# 6. 定期更新参考模型的函数
def update_reference_model(model, ref_model, update_ratio=0.1):
    """软更新参考模型参数"""
    with torch.no_grad():
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            ref_param.data.copy_(update_ratio * param.data + (1 - update_ratio) * ref_param.data)


def log_advantage_stats(rewards, step, logger):
    """记录rewards的多个统计量"""
    stats = {
        "rewards/max": rewards.max().item(),
        "rewards/min": rewards.min().item(),
        "rewards/mean": rewards.mean().item(),
        "rewards/std": rewards.std().item(),
        "rewards/abs_max": rewards.abs().max().item(),  # 绝对值最大值, 特别重要
        "rewards/range": rewards.max().item() - rewards.min().item()
    }
    
    # 可选：计算大值的比例
    large_adv_ratio = (rewards.abs() > 5.0).float().mean().item()
    stats["rewards/large_value_ratio"] = large_adv_ratio
    
    # 记录到你的日志系统
    for name, value in stats.items():
        logger.add_scalar(name, value, step)
    
    return stats


def log_structured_rewards_stats(rewards, num_prompts_per_class, step, logger, class_keys, name='rewards'):
    """记录结构化rewards的统计量"""
    
    # 全局统计量
    flat_adv = rewards.view(-1)
    global_stats = {
        f"{name}/global_max": flat_adv.max().item(),
        f"{name}/global_min": flat_adv.min().item(),
        f"{name}/global_mean": flat_adv.mean().item(),
        f"{name}/global_std": flat_adv.std().item(),
        f"{name}/global_abs_max": flat_adv.abs().max().item(),
    }
    
    # 按类别统计
    rewards = rewards.reshape(-1, num_prompts_per_class)
    for c, cname in zip(range(rewards.shape[0]), class_keys):
        class_adv = rewards[c]
        prefix = f"{name}/class_{cname}/"
        class_stats = {
            f"{prefix}max": class_adv.max().item(),
            f"{prefix}mean": class_adv.mean().item(),
            f"{prefix}std": class_adv.std().item(),
        }
        global_stats.update(class_stats)
    
    # 记录到日志系统
    for name, value in global_stats.items():
        logger.add_scalar(name, value, step)
    
    return global_stats