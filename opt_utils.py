import torch 
from torch import nn
from einops import rearrange
from gen_prompt import generate_prompts_from_semantic_mask, get_prompt_preds, concatenate_masks_and_scores_v2, plot_results


def vanilla_opt(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=3):
    batch_losses = []
    for image_idx, label_idx in zip(range(image_batch.shape[0]), range(label_batch.shape[0])):
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
        # 按照decoded_mask的键的顺序进行排序，然后将值重复三次再全部拼接起来。
        exist_keys = sorted(decoded_mask.keys())
        for key in exist_keys:
            # decoded_mask[key] = np.repeat(decoded_mask[key], 3, axis=0)
            decoded_mask[key] = torch.from_numpy(decoded_mask[key])[None].long().repeat(num_prompts_per_class, 1, 1)
        decoded_mask = torch.concat(list(decoded_mask.values()), dim=0)

        # 函数: 接受 image、point、model, 输出 prediction
        model.set_image(image)
        results = get_prompt_preds(model, prompts, prompt_mode='point', multimask_output=True, only_best_score_pred=True, only_save_best_prompt_pred=False)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict

# 1. 首先创建参考模型（冻结的SAM副本）
def create_reference_model(model):
    ref_model = copy.deepcopy(model)
    for param in ref_model.model.parameters():
        param.requires_grad = False
    ref_model.model.eval()
    return ref_model


# 每3个元素为一组进行标准化
def normalize_in_chunks(x, chunk_size):
    # 复制结果张量
    result = x.clone()
    
    # 遍历每个chunk
    for i in range(0, len(x), chunk_size):
        chunk = x[i:i+chunk_size]
        mean = chunk.mean()
        std = chunk.std(unbiased=False)  # 使用无偏估计
        result[i:i+chunk_size] = (chunk - mean) / std

    return result


# 2. 实现奖励函数计算
def compute_segmentation_rewards(pred_masks, gt_masks, images):
    """计算分割质量奖励"""
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
    # 已设置图像的情况下，直接使用prompts获取logits
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
        raise ValueError("模型预测未返回logits，请确保get_prompt_preds函数支持return_logits参数")
    
    # 如果提供了pred_masks，计算这些掩码的对数概率
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

# 4. 实现GRPO损失函数
def grpo_loss(current_logits, ref_logits, pred_masks, gt_masks, rewards, beta=0.05, clip_param=0.2, gen_log_probs=None):
    """计算GRPO损失"""
    # 计算当前模型的对数概率    # Todo: 确定 pred_masks 的size要跟logits一致
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
    
    # 如果有生成时的对数概率，使用PPO的比率裁剪
    if gen_log_probs is not None:
        gen_pixel_log_probs = pred_binary * F.logsigmoid(gen_log_probs) + (1 - pred_binary) * F.logsigmoid(-gen_log_probs)
        ratio = torch.exp(current_pixel_log_probs - gen_pixel_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_pixel_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # 简化版本，直接使用当前对数概率
        per_pixel_loss = -current_pixel_log_probs * advantages
    
    # 添加KL惩罚项
    per_pixel_loss = per_pixel_loss + beta * per_pixel_kl
    
    # 计算有效像素的掩码 (忽略填充区域)
    valid_mask = (gt_masks >= 0).float()  # 假设-1表示填充区域
    
    # 应用掩码并计算平均损失
    masked_loss = (per_pixel_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    
    return masked_loss

# 5. 主要训练循环修改
def train_with_grpo(model, optimizer, scaler, image_batch, label_batch, num_prompts_per_class=3, beta=0.05, clip_param=0.2, iteration=0, writer=None):
    # 创建参考模型 (如果尚未创建)
    if not hasattr(train_with_grpo, 'ref_model'):
        train_with_grpo.ref_model = create_reference_model(model)
    
    ref_model = train_with_grpo.ref_model
    
    batch_losses = []
    all_rewards = []
    
    for idx, (image_idx, label_idx) in enumerate(zip(range(image_batch.shape[0]), range(label_batch.shape[0]))):
        image, label = image_batch[image_idx].permute(1, 2, 0).cpu().numpy(), label_batch[label_idx].cpu().numpy()
        
        if label.mean() == 0: 
            # print("无前景; Skip")
            continue
        
        # 获取prompts和GT掩码 (与原代码相同)
        prompts = generate_prompts_from_semantic_mask(
            label,
            class_ids=None,
            num_positive_points=(2, 3),
            num_negative_points=(4, 6),
            num_prompts_per_class=num_prompts_per_class,
            point_sampling_strategy="center_weighted",
            box_noise_level=0.1,
            generate_box=True,
            generate_points=True
        )
        
        decoded_mask = prompts['decoded_mask']
        exist_keys = sorted(decoded_mask.keys())
        for key in exist_keys:
            decoded_mask[key] = torch.from_numpy(decoded_mask[key])[None].long().repeat(num_prompts_per_class, 1, 1)
        decoded_mask = torch.concat(list(decoded_mask.values()), dim=0).cuda()
        
        # 设置图像
        model.set_image(image)
        ref_model.set_image(image)
        
        # 1. 获取当前模型预测
        with torch.enable_grad():  # 确保启用梯度
            results = get_prompt_preds(
                model, prompts, prompt_mode='point', 
                multimask_output=True, 
                only_best_score_pred=True, 
                only_save_best_prompt_pred=False,
                # return_logits=True  # 返回logits
            )
            
            current_logits, current_scores, current_category_indices = concatenate_masks_and_scores_v2(
                results['prompts_preds'], sort_keys=True
            )
        
        # 2. 获取参考模型预测 (无梯度)
        with torch.no_grad():
            ref_results = get_prompt_preds(
                ref_model, prompts, prompt_mode='both', 
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
        
        # 3. 计算奖励
        adaptive_pool = nn.AdaptiveAvgPool2d(decoded_mask.shape[-2:])
        pred_binary = (adaptive_pool(current_logits) > 0).float()  # 将预测掩码二值化
        
        # rewards = compute_segmentation_rewards(
        #     resized_masks, 
        #     decoded_mask, 
        #     torch.from_numpy(image).permute(2, 0, 1).cuda() if len(image.shape) == 3 else torch.from_numpy(image).cuda()
        # )
        rewards = current_scores  # 这个需要分组之后再进行标准化
        
        # 4. 计算GRPO损失; grpo_loss_v2 已经优化数值范围
        loss, to_loss, kl_loss, policy_loss, norm_rewards = grpo_loss_v2(
            adaptive_pool(current_logits), 
            adaptive_pool(ref_logits), 
            pred_binary, 
            decoded_mask, 
            rewards, 
            beta=beta, 
            clip_param=clip_param, 
            gen_log_probs=adaptive_pool(gen_logits),
            num_prompts_per_class = num_prompts_per_class
        )
        if writer is not None:
            writer.add_scalar('info/to_loss', to_loss.cpu().item(), iteration*image_batch.shape[0]+idx)
            writer.add_scalar('info/kl_loss', kl_loss.cpu().item(), iteration*image_batch.shape[0]+idx)
            writer.add_scalar('info/policy_loss', policy_loss.cpu().item(), iteration*image_batch.shape[0]+idx)
            writer.add_scalar('info/norm_rewards', norm_rewards.cpu().item(), iteration*image_batch.shape[0]+idx)
        batch_losses.append(loss)
        all_rewards.append(norm_rewards.cpu().item())
    
    # 如果批次中有有效样本，计算平均损失并优化
    if batch_losses:
        total_loss = torch.stack(batch_losses).mean()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        print(f"GRPO Loss: {total_loss.item():.4f}, Avg Reward: {avg_reward:.4f}")
        
        return total_loss.item(), avg_reward
    
    return 0, 0


def grpo_loss_v2(current_logits, ref_logits, pred_binary, gt_masks, rewards, beta=0.05, clip_param=0.2, gen_log_probs=None, num_prompts_per_class=3):
    """适用于单通道二值分割的GRPO损失函数"""
    # 输入维度验证
    assert current_logits.dim() == 3 and ref_logits.dim() == 3
    assert pred_binary.shape == current_logits.shape
    assert gt_masks.shape == current_logits.shape
    assert rewards.dim() == 1 and rewards.shape[0] == current_logits.shape[0]

    current_logits = rearrange(current_logits, '(b n) h w -> b n h w', n=num_prompts_per_class)
    ref_logits = rearrange(ref_logits, '(b n) h w -> b n h w', n=num_prompts_per_class)
    pred_binary = rearrange(pred_binary, '(b n) h w -> b n h w', n=num_prompts_per_class)
    gt_masks = rearrange(gt_masks, '(b n) h w -> b n h w', n=num_prompts_per_class)
    gen_log_probs = rearrange(gen_log_probs, '(b n) h w -> b n h w', n=num_prompts_per_class) if gen_log_probs is not None else None

    # 在模型输出层添加数值约束（例如限制logits在[-10,10]之间）
    current_logits = torch.clamp(current_logits, -10, 10)
    ref_logits = torch.clamp(ref_logits, -10, 10)

    rewards = rearrange(normalize_in_chunks(rewards, chunk_size=num_prompts_per_class), '(b n) -> b n', n=num_prompts_per_class)    # TODO: 感觉这个计算有问题，会出现负数！！

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
    if gen_log_probs is not None:
        advantages = advantages.to(device)
        gen_log_probs = gen_log_probs.to(device)
        # ratio = torch.exp(current_p_log - gen_log_probs)
        # 在计算ratio时添加保护
        ratio = torch.exp(torch.clamp(current_p_log - gen_log_probs, min=-5, max=5))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # 基础策略梯度形式
        policy_loss = -current_p_log * advantages

    # 组合损失项 (策略损失 + KL约束)
    total_loss =  per_pixel_kl * beta + policy_loss   # TODO: 这里需要进行修改

    # 计算有效区域掩码 (忽略填充区域)
    valid_mask = (gt_masks >= 0).float()  # 假设-1表示无效区域

    # 计算加权平均损失
    masked_loss = (total_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    return masked_loss, total_loss.mean(), per_pixel_kl.mean(), policy_loss.mean(), rewards.mean()

# 6. 定期更新参考模型的函数
def update_reference_model(model, ref_model, update_ratio=0.1):
    """软更新参考模型参数"""
    with torch.no_grad():
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            ref_param.data.copy_(update_ratio * param.data + (1 - update_ratio) * ref_param.data)
