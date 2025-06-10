import numpy as np
import random
from PIL import Image
import cv2


def process_pannuke_for_sam(sample, mode='all_nuclei', target_category=None, 
                           num_positive_points=1, num_negative_points=1):
    """
    处理PanNuke数据集样本，生成SAM所需的prompt和目标掩码
    
    Args:
        sample: PanNuke数据集的一个样本
        mode: 'all_nuclei' 或 'specific_category'
        target_category: 当mode='specific_category'时指定目标类别 (0-4)
        num_positive_points: 正样本点的数量
        num_negative_points: 负样本点的数量
    
    Returns:
        dict: {
            'image': 原始RGB图像 (numpy array),
            'point_prompts': 点提示坐标列表 [(x1, y1), (x2, y2), ...],
            'point_labels': 点标签列表 [1, 1, 0, 0, ...] (1=正样本, 0=负样本),
            'box_prompt': 边界框提示 (x1, y1, x2, y2),
            'target_mask': 目标分割掩码 (numpy array),
            'prompt_mask': 用于生成prompt的原始细胞核掩码,
            'selected_category': 选中的细胞核类别,
            'selected_instance_id': 选中的实例ID
        }
    """
    
    # 转换图像为numpy数组
    image = np.array(sample['image'])
    instances = sample['instances']
    categories = sample['categories']
    
    if mode == 'all_nuclei':
        # 模式1: 随机选择一个细胞核，目标是所有细胞核
        if len(instances) == 0:
            return None
        
        # 随机选择一个实例用于生成prompt
        selected_idx = random.randint(0, len(instances) - 1)
        selected_instance = np.array(instances[selected_idx])
        selected_category = categories[selected_idx]
        
        # 生成目标掩码：合并所有细胞核
        target_mask = np.zeros_like(selected_instance)
        for instance in instances:
            target_mask = np.logical_or(target_mask, np.array(instance))
        target_mask = target_mask.astype(np.uint8)
        
    elif mode == 'specific_category':
        # 模式2: 选择特定类别的细胞核，目标是该类别的所有细胞核
        if target_category is None:
            raise ValueError("target_category must be specified when mode='specific_category'")
        
        # 找到目标类别的所有实例
        target_indices = [i for i, cat in enumerate(categories) if cat == target_category]
        
        if len(target_indices) == 0:
            return None  # 该图像中没有目标类别的细胞核
        
        # 随机选择该类别的一个实例用于生成prompt
        selected_idx = random.choice(target_indices)
        selected_instance = np.array(instances[selected_idx])
        selected_category = categories[selected_idx]
        
        # 生成目标掩码：合并该类别的所有细胞核
        target_mask = np.zeros_like(selected_instance)
        for idx in target_indices:
            target_mask = np.logical_or(target_mask, np.array(instances[idx]))
        target_mask = target_mask.astype(np.uint8)
    
    else:
        raise ValueError("mode must be 'all_nuclei' or 'specific_category'")
    
    # 生成正样本点
    positive_points = []
    positive_coords = np.where(selected_instance > 0)
    
    if len(positive_coords[0]) == 0:
        return None
    
    # 第一个正样本点使用质心
    center_y = int(np.mean(positive_coords[0]))
    center_x = int(np.mean(positive_coords[1]))
    positive_points.append([center_x, center_y])
    
    # 如果需要更多正样本点，从选中的细胞核内随机采样
    if num_positive_points > 1:
        available_points = list(zip(positive_coords[1], positive_coords[0]))  # (x, y) format
        
        # 移除质心点，避免重复
        available_points = [p for p in available_points if p != (center_x, center_y)]
        
        # 随机选择额外的正样本点
        additional_points_needed = min(num_positive_points - 1, len(available_points))
        if additional_points_needed > 0:
            additional_points = random.sample(available_points, additional_points_needed)
            positive_points.extend([(x, y) for x, y in additional_points])
    
    # 生成负样本点
    negative_points = []
    if num_negative_points > 0:
        # 创建背景掩码（不属于目标掩码的区域）
        background_mask = 1 - target_mask
        background_coords = np.where(background_mask > 0)
        
        if len(background_coords[0]) > 0:
            available_bg_points = list(zip(background_coords[1], background_coords[0]))  # (x, y) format
            
            # 随机选择负样本点
            num_neg_points = min(num_negative_points, len(available_bg_points))
            if num_neg_points > 0:
                selected_bg_points = random.sample(available_bg_points, num_neg_points)
                negative_points.extend([(x, y) for x, y in selected_bg_points])
    
    # 合并所有点和标签
    all_points = positive_points + negative_points
    all_labels = [1] * len(positive_points) + [0] * len(negative_points)
    
    # 生成box prompt (基于选中的细胞核)
    y_min, y_max = np.min(positive_coords[0]), np.max(positive_coords[0])
    x_min, x_max = np.min(positive_coords[1]), np.max(positive_coords[1])
    box_prompt = (x_min, y_min, x_max, y_max)
    
    return {
        'image': image,
        'point_prompts': all_points,
        'point_labels': all_labels,
        'box_prompt': box_prompt,
        'target_mask': target_mask,
        'prompt_mask': selected_instance,
        'selected_category': selected_category,
        'selected_instance_id': selected_idx,
        'num_positive_points': len(positive_points),
        'num_negative_points': len(negative_points)
    }


# 使用示例
def demo_usage(sample, mode='all_nuclei', num_positive_points=1, num_negative_points=0):
    # sample = dataset['fold1'][0]
    
    if mode == 'all_nuclei':
        # 示例1: 所有细胞核模式，1个正样本点，0个负样本点
        result1 = process_pannuke_for_sam(sample, mode='all_nuclei', 
                                        num_positive_points=num_positive_points, num_negative_points=num_negative_points)
        if result1:
            print("模式1 - 所有细胞核:")
            print(f"包含的类别: {sample['categories']}")
            print(f"Point prompts: {result1['point_prompts']}")
            print(f"Point labels: {result1['point_labels']}")
            print(f"Box prompt: {result1['box_prompt']}")
            print(f"Selected category: {result1['selected_category']}")
            print(f"Actual positive points: {result1['num_positive_points']}")
            print(f"Actual negative points: {result1['num_negative_points']}")
            print(f"Target mask shape: {result1['target_mask'].shape}")
        return result1
    elif mode == 'specific_category':
        # 示例2: 特定类别模式，1个正样本点，3个负样本点
        result2 = process_pannuke_for_sam(sample, mode='specific_category', target_category=0,
                                        num_positive_points=num_positive_points, num_negative_points=num_negative_points)
        if result2:
            print("\n模式2 - 特定类别 (类别0):")
            print(f"包含的类别: {sample['categories']}")
            print(f"Point prompts: {result2['point_prompts']}")
            print(f"Point labels: {result2['point_labels']}")
            print(f"Box prompt: {result2['box_prompt']}")
            print(f"Selected category: {result2['selected_category']}")
            print(f"Actual positive points: {result2['num_positive_points']}")
            print(f"Actual negative points: {result2['num_negative_points']}")
            return result2
    else: raise ValueError("mode must be 'all_nuclei' or 'specific_category'")


# 批量处理函数
def create_sam_dataset(pannuke_dataset, fold_name='fold1', mode='all_nuclei', 
                      target_category=None, num_positive_points=1, num_negative_points=1):
    """
    批量处理PanNuke数据集创建SAM训练数据
    """
    sam_data = []
    fold_data = pannuke_dataset[fold_name]
    
    for i, sample in enumerate(fold_data):
        result = process_pannuke_for_sam(sample, mode=mode, target_category=target_category,
                                       num_positive_points=num_positive_points, 
                                       num_negative_points=num_negative_points)
        if result is not None:
            sam_data.append(result)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(fold_data)} samples")
    
    print(f"Successfully processed {len(sam_data)} samples")
    return sam_data

# 可视化函数（可选）
def visualize_prompts(result):
    """
    可视化生成的prompts
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(result['image'])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 图像 + 点提示
    axes[1].imshow(result['image'])
    for i, (point, label) in enumerate(zip(result['point_prompts'], result['point_labels'])):
        color = 'red' if label == 1 else 'blue'
        marker = 'o' if label == 1 else 'x'
        axes[1].plot(point[0], point[1], marker=marker, color=color, markersize=8)
    
    # 添加边界框
    box = result['box_prompt']
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                           linewidth=2, edgecolor='green', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title('Image + Prompts\n(Red O: Positive, Blue X: Negative)')
    axes[1].axis('off')
    
    # 目标掩码
    axes[2].imshow(result['target_mask'], cmap='gray')
    axes[2].set_title('Target Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import torch
    from datasets import load_dataset

    # 如果数据集有标准格式配置文件
    dataset = load_dataset("/database/wuyonghuang/PanNuke/")
    sample = dataset['fold1'][0]
    print(f"sample 中包含的键: {sample.keys()}")

    result1 = demo_usage(sample, mode='all_nuclei') # 'all_nuclei' or 'specific_category'
    result2 = create_sam_dataset(dataset, fold_name='fold1', mode='all_nuclei')

    # 构建 sam 模型, 进行测试
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"    # 这个是安装的时候写的, 不是相对路径
    sam2 = build_sam2(model_cfg, checkpoint)
    net = SAM2ImagePredictor(sam2)

    with torch.no_grad():
        net.set_image(sample['image'])

        masks, scores, _ = net.predict_in_training(
                        point_coords = [result1['point_prompt']],
                        point_labels = [1],
                        box = None,
                        multimask_output=True
                    )