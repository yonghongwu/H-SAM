import numpy as np
import random
from PIL import Image
import cv2


def process_pannuke_for_sam(sample, mode='all_nuclei', target_category=None):
    """
    处理PanNuke数据集样本，生成SAM所需的prompt和目标掩码
    
    Args:
        sample: PanNuke数据集的一个样本
        mode: 'all_nuclei' 或 'specific_category'
        target_category: 当mode='specific_category'时指定目标类别 (0-4)
    
    Returns:
        dict: {
            'image': 原始RGB图像 (numpy array),
            'point_prompt': 点提示坐标 (x, y),
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
    
    # 生成point prompt (质心)
    coords = np.where(selected_instance > 0)
    if len(coords[0]) == 0:
        return None
    
    center_y = int(np.mean(coords[0]))
    center_x = int(np.mean(coords[1]))
    point_prompt = (center_x, center_y)
    
    # 生成box prompt (边界框)
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    box_prompt = (x_min, y_min, x_max, y_max)
    
    return {
        'image': image,
        'point_prompt': point_prompt,
        'box_prompt': box_prompt,
        'target_mask': target_mask,
        'prompt_mask': selected_instance,
        'selected_category': selected_category,
        'selected_instance_id': selected_idx
    }


# 使用示例
def demo_usage(sample, mode='all_nuclei'):
    # sample = dataset['fold1'][0]
    
    if mode == 'all_nuclei':
        # 示例1: 所有细胞核模式
        result1 = process_pannuke_for_sam(sample, mode='all_nuclei')
        if result1:
            print("模式1 - 所有细胞核:")
            print(f"包含的类别: {sample['categories']}")
            print(f"Point prompt: {result1['point_prompt']}")
            print(f"Box prompt: {result1['box_prompt']}")
            print(f"Selected category: {result1['selected_category']}")
            print(f"Target mask shape: {result1['target_mask'].shape}")
            # print(f"Target mask unique values: {np.unique(result1['target_mask'])}")
            return result1
    elif mode == 'specific_category':
        # 示例2: 特定类别模式 (假设选择类别0)
        result2 = process_pannuke_for_sam(sample, mode='specific_category', target_category=0)
        if result2:
            print("\n模式2 - 特定类别 (类别0):")
            print(f"包含的类别: {sample['categories']}")
            print(f"Point prompt: {result2['point_prompt']}")
            print(f"Box prompt: {result2['box_prompt']}")
            print(f"Selected category: {result2['selected_category']}")
            print(f"Target mask shape: {result2['target_mask'].shape}")
            print(f"Target mask unique values: {np.unique(result2['target_mask'])}")
            return result2
    else: raise ValueError("mode must be 'all_nuclei' or 'specific_category'")


# 批量处理函数
def create_sam_dataset(pannuke_dataset, fold_name='fold1', mode='all_nuclei', target_category=None):
    """
    批量处理PanNuke数据集创建SAM训练数据
    """
    sam_data = []
    fold_data = pannuke_dataset[fold_name]
    
    for i, sample in enumerate(fold_data):
        result = process_pannuke_for_sam(sample, mode=mode, target_category=target_category)
        if result is not None:
            sam_data.append(result)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(fold_data)} samples")
    
    print(f"Successfully processed {len(sam_data)} samples")
    return sam_data


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