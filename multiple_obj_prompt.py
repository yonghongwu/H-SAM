import numpy as np
import cv2
from typing import Tuple, List, Optional
import random

def get_random_small_target_prompts(mask: np.ndarray, 
                                   min_area: int = 10, 
                                   max_area: int = 1000) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    从语义分割mask中随机获取一个小目标的point prompt和box prompt
    
    Args:
        mask: 二值化的语义分割mask (H, W) 或 (H, W, 1)
        min_area: 小目标的最小面积阈值
        max_area: 小目标的最大面积阈值
    
    Returns:
        point_prompt: (x, y) 目标中心点坐标
        box_prompt: (x1, y1, x2, y2) 目标的边界框坐标
    """
    # 确保mask是二值化的
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # 二值化处理
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 寻找连通组件
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # 收集符合尺寸要求的小目标
    small_targets = []
    
    for label_id in range(1, num_labels):  # 跳过背景(label=0)
        # 获取当前连通组件的mask
        component_mask = (labels == label_id)
        area = np.sum(component_mask)
        
        # 筛选小目标
        if min_area <= area <= max_area:
            small_targets.append((label_id, component_mask, area))
    
    if not small_targets:
        print(f"未找到面积在 {min_area}-{max_area} 范围内的小目标")
        return None, None
    
    # 随机选择一个小目标
    selected_target = random.choice(small_targets)
    label_id, target_mask, area = selected_target
    
    # 获取目标的坐标点
    coords = np.where(target_mask)
    y_coords, x_coords = coords
    
    # 计算point prompt (目标中心点)
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    point_prompt = (center_x, center_y)
    
    # 计算box prompt (边界框)
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    box_prompt = (x_min, y_min, x_max, y_max)
    
    print(f"选中目标 - 标签ID: {label_id}, 面积: {area}")
    print(f"Point prompt: {point_prompt}")
    print(f"Box prompt: {box_prompt}")
    
    return point_prompt, box_prompt

def visualize_prompts(image: np.ndarray, 
                     mask: np.ndarray, 
                     point_prompt: Tuple[int, int], 
                     box_prompt: Tuple[int, int, int, int],
                     save_path: Optional[str] = None):
    """
    可视化point prompt和box prompt
    
    Args:
        image: 原始图像 (H, W, 3)
        mask: 语义分割mask
        point_prompt: (x, y) 点坐标
        box_prompt: (x1, y1, x2, y2) 边界框坐标
        save_path: 保存路径，如果为None则不保存
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原图和mask叠加
    axes[0].imshow(image)
    axes[0].imshow(mask, alpha=0.3, cmap='jet')
    axes[0].set_title('Original Image + Mask')
    axes[0].axis('off')
    
    # 显示prompts
    axes[1].imshow(image)
    
    # 绘制point prompt
    if point_prompt:
        axes[1].plot(point_prompt[0], point_prompt[1], 'ro', markersize=8, label='Point Prompt')
    
    # 绘制box prompt
    if box_prompt:
        x1, y1, x2, y2 = box_prompt
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2, label='Box Prompt')
        axes[1].add_patch(rect)
    
    axes[1].set_title('Prompts Visualization')
    axes[1].legend()
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    # 假设你有一个语义分割的mask
    height, width = 512, 512
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建一些小目标区域作为示例
    # 添加几个小的矩形区域
    mask[100:120, 150:180] = 255  # 小目标1
    mask[200:215, 300:320] = 255  # 小目标2
    mask[350:370, 100:130] = 255  # 小目标3
    mask[400:450, 400:480] = 255  # 大目标（可能不会被选中）
    
    # 添加一些噪声点
    for _ in range(20):
        y, x = random.randint(0, height-5), random.randint(0, width-5)
        mask[y:y+3, x:x+3] = 255
    
    # 获取随机小目标的prompts
    point_prompt, box_prompt = get_random_small_target_prompts(
        mask, 
        min_area=10, 
        max_area=1000
    )
    
    if point_prompt and box_prompt:
        print(f"\n成功获取prompts:")
        print(f"Point prompt (x, y): {point_prompt}")
        print(f"Box prompt (x1, y1, x2, y2): {box_prompt}")
        
        # 如果你有原始图像，可以进行可视化
        # image = cv2.imread('your_image.jpg')
        # visualize_prompts(image, mask, point_prompt, box_prompt)
