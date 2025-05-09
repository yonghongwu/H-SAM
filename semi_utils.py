import torch
import random
import math
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


# 假设你已经有了一个原始的 Dataset 对象: original_dataset
# 例如:
# class MyOriginalDataset(Dataset):
#     def __init__(self, num_samples=100):
#         self.num_samples = num_samples
#         # 实际应用中这里会加载真实数据
#         self.data = [{'Image': f'img_{i}', 'mask': f'mask_{i}', 'mask2': f'mask2_{i}', 'casename': f'case_{i}'}
#                      for i in range(num_samples)]
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         if idx >= len(self):
#             raise IndexError("Index out of range")
#         return self.data[idx]
#
# original_dataset = MyOriginalDataset(num_samples=100) # 示例数据集


# --- 4. 创建新的 Dataset 类 ---
class SubsetDataset(Dataset):
    """
    通用的子集 Dataset 类，通过索引列表访问原始数据集。
    """
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # idx 是 SubsetDataset 内部的索引 (0 to len(self)-1)
        # 需要映射回 original_dataset 的真实索引
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]


class LabeledDataset(SubsetDataset):
    """
    有标注数据集，直接返回原始数据的完整字典。
    继承自 SubsetDataset，无需额外操作。
    """
    def __init__(self, original_dataset, indices):
        super().__init__(original_dataset, indices)
        print(f"Initialized Labeled Dataset with {len(self.indices)} samples.")


class UnlabeledDataset(SubsetDataset):
    """
    无标注数据集，只返回 'Image' 和 'casename'。
    """
    def __init__(self, original_dataset, indices):
        super().__init__(original_dataset, indices)
        print(f"Initialized Unlabeled Dataset with {len(self.indices)} samples.")

    def __getitem__(self, idx):
        # 先获取完整的原始数据项
        full_data_item = super().__getitem__(idx)
        # 只保留需要的键
        unlabeled_item = {
            'image': full_data_item['image'],
            'case_name': full_data_item['case_name'],
            'label': full_data_item['label'] # 被丢弃
        }
        return unlabeled_item


def semi_split(original_dataset=None, labeled_ratio=0.1):
    # --- 1. 定义参数 ---
    # 你可以指定有标注数据的比例或绝对数量
    
    # labeled_ratio = 0.1  # 例如，10% 的数据作为有标注数据
    # 或者
    # num_labeled_samples = 10 # 指定具体数量

    total_samples = len(original_dataset)

    if 'labeled_ratio' in locals():
        num_labeled_samples = math.ceil(total_samples * labeled_ratio) # 向上取整确保至少有1个（如果ratio>0）
    else:
        # 确保指定的数量不超过总数
        num_labeled_samples = min(num_labeled_samples, total_samples)

    num_unlabeled_samples = total_samples - num_labeled_samples

    print(f"Total samples: {total_samples}")
    print(f"Target labeled samples: {num_labeled_samples}")
    print(f"Target unlabeled samples: {num_unlabeled_samples}")

    # --- 2. 获取并打乱索引 ---
    all_indices = list(range(total_samples))
    random.seed(42) # 为了可复现性，设置随机种子
    random.shuffle(all_indices)

    # --- 3. 划分索引 ---
    labeled_indices = all_indices[:num_labeled_samples]
    unlabeled_indices = all_indices[num_labeled_samples:]

    return labeled_indices, unlabeled_indices


import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any

def get_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    计算二值掩码的最大连通区域。

    Args:
        mask (np.ndarray): 输入的二值掩码 (H, W)，数据类型应为 uint8，值为 0 或 非0。

    Returns:
        np.ndarray: 最大连通区域的二值掩码 (H, W)，数据类型为 uint8。
                    如果没有找到前景区域，则返回全零掩码。
    """
    # 查找连通组件
    # num_labels: 连通组件的数量（包括背景）
    # labels: 标记图像，每个像素标有其所属组件的 ID
    # stats: 每个组件的统计信息 [x, y, width, height, area]
    # centroids: 每个组件的质心 [cx, cy]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:  # 只有背景
        return np.zeros_like(mask, dtype=np.uint8)

    # 忽略背景标签 (标签 0)
    # 找到面积最大的前景组件的标签 ID
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 # 加1是因为我们忽略了背景标签0

    # 创建只包含最大连通区域的掩码
    largest_component_mask = np.where(labels == largest_component_label, 255, 0).astype(np.uint8)

    return largest_component_mask

def generate_prompts_from_mask(mask: np.ndarray) -> Dict[str, Any]:
    """
    从二值掩码（假定为单个目标）生成 SAM 的 prompts。

    Args:
        mask (np.ndarray): 输入的二值掩码 (H, W)，数据类型 uint8，值为 0 或 255。
                           应只包含一个主要的目标区域。

    Returns:
        Dict[str, Any]: 包含 'box_prompt' 和 'point_prompt' 的字典。
                        如果掩码为空，则返回 None。
    """
    if np.sum(mask) == 0: # 检查掩码是否为空
        return None

    # 1. 计算 Bounding Box Prompt
    # 找到非零像素的坐标
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None # 以防万一

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # SAM 通常期望 [x_min, y_min, x_max, y_max] 格式
    box_prompt = [int(x_min), int(y_min), int(x_max), int(y_max)]

    # 2. 计算 Point Prompt (使用质心)
    # 可以直接使用 findContours 计算质心，或者重新计算最大区域的质心
    # 为了简单起见，我们直接计算掩码的质心
    # 注意：cv2.moments 需要 uint8 类型的输入
    moments = cv2.moments(mask)
    if moments["m00"] == 0: # 避免除以零
         # 如果面积为0，可以取 BBox 的中心点
         center_x = (x_min + x_max) / 2
         center_y = (y_min + y_max) / 2
    else:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

    # SAM 期望 (x, y) 格式的坐标点
    point_prompt = [(int(center_x), int(center_y))] # 返回一个包含单个点的列表

    # SAM 还可以接受点标签 (前景点为1，背景点为0)
    point_labels = [1] # 假设这个点是前景点

    return {
        "box_prompt": box_prompt,
        "point_prompt": point_prompt,
        "point_labels": point_labels # 可选，但通常与点提示一起提供
    }


def process_segmentation_output(output_tensor: torch.Tensor,
                                background_class_id: int = 0,
                                apply_morphology: bool = False,
                                morph_kernel_size: int = 5) -> List[Dict[int, Dict[str, Any]]]:
    """
    处理语义分割模型的输出，为每个检测到的类别生成 prompts。

    Args:
        output_tensor (torch.Tensor): 模型输出张量，形状 (B, cls, H, W)。
        background_class_id (int): 背景类别的索引 ID。默认为 0。
        apply_morphology (bool): 是否在计算连通分量前应用形态学操作（开运算）去噪。
        morph_kernel_size (int): 形态学操作的核大小。

    Returns:
        List[Dict[int, Dict[str, Any]]]: 一个列表，每个元素对应批处理中的一张图像。
            每个元素是一个字典，键是检测到的类别 ID (非背景)，
            值是包含该类别最大连通区域信息的字典:
            {
                'largest_component_mask': np.ndarray (H, W), uint8,
                'prompts': {
                    'box_prompt': [x_min, y_min, x_max, y_max],
                    'point_prompt': [(x, y)],
                    'point_labels': [1]
                } or None if no object found
            }
    """
    results_batch = []
    batch_size = output_tensor.shape[0]

    # 将 logits/概率 转换为预测的类别图
    # pred_masks 的形状为 (B, H, W)
    pred_masks = torch.argmax(output_tensor, dim=1)

    for i in range(batch_size):
        pred_mask_np = pred_masks[i].cpu().numpy().astype(np.uint8)
        H, W = pred_mask_np.shape
        # results_image: Dict[int, Dict[str, Any]] = {}
        results_image = np.zeros((H, W), dtype=np.uint8)

        # 1. 判断模型识别出哪些类别 (忽略背景)
        unique_classes = np.unique(pred_mask_np)
        present_classes = [cls_id for cls_id in unique_classes if cls_id != background_class_id]

        print(f"Image {i}: Found classes (excluding background): {present_classes}")

        for cls_id in present_classes:
            # 为当前类别创建二值掩码
            class_mask = np.where(pred_mask_np == cls_id, 255, 0).astype(np.uint8)

            # 可选：应用形态学操作（开运算）去除小的噪声点/断开细连接
            if apply_morphology:
                kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
                # 开运算 = 腐蚀 + 膨胀
                class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

            # 2. 计算最大连通区域
            largest_component_mask = get_largest_connected_component(class_mask)
            results_image[largest_component_mask > 0] = cls_id
            # [[plt.imshow(results_image==i), plt.savefig(f'test_{i}.png')] for i in range(1, 9)]

    return results_image


# --- 示例用法 ---
if __name__ == '__main__':
    # 假设的模型输出 (示例数据)
    B, Cls, H, W = 2, 4, 100, 120 # 2张图片, 4个类 (0=背景, 1=人, 2=车, 3=树), 100x120 尺寸
    # 创建一个随机的 logits 张量 (实际中应来自你的模型)
    # output_logits = torch.randn(B, Cls, H, W)

    # 或者创建一个更结构化的示例
    output_logits = torch.zeros(B, Cls, H, W)
    # 图片 0: 添加一个人 (类 1) 和一辆车 (类 2)
    output_logits[0, 1, 20:50, 30:60] = 1.0 # 人区域
    output_logits[0, 1, 25:45, 35:55] = 2.0 # 人区域内部置信度更高
    output_logits[0, 2, 60:80, 70:100] = 1.5 # 车区域
    # 添加一些噪声点 (类 3)
    output_logits[0, 3, 10:15, 10:15] = 0.8

    # 图片 1: 添加一棵大树 (类 3) 和一个小的、分离的人区域 (类 1)
    output_logits[1, 3, 10:90, 10:110] = 1.8 # 大树
    output_logits[1, 1, 5:15, 5:15] = 1.2   # 小人区域 1
    output_logits[1, 1, 85:95, 105:115] = 1.1 # 小人区域 2 (会被开运算去除或只保留大的)

    # 处理输出
    # 可以尝试 apply_morphology=True 看看效果
    results_image = process_segmentation_output(output_logits, background_class_id=0, apply_morphology=False)
    # results_morph = process_segmentation_output(output_logits, background_class_id=0, apply_morphology=True, morph_kernel_size=5)

    plt.imshow(results_image); plt.savefig('test.png')


if __name__ == "__main__1":
    labeled_indices, unlabeled_indices = semi_split(original_dataset, labeled_ratio=0.1)

    print(f"Actual labeled indices count: {len(labeled_indices)}")
    print(f"Actual unlabeled indices count: {len(unlabeled_indices)}")

    # --- 5. 实例化新的 Dataset ---
    labeled_dataset = LabeledDataset(original_dataset, labeled_indices)
    unlabeled_dataset = UnlabeledDataset(original_dataset, unlabeled_indices)

    # --- 验证和使用 ---
    # 检查长度
    print(f"\nLength of labeled dataset: {len(labeled_dataset)}")
    print(f"Length of unlabeled dataset: {len(unlabeled_dataset)}")

    # 检查样本内容
    if len(labeled_dataset) > 0:
        print("\nSample from Labeled Dataset:")
        sample_labeled = labeled_dataset[0]
        print(sample_labeled)
        assert 'mask' in sample_labeled # 应该包含 mask

    if len(unlabeled_dataset) > 0:
        print("\nSample from Unlabeled Dataset:")
        sample_unlabeled = unlabeled_dataset[0]
        print(sample_unlabeled)
        assert 'mask' not in sample_unlabeled # 不应包含 mask
        assert 'Image' in sample_unlabeled # 应包含 Image

    # 现在你可以像使用普通 Dataset 一样使用这两个新的 dataset，例如创建 DataLoader
    # 注意：DataLoader 的 shuffle=True 会在每个 epoch 开始时再次打乱各自子集内部的顺序
    labeled_loader = DataLoader(labeled_dataset, batch_size=4, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=True)

    # 迭代 DataLoader 示例
    # print("\nIterating through labeled loader (first batch):")
    # for batch in labeled_loader:
    #     print(batch['casename']) # 打印批次中的 casenames
    #     # 处理有标注批次...
    #     break # 只看第一个批次

    # print("\nIterating through unlabeled loader (first batch):")
    # for batch in unlabeled_loader:
    #     print(batch['casename']) # 打印批次中的 casenames
    #     # 处理无标注批次...
    #     break # 只看第一个批次
