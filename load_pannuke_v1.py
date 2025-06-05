import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import os
import albumentations as A
from typing import Optional, Tuple, List

class PanNukeDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 folds: List[int] = [1, 2, 3],
                 transform: Optional[A.Compose] = None, 
                 task: str = 'both'):
        """
        Args:
            data_dir: 数据集根目录
            folds: 使用的fold列表，例如[1,2]表示使用fold1和fold2
            transform: 数据增强
            task: 'semantic', 'instance', 'both'
        """
        self.data_dir = data_dir
        self.folds = folds
        self.transform = transform
        self.task = task
        
        # 加载所有数据
        self.images, self.masks, self.types = self._load_data()
        
        print(f"Loaded {len(self.images)} samples from folds {folds}")
        
    def _load_data(self):
        """加载所有指定fold的数据"""
        all_images = []
        all_masks = []
        all_types = []
        
        for fold in self.folds:
            fold_dir = os.path.join(self.data_dir, f'fold_{fold}')
            
            # 加载images.npy
            images_path = os.path.join(fold_dir, 'images', f'fold{fold}', 'images.npy')
            masks_path = os.path.join(fold_dir, 'masks', f'fold{fold}', 'masks.npy')
            types_path = os.path.join(fold_dir, 'images', f'fold{fold}', 'types.npy')
            
            if not all(os.path.exists(p) for p in [images_path, masks_path, types_path]):
                raise FileNotFoundError(f"Missing data files in fold {fold}")
            
            # 加载数据
            fold_images = np.load(images_path)  # Shape: [N, H, W, 3]
            fold_masks = np.load(masks_path)    # Shape: [N, H, W, 6]
            fold_types = np.load(types_path)    # Shape: [N, 6]
            
            print(f"Fold {fold} - Images: {fold_images.shape}, Masks: {fold_masks.shape}, Types: {fold_types.shape}")
            
            # 添加到总列表
            for i in range(fold_images.shape[0]):
                all_images.append(fold_images[i])
                all_masks.append(fold_masks[i])
                all_types.append(fold_types[i])
        
        return all_images, all_masks, all_types
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 获取数据
        image = self.images[idx].copy()  # [H, W, 3]
        mask = self.masks[idx].copy()    # [H, W, 6]
        cell_type = self.types[idx].copy()  # [6]
        
        # 创建语义分割mask (取每个像素的最大概率类别)
        semantic_mask = np.argmax(mask, axis=-1)  # [H, W]
        
        # 创建实例分割mask
        instance_mask = self._create_instance_mask(mask)  # [H, W]
        
        # 数据增强
        if self.transform:
            # 准备albumentations的输入
            targets = {
                'image': image,
                'mask': semantic_mask,
                'instance_mask': instance_mask
            }
            augmented = self.transform(**targets)
            
            image = augmented['image']
            semantic_mask = augmented['mask']
            instance_mask = augmented['instance_mask']
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        semantic_mask = torch.from_numpy(semantic_mask).long()  # [H, W]
        instance_mask = torch.from_numpy(instance_mask).long()  # [H, W]
        cell_type = torch.from_numpy(cell_type).float()  # [6]
        
        # 根据任务返回不同的数据
        if self.task == 'semantic':
            return {
                'image': image,
                'semantic_mask': semantic_mask,
                'cell_type': cell_type
            }
        elif self.task == 'instance':
            return {
                'image': image,
                'instance_mask': instance_mask,
                'cell_type': cell_type
            }
        else:  # both
            return {
                'image': image,
                'semantic_mask': semantic_mask,
                'instance_mask': instance_mask,
                'cell_type': cell_type
            }
    
    def _create_instance_mask(self, masks):
        """从PanNuke的mask格式创建实例分割mask"""
        instance_mask = np.zeros(masks.shape[:2], dtype=np.int32)
        instance_id = 1
        
        # PanNuke的6个类别: 0-背景, 1-肿瘤细胞, 2-炎症细胞, 3-结缔组织细胞, 4-坏死细胞, 5-非肿瘤上皮细胞
        for class_idx in range(1, masks.shape[-1]):  # 跳过背景类(0)
            class_mask = (masks[:, :, class_idx] > 0.5).astype(np.uint8)
            
            if np.sum(class_mask) == 0:
                continue
            
            # 连通域分析分离不同实例
            num_labels, labels = cv2.connectedComponents(class_mask)
            
            for label_id in range(1, num_labels):
                # 过滤太小的连通域
                component_mask = (labels == label_id)
                if np.sum(component_mask) < 10:  # 过滤小于10个像素的区域
                    continue
                    
                instance_mask[component_mask] = instance_id
                instance_id += 1
        
        return instance_mask
    
    def get_class_names(self):
        cell_data = {
            "Neoplastic": {"count": 20414, "percentage": 43.38},
            "Non-Neoplastic Epithelial": {"count": 8380, "percentage": 17.81},
            "Inflammatory": {"count": 9840, "percentage": 20.69},
            "Connective": {"count": 5374, "percentage": 11.42},
            "Dead": {"count": 2547, "percentage": 5.41},
            "Non-Nuclei": {"count": 500, "percentage": 1.06}
        }
        return cell_data

def get_transforms(mode='train', image_size=256):
    """获取数据增强变换"""
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ], additional_targets={'instance_mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
        ], additional_targets={'instance_mask': 'mask'})

def create_pannuke_dataloaders(data_dir: str, 
                              train_folds: List[int] = [1, 2],
                              val_folds: List[int] = [3],
                              batch_size: int = 8,
                              num_workers: int = 4,
                              image_size: int = 256,
                              task: str = 'both'):
    """
    创建PanNuke数据加载器
    
    Args:
        data_dir: 数据集根目录
        train_folds: 训练用的fold列表
        val_folds: 验证用的fold列表  
        batch_size: 批次大小
        num_workers: 工作进程数
        image_size: 图像尺寸
        task: 任务类型
    """
    
    # 创建数据集
    train_dataset = PanNukeDataset(
        data_dir=data_dir,
        folds=train_folds,
        transform=get_transforms('train', image_size),
        task=task
    )
    
    val_dataset = PanNukeDataset(
        data_dir=data_dir,
        folds=val_folds,
        transform=get_transforms('val', image_size),
        task=task
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

# 使用示例和测试
if __name__ == "__main__":
    # 设置数据路径
    data_dir = "/database/wuyonghuang/pannuke/OpenDataLab___PanNuke/raw"
    
    # 创建数据加载器
    train_loader, val_loader, train_dataset, val_dataset = create_pannuke_dataloaders(
        data_dir=data_dir,
        train_folds=[1, 2],  # 使用fold1和fold2训练
        val_folds=[3],       # 使用fold3验证
        batch_size=4,
        task='both'
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Class names: {train_dataset.get_class_names()}")
    
    # 测试数据加载
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Semantic mask shape: {batch['semantic_mask'].shape}")
        print(f"Instance mask shape: {batch['instance_mask'].shape}")
        print(f"Cell type shape: {batch['cell_type'].shape}")
        
        # 打印一些统计信息
        print(f"Semantic mask unique values: {torch.unique(batch['semantic_mask'])}")
        print(f"Instance mask max value: {torch.max(batch['instance_mask'])}")
        
        if batch_idx == 0:  # 只看第一个batch
            break
