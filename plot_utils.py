import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image


def organize_prompts(prompts):
    organized_prompts = {}

    cls_names = list(prompts['class_prompts'].keys())
    cls_names.sort()

    # 遍历所有类别
    for class_id in cls_names:
        # 获取当前类别的所有 prompt
        class_prompts = prompts['class_prompts'][class_id]
        class_box_prompts = []
        class_point_prompts = []
        
        # 遍历当前类别的所有 prompt
        for prompt in class_prompts:
            if 'box_prompt' in prompt and prompt['box_prompt']:
                class_box_prompts.append(prompt['box_prompt'])
            
            if 'point_prompts' in prompt and prompt['point_prompts']:
                class_point_prompts.append(prompt['point_prompts'])
        
        # 将整理好的 prompts 存储到字典中
        organized_prompts[class_id] = {
            'box_prompts': class_box_prompts,
            'point_prompts': class_point_prompts
        }
    
    all_class_box_prompts = [organized_prompts[idx_cls]['box_prompts'] for idx_cls in sorted(list(organized_prompts.keys()))]
    
    all_class_point_prompts = [organized_prompts[idx_cls]['point_prompts'] for idx_cls in sorted(list(organized_prompts.keys()))]
    
    return organized_prompts, (all_class_box_prompts, all_class_point_prompts)


def draw_prompts_on_image(image, organized_prompts, class_id=2):
    """
    将 prompts 绘制到图像上
    
    参数:
        image: 输入图像，numpy 数组
        organized_prompts: 整理好的 prompts 字典
        class_id: 要绘制的类别 ID，默认为 2
    
    返回:
        绘制了 prompts 的图像
    """

    
    # 获取指定类别的 prompts
    if class_id not in organized_prompts:
        print(f"类别 {class_id} 不存在于 prompts 中")
        return vis_image
    
    class_prompts = organized_prompts[class_id]
    
    vis_images = []
    for box, points in zip(class_prompts['box_prompts'], class_prompts['point_prompts']):
        # 创建图像的副本，避免修改原图
        vis_image = image.copy()
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for point in points:
            x, y, label = point
            x, y = int(x), int(y)
        
            # 正样本点用绿色，负样本点用红色
            color = (255, 0, 0) if label == 1 else (0, 0, 255)
            
            # 绘制点
            cv2.circle(vis_image, (x, y), 5, color, -1)
            
        vis_images.append(vis_image)

    return vis_images


def plot_results_np(images: np.ndarray, prompts_img:np.ndarray, preds: np.ndarray, masks: np.ndarray, rewards:np.ndarray=None, save_paths=None):
    # 画同一行显示的三个图像(子图)
    if rewards is None:
        rewards = [0] * len(images)

    for idx in range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(images[idx]); axs[0].set_title('Image')
        axs[1].imshow(prompts_img[idx]); axs[1].set_title('prompt_vis')
        axs[2].imshow(preds[idx]); axs[2].set_title(f'Final Prediction: reward: {rewards[idx]:.03f}')
        axs[3].imshow(masks[idx]); axs[3].set_title('mask')
        plt.savefig(save_paths[idx]) if save_paths is not None else None
        plt.close()


def plot_points_on_image(img, points, boxes=None):
    """
    针对test_prompts获取的内容, 在图像上绘制点
    
    Args:
        image_path: 图像文件路径
        points: 点的列表，格式为[(x1, y1, c1), (x2, y2, c2), ...]
    """
    # 读取图像
    # img = Image.open(image_path)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    plt.imshow(img)

        # 绘制边界框（如果提供）
    if boxes is not None:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # 计算宽度和高度
            width = x2 - x1
            height = y2 - y1
            
            # 创建矩形框
            rect = patches.Rectangle((x1, y1), width, height,
                                   linewidth=2, edgecolor='blue', 
                                   facecolor='none', alpha=0.8)
            plt.gca().add_patch(rect)
            
            # 添加框的标签
            plt.text(x1, y1-5, f'Box {i+1}', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7),
                    fontsize=10, color='white', fontweight='bold')

    
    # 分离正负样本点
    positive_points = [(x, y) for x, y, c in points if c == 1]
    negative_points = [(x, y) for x, y, c in points if c == 0]
    
    # 绘制正样本点（绿色）
    if positive_points:
        pos_x, pos_y = zip(*positive_points)
        plt.scatter(pos_x, pos_y, c='green', marker='o', s=50, 
                   label='Positive samples', alpha=0.8)
    
    # 绘制负样本点（红色）
    if negative_points:
        neg_x, neg_y = zip(*negative_points)
        plt.scatter(neg_x, neg_y, c='red', marker='x', s=50, 
                   label='Negative samples', alpha=0.8)
    
    plt.legend()
    plt.title('Points on Image')
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()
    plt.savefig('test.png')
    plt.show()

# 使用示例
# points = [(100, 150, 1), (200, 300, 0), (50, 80, 1), (180, 250, 0)]
# plot_points_on_image('your_image.jpg', points)
# points = test_prompts['prompts'][0]['point_prompts']
# boxes = test_prompts['prompts'][0]['box_prompts']
# image = Image.fromarray(test_prompts['image'])
# plot_points_on_image(image, points, [boxes])


if __name__ == '__main__':

    # 使用示例
    organized_prompts, (class_box_prompts, class_point_prompts)  = organize_prompts(prompts)

    all_prompts_vis_imgs = []
    for i_cls in organized_prompts.keys():
        result_images = draw_prompts_on_image(image, organized_prompts, class_id=i_cls)
        all_prompts_vis_imgs.extend(result_images)
        # for idx in range(len(result_images)):
            # plt.figure(); plt.imshow(result_images[idx]); plt.title(f'cls{i_cls}-{idx}'); plt.savefig(f'./cls{i_cls}-{idx}.jpg'); plt.close()

    save_paths= [f'./cls{i_cls}-{idx}.jpg' for i_cls in organized_prompts.keys() for idx in range(len(result_images))]

    plot_results_np([image] * 15, all_prompts_vis_imgs, (current_logits > 0).cpu().numpy(), decoded_mask.cpu().numpy(), rewards=advantages, save_paths=save_paths)