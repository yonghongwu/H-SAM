import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict, Union, Optional


def generate_prompts_from_semantic_mask(
    mask: np.ndarray,
    class_ids: Optional[List[int]] = None,  # 要处理的类别ID列表, 如果为None则处理所有存在的类别
    num_positive_points: Union[int, Tuple[int, int]] = (1, 3),  # 每个类别生成的正样本点数量
    num_negative_points: Union[int, Tuple[int, int]] = (0, 2),  # 每个类别生成的负样本点数量
    num_prompts_per_class: int = 1,  # 每个类别生成的prompt组数
    point_sampling_strategy: str = "random",  # "random", "center_weighted", "edge_weighted"
    is_strict=False,
    box_noise_level: float = 0.05,  # 边界框的随机扰动程度(相对于对象尺寸的比例)
    generate_box: bool = True,
    generate_points: bool = True,
    return_visualization: bool = False,
    visualization_colors: Optional[Dict[int, Tuple[int, int, int]]] = None  # 可视化时每个类别的颜色
) -> Dict:
    """
    从多类别语义分割掩码生成带有随机性的点提示和框提示
    
    参数:
        mask: 语义分割掩码, 形状为 (H, W), 每个像素值代表一个类别ID
        class_ids: 要处理的类别ID列表, 如果为None则处理所有存在的类别
        num_positive_points: 每个类别生成的正样本点数量, 可以是固定数字或者范围元组 (min_points, max_points)
        num_negative_points: 每个类别生成的负样本点数量, 可以是固定数字或者范围元组 (min_points, max_points)
        num_prompts_per_class: 每个类别生成的prompt组数
        point_sampling_strategy: 点采样策略, 可以是 "random"、"center_weighted" 或 "edge_weighted"
        box_noise_level: 边界框扰动程度, 相对于对象尺寸的比例
        generate_box: 是否生成边界框提示
        generate_points: 是否生成点提示
        return_visualization: 是否返回可视化结果
        visualization_colors: 可视化时每个类别的颜色, 格式为 {class_id: (R, G, B)}
        is_strict: 负样本只在bounding box和mask不交集的地方选择。
    返回:
        包含生成的提示的字典, 格式为:
        {
            "class_prompts": {
                class_id_1: [
                    {  # 第一组prompt
                        "point_prompts": [(x1, y1, 1), (x2, y2, 1), ..., (x3, y3, 0), ...],  # 1表示正样本, 0表示负样本
                        "box_prompt": [x_min, y_min, x_max, y_max]
                    },
                    {  # 第二组prompt
                        ...
                    },
                    ...
                ],
                class_id_2: [
                    ...
                ],
                ...
            },
            "decoded_mask": {  # 解码后的二值掩码, 格式为 {class_id: binary_mask}
                class_id_1: binary_mask_1,
                class_id_2: binary_mask_2,
                ...
            },
            "visualization": 可视化图像(如果 return_visualization=True)
        }
    """
    if not (generate_box or generate_points):
        raise ValueError("至少需要生成一种提示(点或框)")
    
    h, w = mask.shape
    
    # 确定要处理的类别
    unique_classes = np.unique(mask)
    if class_ids is None:
        # 排除背景类(通常为0)
        class_ids = [cls_id for cls_id in unique_classes if cls_id > 0]
    else:
        # 只处理掩码中实际存在的类别
        class_ids = [cls_id for cls_id in class_ids if cls_id in unique_classes]
    
    if len(class_ids) == 0:
        raise ValueError("掩码中没有找到指定的类别")
    
    result = {"class_prompts": {}, "decoded_mask": {}}
    
    # 为每个类别生成提示
    for class_id in class_ids:
        # 为当前类别创建二值掩码
        binary_mask = (mask == class_id)

        result['decoded_mask'][int(class_id)] = binary_mask.astype(np.uint8)
        
        # 如果该类别在掩码中不存在, 则跳过
        if not np.any(binary_mask):
            continue
        
        # 初始化当前类别的多组prompt列表
        class_prompts = []
        
        # 找到掩码的轮廓
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
        else:  # OpenCV 3.x
            _, contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算边界框
        if len(contours) > 0:
            all_points = np.concatenate(contours).reshape(-1, 2)
            x_min, y_min = all_points.min(axis=0)
            x_max, y_max = all_points.max(axis=0)
        else:
            # 如果没有找到轮廓, 使用掩码中所有非零点
            y_indices, x_indices = np.where(binary_mask)
            if len(y_indices) == 0:
                continue  # 跳过这个类别
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 为当前类别生成多组prompt
        for prompt_idx in range(num_prompts_per_class):
            prompt_result = {}
            
            # 生成带有随机扰动的边界框
            if generate_box:
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                # 添加随机扰动
                noise_x = max(1, int(box_width * box_noise_level))
                noise_y = max(1, int(box_height * box_noise_level))
                
                # 确保扰动后的边界框仍在图像范围内
                x_min_noisy = max(0, x_min - random.randint(0, noise_x))
                y_min_noisy = max(0, y_min - random.randint(0, noise_y))
                x_max_noisy = min(w - 1, x_max + random.randint(0, noise_x))
                y_max_noisy = min(h - 1, y_max + random.randint(0, noise_y))
                
                prompt_result["box_prompt"] = [int(x_min_noisy), int(y_min_noisy), 
                                             int(x_max_noisy), int(y_max_noisy)]
            
            # 生成点提示
            if generate_points:
                point_prompts = []
                
                # 生成正样本点(前景点)
                if isinstance(num_positive_points, tuple):
                    n_pos_points = random.randint(num_positive_points[0], num_positive_points[1])
                else:
                    n_pos_points = num_positive_points
                
                for _ in range(n_pos_points):
                    # 生成前景点
                    if point_sampling_strategy == "random":
                        # 随机采样前景点
                        y_indices, x_indices = np.where(binary_mask)
                        if len(y_indices) == 0:
                            continue
                        idx = random.randint(0, len(y_indices) - 1)
                        x, y = int(x_indices[idx]), int(y_indices[idx])
                    
                    elif point_sampling_strategy == "center_weighted":
                        # 计算掩码的中心
                        center_y, center_x = np.mean(np.where(binary_mask), axis=1)
                        
                        # 计算到中心的距离权重
                        y_indices, x_indices = np.where(binary_mask)
                        if len(y_indices) == 0:
                            continue
                        
                        distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                        weights = 1 / (1 + distances)
                        weights /= weights.sum()
                        
                        # 根据权重采样
                        idx = np.random.choice(len(y_indices), p=weights)
                        x, y = int(x_indices[idx]), int(y_indices[idx])
                    
                    elif point_sampling_strategy == "edge_weighted":
                        # 创建边缘图像
                        edge = cv2.Canny((binary_mask * 255).astype(np.uint8), 100, 200)
                        
                        # 如果边缘检测失败或没有边缘, 回退到随机采样
                        if np.sum(edge) == 0:
                            y_indices, x_indices = np.where(binary_mask)
                            if len(y_indices) == 0:
                                continue
                            idx = random.randint(0, len(y_indices) - 1)
                            x, y = int(x_indices[idx]), int(y_indices[idx])
                        else:
                            # 从边缘采样
                            y_indices, x_indices = np.where(edge > 0)
                            idx = random.randint(0, len(y_indices) - 1)
                            x, y = int(x_indices[idx]), int(y_indices[idx])
                    
                    else:
                        raise ValueError(f"不支持的点采样策略: {point_sampling_strategy}")
                    
                    # 添加正样本点 (is_foreground=1)
                    point_prompts.append((int(x), int(y), 1))
                
                # # 生成负样本点(背景点)
                # if isinstance(num_negative_points, tuple):
                #     n_neg_points = random.randint(num_negative_points[0], num_negative_points[1])
                # else:
                #     n_neg_points = num_negative_points
                
                # for _ in range(n_neg_points):
                #     # 生成背景点 (当前类别掩码外的点)
                #     attempts = 0
                #     max_attempts = 100  # 防止无限循环
                    
                #     while attempts < max_attempts:
                #         x = random.randint(0, w - 1)
                #         y = random.randint(0, h - 1)
                #         # 确保点不在当前类别的掩码内
                #         if not binary_mask[y, x]:
                #             break
                #         attempts += 1
                    
                #     # 如果无法找到背景点, 则跳过
                #     if attempts >= max_attempts:
                #         continue
                    
                #     # 添加负样本点 (is_foreground=0)
                #     point_prompts.append((int(x), int(y), 0))

                    # 生成负样本点(背景点)
                if isinstance(num_negative_points, tuple):
                    n_neg_points = random.randint(num_negative_points[0], num_negative_points[1])
                else:
                    n_neg_points = num_negative_points
                
                for _ in range(n_neg_points):
                    # 生成背景点
                    attempts = 0
                    max_attempts = 100  # 防止无限循环
                    
                    while attempts < max_attempts:
                        if is_strict and generate_box:
                            # 严格模式：在边界框内的非目标区域采样
                            x_min, y_min, x_max, y_max = prompt_result["box_prompt"]
                            x = random.randint(x_min, x_max)
                            y = random.randint(y_min, y_max)
                            # 确保点在边界框内且不在当前类别的掩码内
                            if x_min <= x <= x_max and y_min <= y <= y_max and not binary_mask[y, x]:
                                break
                        else:
                            # 原有模式：在整个图像的非目标区域采样
                            x = random.randint(0, w - 1)
                            y = random.randint(0, h - 1)
                            # 确保点不在当前类别的掩码内
                            if not binary_mask[y, x]:
                                break
                        attempts += 1
                    
                    # 如果无法找到背景点，则跳过
                    if attempts >= max_attempts:
                        continue
                    
                    # 添加负样本点 (is_foreground=0)
                    point_prompts.append((int(x), int(y), 0))
                
                prompt_result["point_prompts"] = point_prompts
            
            # 将当前prompt添加到类别的prompt列表中
            class_prompts.append(prompt_result)
        
        # 将当前类别的所有prompt添加到总结果中
        result["class_prompts"][int(class_id)] = class_prompts
    
    # 可视化
    if return_visualization:
        # 创建彩色可视化图像
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 如果没有提供颜色映射, 则创建一个随机颜色映射
        if visualization_colors is None:
            visualization_colors = {}
            for class_id in class_ids:
                # 为每个类别生成一个随机颜色
                visualization_colors[class_id] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
        
        # 绘制每个类别的掩码
        for class_id in result["class_prompts"].keys():
            binary_mask = (mask == class_id)
            color = visualization_colors.get(class_id, (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
            vis_img[binary_mask] = color
        
        # 选择第一组prompt进行可视化
        for class_id, prompts_list in result["class_prompts"].items():
            if not prompts_list:
                continue
                
            # 使用第一组prompt进行可视化
            prompt = prompts_list[0]
            
            # 绘制边界框
            # if generate_box and "box_prompt" in prompt:
            #     x_min, y_min, x_max, y_max = prompt["box_prompt"]
            #     # 使用白色绘制边界框
            #     cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                
            #     # 添加类别标签
            #     cv2.putText(vis_img, f"Class {class_id}", (x_min, y_min - 5),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制点
            if generate_points and "point_prompts" in prompt:
                for x, y, is_fg in prompt["point_prompts"]:
                    # 正样本点用绿色, 负样本点用红色
                    color = (0, 255, 0) if is_fg else (0, 0, 255)
                    cv2.circle(vis_img, (x, y), 1, color, -1)
                    # 在点周围绘制一个白色环
                    # cv2.circle(vis_img, (x, y), 5, (255, 255, 255), 1)
        
        result["visualization"] = vis_img
    
    return result


def concatenate_masks_and_scores(category_dict):
    """
    将字典中所有类别的mask和score拼接在一起
    
    参数:
    category_dict: 字典, 键是类别名称(如'1'、'2'、'3'), 
                  值是包含字典的列表, 每个字典有'mask'和'score'两个键, 
                  'mask'对应的值是形状为(C, H, W)的tensor, 
                  'score'对应的值是tensor标量
    
    返回:
    tuple: (all_masks, all_scores)
           all_masks: 拼接后的所有mask, 形状为(N, C, H, W), 其中N是所有类别中所有mask的总数
           all_scores: 拼接后的所有score, 形状为(N,)
    """
    all_masks = []
    all_scores = []
    
    # 遍历所有类别
    for category in category_dict:
        # 获取当前类别的所有字典列表
        dict_list = category_dict[category]
        
        # 遍历当前类别中的所有字典
        for item in dict_list:
            # 将mask添加到列表中
            all_masks.append(item['mask'])
            
            # 将score添加到列表中
            all_scores.append(item['score'])
    
    # 检查是否有mask和score
    if not all_masks or not all_scores:
        return torch.tensor([]), torch.tensor([])
    
    # 将所有mask拼接成一个tensor, 形状为(N, C, H, W)
    all_masks = torch.stack(all_masks, dim=0)
    
    # 将所有score拼接成一个tensor, 形状为(N,)
    all_scores = torch.stack(all_scores, dim=0)
    
    return all_masks, all_scores


def concatenate_masks_and_scores_v2(category_dict, sort_keys=True):
    """
    将字典中所有类别的mask和score拼接在一起, 可以选择是否按类别键排序
    
    参数:
    category_dict: 字典, 键是类别名称(如'1'、'2'、'3'), 
                  值是包含字典的列表, 每个字典有'mask'和'score'两个键
    sort_keys: 布尔值, 是否按类别键排序, 默认为True
    
    返回:
    tuple: (all_masks, all_scores, category_indices)
           all_masks: 拼接后的所有mask, 形状为(N, C, H, W)
           all_scores: 拼接后的所有score, 形状为(N,)
           category_indices: 字典, 记录每个类别在拼接结果中的索引范围
    """
    all_masks = []
    all_scores = []
    category_indices = {}  # 记录每个类别在拼接结果中的索引范围
    
    # 确定迭代顺序
    if sort_keys:
        # 尝试将键转换为数字进行排序(如果键是数字字符串)
        try:
            categories = sorted(category_dict.keys(), key=lambda x: float(x))
        except (ValueError, TypeError):
            # 如果转换失败, 按字符串排序
            categories = sorted(category_dict.keys())
    else:
        categories = category_dict.keys()
    
    current_index = 0
    # 按确定的顺序遍历所有类别
    for category in categories:
        # 获取当前类别的所有字典列表
        dict_list = category_dict[category]
        
        # 记录当前类别的起始索引
        start_idx = current_index
        
        # 遍历当前类别中的所有字典
        for item in dict_list:
            # 将mask添加到列表中
            all_masks.append(item['mask'])
            
            # 将score添加到列表中
            all_scores.append(item['score'])
            
            current_index += 1
        
        # 记录当前类别的索引范围
        category_indices[category] = (start_idx, current_index)
    
    # 检查是否有mask和score
    if not all_masks or not all_scores:
        return torch.tensor([]), torch.tensor([]), {}
    
    # 将所有mask拼接成一个tensor, 形状为(N, C, H, W)
    all_masks = torch.stack(all_masks, dim=0)
    
    # 将所有score拼接成一个tensor, 形状为(N,)
    all_scores = torch.stack(all_scores, dim=0)
    
    return all_masks, all_scores, category_indices


def get_prompt_preds(predictor, prompts, prompt_mode='box', multimask_output=True, only_best_score_pred=True, only_save_best_prompt_pred=False, scale_prompt=1):
    """
    该函数用于根据给定的提示(predictor)和不同的提示模式(prompt_mode)执行分割任务, 并返回结果。函数支持对点提示(point prompts)、
    框提示(box prompts)或两者结合(both)进行分割预测, 并在多个预测结果中选择最佳的掩码(mask)和分数(score)。

    参数:
    predictor: 用于执行分割任务的预测器对象。
    prompt_mode (str): 提示模式, 可选值为'point'、'box'或'both', 决定使用点提示、框提示或两者结合进行分割。
    multimask_output (bool): 是否输出多个掩码结果。
    only_best_score_pred (bool): 是否只使用得分最高的掩码作为该prompt的最终结果。
    only_save_best_prompt_pred (bool): 每一个目标有多组prompt, 是否只保存得分最高的prompt的预测结果。

    返回:
    dict: 返回包含每个类别的最佳掩码和得分, 以及每个类别的不同prompt的预测结果 (如果only_save_best_prompt_pred为False)。

    """
    results = {}
    results['prompts_preds'] = {}

    for class_id, prompts_list in prompts["class_prompts"].items():
        best_score = -1
        best_mask = None
        results['prompts_preds'][class_id] = []
        
        # 尝试每一组prompt
        for prompt_idx, prompt in enumerate(prompts_list):
            # 准备点提示
            if "point_prompts" in prompt and len(prompt["point_prompts"]) > 0:
                point_coords = np.array([[x, y] for x, y, _ in prompt["point_prompts"]]) * scale_prompt
                point_labels = np.array([label for _, _, label in prompt["point_prompts"]])
            else:
                point_coords = None
                point_labels = None
            
            # 准备框提示
            if "box_prompt" in prompt:
                box = np.array(prompt["box_prompt"]) * scale_prompt
            else:
                box = None
            
            if prompt_mode == 'point':
                new_point_coords = point_coords 
                new_point_labels = point_labels
                new_box = None
            elif prompt_mode == 'box':
                new_point_coords = None
                new_point_labels = None
                new_box = box
            elif prompt_mode == 'both':
                new_point_coords = point_coords
                new_point_labels = point_labels
                new_box = box
            else: raise ValueError

            masks, scores, _ = predictor.predict_in_training(
                point_coords = new_point_coords,
                point_labels = new_point_labels,
                box = new_box,
                multimask_output=multimask_output
            )
            
            # 找出最高分数的掩码
            if len(scores) > 0:
                if only_best_score_pred:
                    best_mask_idx = torch.argmax(scores) if isinstance(scores, torch.Tensor) else np.argmax(scores)
                else:
                    best_mask_idx = np.random.randint(len(scores))
                current_score = scores[best_mask_idx]
                
                # 如果这组prompt的结果比之前的更好, 则更新最佳结果
                if current_score > best_score:
                    best_score = current_score
                    best_mask = masks[best_mask_idx]
                    # print(f"类别 {class_id}, Prompt组 {prompt_idx+1}: 新的最佳分数 {best_score:.4f}")
                
                # 保存所有prompt的结果, 但每个prompt都只保留一个score最好(或者随机)的prediction
                if not only_save_best_prompt_pred:
                    results['prompts_preds'][class_id].append({
                        "mask": masks[best_mask_idx],
                        "score": scores[best_mask_idx]
                    })
        
        # 存储最佳结果
        if best_mask is not None:
            results[class_id] = {
                "mask": best_mask,
                "score": best_score
            }
    return results


def plot_results(results, image: np.ndarray, mask: np.ndarray, plot_prompts_preds=False, save_path_lst=None):
    # 画同一行显示的三个图像(子图)
    with torch.no_grad():
        results_prompts_preds = results.pop('prompts_preds')
        for idx, i in enumerate(results.keys()):
            tmp = (results[i]['mask'].cpu().numpy() > 0).astype(int)
            if idx == 0:
                final_pred = np.zeros_like(tmp)
            final_pred += (idx+1) * tmp
        
        plt.figure()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image); axs[0].set_title('Image')
        axs[1].imshow(mask); axs[2].set_title('mask')
        axs[2].imshow(final_pred); axs[1].set_title('Final Prediction')
        plt.savefig(save_path_lst[0]) if save_path_lst is not None else None
        
        # -------------------------------------------------------------------
        # 对于 保存了每一个prompt的预测结果的情况, 需要通过下面的方式打印出结果
        if plot_prompts_preds:
            ith_prompt_pred_in_cls = 0    # 获取每一个类别中的第一个prompt的预测结果
            for idx, i in enumerate(results_prompts_preds.keys()):
                tmp = (results_prompts_preds[i][ith_prompt_pred_in_cls]['mask'].cpu().numpy() > 0).astype(int)
                if idx == 0:
                    final_pred = np.zeros_like(tmp)
                final_pred += (idx+1) * tmp
            
            plt.figure()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(image); axs[0].set_title('Image')
            axs[1].imshow(final_pred); axs[1].set_title('Final Prediction')
            axs[2].imshow(mask); axs[2].set_title('mask')
            plt.savefig(save_path_lst[-1]) if save_path_lst is not None else None



if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    image = np.load('./raw.npy')
    img_H, img_W, chns = image.shape
    mask = np.load('./mask.npy')
    gt_H, gt_W = mask.shape

    # 构建模型
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"    # 这个是安装的时候写的
    sam2 = build_sam2(model_cfg, checkpoint)
    predictor = SAM2ImagePredictor(sam2)

    # 设置图像
    predictor.set_image(image)

    # 从语义掩码生成提示
    semantic_mask = mask
    prompts = generate_prompts_from_semantic_mask(
        semantic_mask,
        class_ids=None,  # 处理所有类别
        num_positive_points=(2, 3),
        num_negative_points=(4, 6),
        num_prompts_per_class=3,  # 每个类别生成3组prompt
        point_sampling_strategy="center_weighted",
        box_noise_level=0.1,
        generate_box=True,
        generate_points=True
    )

    # 为每个类别执行分割, 使用多组prompt并选择最佳结果: 给定 prompt结果、predictor、prompt_mode、
    results = get_prompt_preds(predictor, prompts, prompt_mode='box', multimask_output=True, only_best_score_pred=True, only_save_best_prompt_pred=False)

    plot_results(results, image, mask, plot_prompts_preds=False, save_path_lst=None)