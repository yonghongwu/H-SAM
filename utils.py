import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
import csv
from tqdm import tqdm
from skimage.metrics import adapted_rand_error
from scipy.spatial.distance import cdist
from scipy.ndimage.interpolation import zoom
from gen_prompt import generate_prompts_from_semantic_mask, get_prompt_preds, concatenate_masks_and_scores_v2, plot_results

def dice_coefficient(mask1, mask2):
    # Convert masks to boolean arrays
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate intersection and sum of the masks
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    
    # Calculate dice coefficient
    if sum_masks == 0:
        return 1.0
    return 2. * intersection / sum_masks


def hd95(mask1, mask2):
    # Ensure the masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Get the coordinates of the boundaries
    coords1 = np.array(np.nonzero(mask1)).T
    coords2 = np.array(np.nonzero(mask2)).T
    
    # Compute all pairwise distances between the boundaries
    distances = cdist(coords1, coords2, metric='euclidean')
    
    # Calculate the directed distances from mask1 to mask2 and vice versa
    min_distances_1_to_2 = np.min(distances, axis=1)
    min_distances_2_to_1 = np.min(distances, axis=0)
    
    # Combine and sort the distances
    all_min_distances = np.concatenate((min_distances_1_to_2, min_distances_2_to_1))
    sorted_distances = np.sort(all_min_distances)
    
    # Compute the 85th percentile of the sorted distances
    hd95_value = np.percentile(sorted_distances, 95)
    
    return hd95_value

from medpy.metric import binary
# 或
from surface_distance import compute_robust_hausdorff

def hd95(result, reference):
    try:
        if np.sum(result) > 0 and np.sum(reference) > 0:
            return binary.hd95(result, reference)
        else:
            return np.nan  # 或其他适当的值
    except Exception as e:
        print(f"HD95计算错误: {e}")
        return np.nan

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     elif pred.sum() > 0 and gt.sum() == 0:
#         return 1, 0
#     else:
#         return 0, 0
# import numpy as np
# from scipy.spatial.distance import directed_hausdorff
# ------------------------------------------------------------------------------------
# def calculate_dice(pred, gt):
#     """计算Dice系数"""
#     intersection = np.sum(pred * gt)
#     dice = (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)  # 加个小数防止除0
#     return dice

# def calculate_hd95(pred, gt):
#     """计算Hausdorff Distance 95%"""
#     pred_points = np.argwhere(pred > 0)
#     gt_points = np.argwhere(gt > 0)
#     if len(pred_points) == 0 or len(gt_points) == 0:  # 如果某个为空集, 返回最大值
#         return np.inf
#     forward_hd = directed_hausdorff(pred_points, gt_points)[0]
#     backward_hd = directed_hausdorff(gt_points, pred_points)[0]
#     return max(forward_hd, backward_hd)  # Hausdorff距离取最大值

# def calculate_metric_percase(pred, gt):
#     """结合Dice和HD95的计算"""
#     pred = (pred > 0).astype(np.uint8)  # 确保二值化
#     gt = (gt > 0).astype(np.uint8)
    
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = calculate_dice(pred, gt)
#         hd95 = calculate_hd95(pred, gt)
#         return dice, hd95
#     elif pred.sum() > 0 and gt.sum() == 0:
#         return 1, np.inf  # 如果gt没东西, HD95返回正无穷
#     else:
#         return 0, np.inf  # 如果pred和gt都没东西

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def calculate_metric_percase(pred, gt):
    # 快速二值化
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    pred_any = np.any(pred)
    gt_any = np.any(gt)
    
    if not pred_any and not gt_any:
        return 0.0, 0.0
    if pred_any and not gt_any:
        return 1.0, 0.0
    
    # Dice计算
    intersection = np.sum(pred & gt)
    dice = (2.0 * intersection) / (np.sum(pred) + np.sum(gt))
    
    # HD95优化计算
    if np.all(gt == pred):  # 完全匹配时快速返回
        return dice, 0.0
    
    # 提取边界点坐标（大幅减少计算量）
    pred_coords = np.argwhere(pred)
    gt_coords = np.argwhere(gt)
    
    # 使用Scipy的directed_hausdorff计算（双向取最大值）
    hd95 = max(
        directed_hausdorff(pred_coords, gt_coords)[0],
        directed_hausdorff(gt_coords, pred_coords)[0]
    )
    return dice, hd95


def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1, stage=2, mode='test', model='hsam', save_nii=False, args=None):
    args.num_prompts_per_class = 1
    lab = label.squeeze(0)
    net.model.eval()
    net.model.cuda()

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        dice_lst, hd95_lst = [], []
        for ind in tqdm(range(image.shape[0])):
            with torch.no_grad():
                if model == 'hsam':
                    slice = image[ind, :, :]
                    l = lab[ind, :, :]
                    x, y = slice.shape[0], slice.shape[1]
                    if x != input_size[0] or y != input_size[1]:
                        slice = zoom(slice, (input_size[0] / x, input_size[1] / y), order=3)  # previous using 0
                    new_x, new_y = slice.shape[0], slice.shape[1]  # [input_size[0], input_size[1]]
                    if new_x != patch_size[0] or new_y != patch_size[1]:
                        slice = zoom(slice, (patch_size[0] / new_x, patch_size[1] / new_y), order=3)  # previous using 0, patch_size[0], patch_size[1]
                    inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
                    inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
                    net.eval()

                    outputs1,outputs2,_,_ = net(inputs, multimask_output, patch_size[0], gt=None, mode='test')
                    if stage == 3:
                        output_masks = (outputs1['masks'] + outputs2['masks'])/2
                    elif stage == 2:
                        output_masks = outputs2['masks']

                    out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    out_h, out_w = out.shape

                    if x != out_h or y != out_w:
                        pred = zoom(out, (x / out_h, y / out_w), order=0)
                    else:
                        pred = out

                elif model == 'sam2':
                    net.model.eval()
                    adaptive_pool = nn.AdaptiveAvgPool2d(input_size)
                    slice = image[ind, :, :]
                    x, y = slice.shape[0], slice.shape[1]
                    if x != input_size[0] or y != input_size[1]:
                        slice = zoom(slice, (input_size[0] / x, input_size[1] / y), order=3)
                    
                    lab_slice = label[ind].astype(np.uint8)
                    x_lab, y_lab = lab_slice.shape[0], lab_slice.shape[1]
                    if x_lab != input_size[0] or y_lab != input_size[1]:
                        lab_slice = zoom(lab_slice, (input_size[0] / x_lab, input_size[1] / y_lab), order=0)    # 使用最近邻算法而不是插值
                    idx_image, idx_label = repeat(slice[..., None], 'h w c -> h w (repeat c)', repeat=3), lab_slice   # low_res_label_batch[0].cpu().numpy()

                    if idx_label.mean() == 0: 
                        # print("无前景; Skip")
                        continue

                    if isinstance(args.pos_point_num, tuple):
                        num_pos_points = np.random.randint(*args.pos_point_num, size=(1))[0]
                    else: num_pos_points = args.pos_point_num
                    if isinstance(args.neg_point_num, tuple):
                        num_neg_points = np.random.randint(*args.neg_point_num, size=(1))[0]
                    else: num_neg_points = args.neg_point_num

                    # 获取 prompts、二值化的多类别掩码
                    prompts = generate_prompts_from_semantic_mask(
                        idx_label,
                        class_ids=None,  # 处理所有类别
                        num_positive_points=num_pos_points,  # 别人在测试的时候是怎么设置的？
                        num_negative_points=num_neg_points,
                        num_prompts_per_class=args.num_prompts_per_class,  # note: 为什么每个类别生成3组prompt, 测试时是不是只使用一个比较合理
                        point_sampling_strategy="center_weighted",
                        box_noise_level=0.,
                        generate_box=True,
                        generate_points=True,
                        is_strict=args.point_strict
                    )

                    # # 按照decoded_mask的键的顺序进行排序, 然后将值重复三次再全部拼接起来。
                    decoded_mask = get_decoded_mask(prompts['decoded_mask'], num_prompts_per_class=args.num_prompts_per_class)
                    net.set_image(idx_image)
                    results = get_prompt_preds(net, prompts, prompt_mode=args.prompt_type, multimask_output=True, only_best_score_pred=True, only_save_best_prompt_pred=False)
                    all_logits, all_scores, category_indices = concatenate_masks_and_scores_v2(results['prompts_preds'], sort_keys=True)

                    criterion = nn.BCEWithLogitsLoss()
                    adaptive_pool = nn.AdaptiveAvgPool2d(decoded_mask.shape[-2:])

                    # sample_loss = criterion(adaptive_pool(all_logits), decoded_mask.float().cuda())
                    # sample_iou = compute_iou(adaptive_pool(all_logits), decoded_mask.float().cuda())

                    # 展示结果
                    # plot_results(results, image, label, plot_prompts_preds=True, save_path_lst=None)
                    out = np.zeros_like(idx_label, dtype=np.uint8) # size: (H, W)
                    for idx, i in enumerate(sorted(set(np.unique(idx_label).astype(np.uint8)) - set([0]))):
                        out += (adaptive_pool(all_logits) > 0).cpu().numpy().astype(np.uint8)[idx] * i
                    out_h, out_w = out.shape
                    if x != out_h or y != out_w:
                        pred = zoom(out, (x / out_h, y / out_w), order=0)
                    else:
                        pred = out

                    idx_label = zoom(idx_label, (out.shape[0] / idx_label.shape[0], out.shape[1] / idx_label.shape[1]), order=0)
                    gt = idx_label.astype(np.uint8)  # 直接转为整数类型

                    # 计算dice
                    dice_score = dice_coefficient(out, gt)
                    hd95_score = -1 # hd95(out, gt)
                    print(f"idx_slide: {ind}, cls_in_the_label:{list(set(np.unique(label).astype(np.uint8)) - set([0]))}, Dice Coefficient: {dice_score}, hd95_score: {hd95_score}")
                    dice_lst.append(dice_score), hd95_lst.append(hd95_score)

                prediction[ind] = pred
    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    metric_list_dice = []
    metric_list_hd = []
    for i in range(1, classes + 1):
        tmp = calculate_metric_percase(prediction == i, label == i)
        metric_list_dice.append(tmp[0])
        metric_list_hd.append(tmp[1])
        metric_list.append(tmp)
    if save_nii and test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    with open(test_save_path + '/' + 'dice' + ".csv",'a+',newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(metric_list_dice)
    return metric_list

def mask_latent_code_spatial_wise(latent_code, loss, percentile=1 / 3.0, random=False, loss_type='corr', if_detach=True, if_soft=False):
    '''
    given a latent code return a perturbed code where top % areas are masked 
    '''
    use_gpu = True if latent_code.device != torch.device('cpu') else False
    code = latent_code
    num_images = code.size(0)
    spatial_size = code.size(2) * code.size(3)
    H, W = code.size(2), code.size(3)

    gradient = torch.autograd.grad(loss, [code])[0]
    # mask gradient with largest response:
    spatial_mean = torch.mean(gradient, dim=1, keepdim=True)
    spatial_mean = spatial_mean.squeeze().view(num_images, spatial_size)

    # select the threshold at top XX percentile
    if random:
        percentile = np.random.rand() * percentile

    vector_thresh_percent = int(spatial_size * percentile)
    vector_thresh_value = torch.sort(spatial_mean, dim=1, descending=True)[
        0][:, vector_thresh_percent]

    vector_thresh_value = vector_thresh_value.view(
        num_images, 1).expand(num_images, spatial_size)

    if if_soft:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             0.5 * torch.rand_like(spatial_mean),
                             torch.ones_like(spatial_mean))
    else:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             torch.zeros_like(spatial_mean),
                             torch.ones_like(spatial_mean))

    mask_all = vector.view(num_images, 1, H, W)
    if not if_detach:
        masked_latent_code = latent_code * mask_all
    else:
        masked_latent_code = code * mask_all

    try:
        decoder_function.zero_grad()
    except:
        pass
    return masked_latent_code, mask_all

def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad


def get_decoded_mask(decoded_mask, num_prompts_per_class):
    # 按照decoded_mask的键的顺序进行排序, 然后将值重复三次再全部拼接起来。
    exist_keys = sorted(decoded_mask.keys())
    for key in exist_keys:
        # decoded_mask[key] = np.repeat(decoded_mask[key], 3, axis=0)
        decoded_mask[key] = torch.from_numpy(decoded_mask[key])[None].long().repeat(num_prompts_per_class, 1, 1)
    decoded_mask = torch.concat(list(decoded_mask.values()), dim=0)
    return decoded_mask


def process_list_A(A):
    """
    处理列表A, 将其转换为三个列表, 用于批量训练
    
    Args:
        A: 长度为L的列表, 每个元素是长度为L2的列表B, 
           B中每个元素是包含"point_prompts"和"box_prompts"的字典
    
    Returns:
        tuple: (coordinates_list, labels_list, boxes_list)
    """
    L = len(A)
    coordinates_list = []
    labels_list = []
    boxes_list = []
    
    for i in range(L):
        B = A[i]  # 长度为L2的列表
        L2 = len(B)
        
        # 获取第一个字典来确定N的大小
        if L2 > 0 and "point_prompts" in B[0]:
            N = len(B[0]["point_prompts"])
        else:
            N = 0
        
        # 初始化当前批次的数组
        coordinates = np.zeros((L2, N, 2))  # (L2, N, 2)
        labels = np.zeros((L2, N))          # (L2, N)
        boxes = np.zeros((L2, 4))           # (L2, 4)
        
        # 处理每个字典
        for j in range(L2):
            dict_item = B[j]
            
            # 处理point_prompts
            if "point_prompts" in dict_item:
                point_prompts = dict_item["point_prompts"]
                for k, (x, y, c) in enumerate(point_prompts):
                    if k < N:  # 防止索引越界
                        coordinates[j, k, 0] = x
                        coordinates[j, k, 1] = y
                        labels[j, k] = c
            
            # 处理box_prompts
            if "box_prompts" in dict_item:
                box_prompts = dict_item["box_prompts"]
                if len(box_prompts) >= 4:  # 确保有足够的坐标
                    boxes[j, :] = box_prompts[:4]  # 取前4个值
        
        coordinates_list.append(coordinates)
        labels_list.append(labels)
        boxes_list.append(boxes)
    
    return coordinates_list, labels_list, boxes_list
