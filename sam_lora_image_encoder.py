from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
from typing import List, Dict, Any, Tuple

from icecream import ic


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        prompt_encoder2_tensors = {}
        mask_decoder_tensors = {}
        mask_decoder2_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'prompt_encoder2' in key:
                prompt_encoder2_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if 'mask_decoder2' in key:
                mask_decoder2_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors,**prompt_encoder2_tensors, **mask_decoder_tensors, **mask_decoder2_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)

        # load prompt encoder
        prompt_encoder2_keys = [k for k in sam_keys if 'prompt_encoder2' in k]
        prompt_encoder2_values = [state_dict[k] for k in prompt_encoder2_keys]
        prompt_encoder2_new_state_dict = {k: v for k, v in zip(prompt_encoder2_keys, prompt_encoder2_values)}
        sam_dict.update(prompt_encoder2_new_state_dict)

        # load mask decoder
        mask_decoder2_keys = [k for k in sam_keys if 'mask_decoder2' in k]
        mask_decoder2_values = [state_dict[k] for k in mask_decoder2_keys]
        mask_decoder2_new_state_dict = {k: v for k, v in zip(mask_decoder2_keys, mask_decoder2_values)}
        sam_dict.update(mask_decoder2_new_state_dict)

        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size, gt=None, mode='train'):
        return self.sam(batched_input, multimask_output, image_size, gt=gt, mode=mode)


class LoRA_Sam2(LoRA_Sam):
    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.trunk.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model
    

class LoRA_Sam3(nn.Module):
    # 在你的模型类中
    def __init__(self, model, rank=4, target_modules=["Linear", "Conv2d"]):
        super().__init__()
        self.model = model
        
        # 添加LoRA到所有线性层
        self.add_lora_to_model(self.model, rank=rank, target_modules=target_modules)
        
        # 或者，只添加到特定层
        # self.add_lora_to_model(self.model, rank=rank, lora_layer_idxs=[0, 1, 4, 5])
        
        # 冻结原始模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只训练LoRA参数
        for param in self.w_As.parameters():
            param.requires_grad = True
        for param in self.w_Bs.parameters():
            param.requires_grad = True
    
    def add_lora_to_model(self, model, rank=4, lora_layer_idxs=None, target_modules=["Linear", "Conv2d"]):
        """
        为模型中的所有线性层添加LoRA参数
        
        参数:
            model: 要添加LoRA的模型
            rank: LoRA的秩
            lora_layer_idxs: 如果只想为特定层添加LoRA，提供层索引列表，None表示所有层
            target_modules: 要应用LoRA的模块类型列表
        """
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()
        self.lora_layers = {}
        
        # 用于跟踪已处理的层
        layer_counter = 0
        
        def add_lora_to_layer(module, name):
            nonlocal layer_counter
            
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                # 如果是目标模块类型(Linear或Conv2d等)
                if type(child).__name__ in target_modules:
                    # 检查是否在指定的层索引列表中，或者是否处理所有层
                    if lora_layer_idxs is None or layer_counter in lora_layer_idxs:
                        if isinstance(child, nn.Linear):
                            in_features = child.in_features
                            out_features = child.out_features
                            
                            w_a = nn.Linear(in_features, rank, bias=False)
                            w_b = nn.Linear(rank, out_features, bias=False)
                            
                            # 初始化为零以确保开始训练时不影响原始模型
                            nn.init.zeros_(w_b.weight)
                            
                            self.w_As.append(w_a)
                            self.w_Bs.append(w_b)
                            
                            # 创建LoRA包装层
                            lora_layer = _LoRA_Linear(child, w_a, w_b, rank)
                            
                            # 替换原始层
                            setattr(module, child_name, lora_layer)
                            
                            # 记录已修改的层
                            self.lora_layers[full_name] = lora_layer
                            
                        elif isinstance(child, nn.Conv2d):
                            # 对于卷积层的LoRA实现（如果需要）
                            in_channels = child.in_channels
                            out_channels = child.out_channels
                            
                            w_a = nn.Conv2d(in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False)
                            w_b = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                            
                            nn.init.zeros_(w_b.weight)
                            
                            self.w_As.append(w_a)
                            self.w_Bs.append(w_b)
                            
                            # 创建LoRA包装层（需要实现_LoRA_Conv2d类）
                            lora_layer = _LoRA_Conv2d(child, w_a, w_b)
                            
                            # 替换原始层
                            setattr(module, child_name, lora_layer)
                            
                            # 记录已修改的层
                            self.lora_layers[full_name] = lora_layer
                    
                    layer_counter += 1
                
                # 递归处理子模块
                add_lora_to_layer(child, full_name)
        
        # 从模型的根开始递归添加LoRA
        add_lora_to_layer(model, "")
        
        return model
    
    @staticmethod
    def set_trainable_para(model, original_linear=False):
        for module in model.modules():
            if isinstance(module, _LoRA_Linear):
                for param in module.w_a.parameters():
                    param.requires_grad = True
                for param in module.w_b.parameters():
                    param.requires_grad = True
                if original_linear:
                    # 设置原始线性层为可训练
                    for param in module.linear.parameters():
                        param.requires_grad = True
                    # 设置 LoRA 参数也为可训练（这通常是默认的，但也可以显式设置）

                    # 如果有 bias，也要处理（如果存在的话）
                    if module.linear.bias is not None:
                        module.linear.bias.requires_grad = True
        trainable_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_para = sum(p.numel() for p in model.parameters())
        model.para_desc = f'可训练参数量: {trainable_para}/{total_para} ≈ {trainable_para/total_para*100:.2f}%'
        print(model.para_desc)
    
    @staticmethod
    def save_lora_weights(model, save_path):
        lora_state_dict = {}
        lora_config = {}
        for name, module in model.named_modules():
            if isinstance(module, _LoRA_Linear):
                # 使用唯一标识名保存 lora_A 和 lora_B
                lora_state_dict[f"{name}.w_a.weight"] = module.w_a.weight
                lora_state_dict[f"{name}.w_b.weight"] = module.w_b.weight
                # 保存配置
                lora_config[name] = {
                    'r': module.r,  # LoRA 秩
                    'scaling': module.scaling,
                    'target_module': name  # 原始模块名称
                }

        torch.save({
            'state_dict': lora_state_dict,
            'config': lora_config
        }, save_path)
        print(f"LoRA weights saved successfully to {save_path}")
    
    @staticmethod
    def load_lora_weights(model, load_path):
        # lora_state_dict = torch.load(load_path)
        # model_state_dict = model.state_dict()

        checkpoint = torch.load(load_path, map_location='cpu')
        lora_state_dict = checkpoint['state_dict']
        lora_config = checkpoint.get('config', {})  # 如果没有配置，使用空字典

        # 计数器用于记录成功加载的模块数量
        loaded_modules = 0
        
        for name, module in model.named_modules():
            if isinstance(module, _LoRA_Linear):
                # 检查该模块的权重是否存在于保存的状态字典中
                if f"{name}.w_a.weight" in lora_state_dict and f"{name}.w_b.weight" in lora_state_dict:
                    module.w_a.weight.data.copy_(lora_state_dict[f"{name}.w_a.weight"])
                    module.w_b.weight.data.copy_(lora_state_dict[f"{name}.w_b.weight"])
                    
                    if name in lora_config:
                        config = lora_config[name]
                        # 可以根据需要设置其他参数，例如：
                        if hasattr(module, 'r') and 'r' in config:
                            module.r = config['r']
                    
                    loaded_modules += 1

        print(f"LoRA weights loaded successfully from {load_path}")


# LoRA包装类的实现
class _LoRA_Linear(nn.Module):
    def __init__(self, linear_layer, w_a, w_b, rank):
        super().__init__()
        self.linear = linear_layer
        self.w_a = w_a
        self.w_b = w_b
        self.scaling = 1.0  # 可选的缩放因子
        self.r = rank
        
    def forward(self, x):
        # 原始线性层的输出加上LoRA路径的输出
        return self.linear(x) + self.scaling * self.w_b(self.w_a(x))


class _LoRA_Conv2d(nn.Module):
    def __init__(self, conv_layer, w_a, w_b):
        super().__init__()
        self.conv = conv_layer
        self.w_a = w_a
        self.w_b = w_b
        self.scaling = 1.0
        
    def forward(self, x):
        # 原始卷积层的输出加上LoRA路径的输出
        return self.conv(x) + self.scaling * self.w_b(self.w_a(x))


# 特殊情况：处理QKV注意力层
class _LoRA_qkv(nn.Module):
    def __init__(self, qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v):
        super().__init__()
        self.qkv = qkv_linear
        self.w_a_q = w_a_q
        self.w_b_q = w_b_q
        self.w_a_v = w_a_v
        self.w_b_v = w_b_v
        self.scaling = 1.0
        
    def forward(self, x):
        # 原始QKV输出
        qkv_output = self.qkv(x)
        
        # LoRA路径：只修改Q和V部分
        batch_size = x.shape[0]
        q_dim = v_dim = self.qkv.out_features // 3
        
        # 计算LoRA的贡献
        q_lora = self.w_b_q(self.w_a_q(x)) * self.scaling
        v_lora = self.w_b_v(self.w_a_v(x)) * self.scaling
        
        # 将LoRA的贡献添加到QKV输出的相应部分
        qkv_output = qkv_output.view(batch_size, -1, 3, q_dim)
        qkv_output[:, :, 0, :] += q_lora.view(batch_size, -1, q_dim)  # 添加到Q部分
        qkv_output[:, :, 2, :] += v_lora.view(batch_size, -1, v_dim)  # 添加到V部分
        
        return qkv_output.view(batch_size, -1, 3 * q_dim)



if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    lora_sam = LoRA_Sam(sam, 4)
    lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
