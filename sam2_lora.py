import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import types
from typing import List, Dict, Any, Tuple
import warnings

# Assume SAM2Base class and its dependencies are defined elsewhere
# from your_module import SAM2Base, Sam # Example import

# --- Helper Function to get nested modules ---
def _get_module_by_name(module: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Gets a submodule and its parent using a dotted name string."""
    names = name.split('.')
    parent = module
    for i, sub_name in enumerate(names[:-1]):
        try:
            parent = getattr(parent, sub_name)
        except AttributeError:
            raise AttributeError(f"Module {module.__class__.__name__} has no attribute {'.'.join(names[:i+1])}")
    target_name = names[-1]
    try:
        target = getattr(parent, target_name)
    except AttributeError:
        raise AttributeError(f"Module {parent.__class__.__name__} has no attribute {target_name} (full path: {name})")
    return parent, target_name

# --- LoRA Layer Implementation ---
class LoRALinear(nn.Module):
    """Applies LoRA to a nn.Linear layer."""
    def __init__(self, original_linear: nn.Linear, r: int, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        if r > 0:
            self.lora_A = nn.Linear(self.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            self.reset_lora_parameters() # Initialize LoRA weights
    def reset_lora_parameters(self):
        if self.r > 0:
            # Initialize A with Kaiming uniform (He initialization)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            # Initialize B with zeros
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen weights)
        original_output = self.original_linear(x)

        # LoRA path (trainable weights)
        if self.r > 0:
            lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return original_output + lora_output
        else:
            # If r=0, LoRA is disabled for this layer
            return original_output

    def extra_repr(self) -> str:
        return f'r={self.r}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout.p}, in_features={self.in_features}, out_features={self.out_features}'


# --- General LoRA Adapter ---
class LoRA_Adapter(nn.Module):
    """
    Applies Low-Rank Adaptation (LoRA) to specified linear layers within a given model.

    Args:
        model: The base PyTorch model (nn.Module) to adapt.
        r: The rank of the LoRA decomposition.
        lora_alpha: The scaling factor for the LoRA update (alpha in the paper).
        lora_target_modules: A list of names (strings) of the nn.Linear layers
                             within the model to apply LoRA to. Use dot notation
                             for nested modules (e.g., 'encoder.layer.0.attention.qkv').
        lora_dropout: Dropout probability for the LoRA path.
        verbose: If True, print information about replaced layers.
    """
    def __init__(self,
                 model: nn.Module,
                 r: int,
                 lora_alpha: float = 1.0,
                 lora_target_modules: List[str] = None,
                 lora_dropout: float = 0.0,
                 verbose: bool = False):
        super().__init__()
        assert r >= 0, "LoRA rank 'r' must be non-negative."
        self.model = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules else []
        self._lora_layers = nn.ModuleDict() # Store LoRA layers for easy access/saving

        # Freeze all parameters of the original model first
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA to target modules
        if self.r > 0 and self.lora_target_modules:
            self._apply_lora(verbose)

    def _apply_lora(self, verbose: bool):
        """Finds and replaces target linear layers with LoRALinear layers."""
        replaced_count = 0
        for target_name in self.lora_target_modules:
            try:
                parent_module, attr_name = _get_module_by_name(self.model, target_name)
                original_module = getattr(parent_module, attr_name)

                if isinstance(original_module, nn.Linear):
                    lora_layer = LoRALinear(
                        original_module,
                        self.r,
                        self.lora_alpha,
                        self.lora_dropout
                    )
                    setattr(parent_module, attr_name, lora_layer)
                    # Store reference for saving/loading
                    self._lora_layers[target_name.replace('.', '_')] = lora_layer # Use safe key name
                    replaced_count += 1
                    if verbose:
                        print(f"Applied LoRA to: {target_name}")
                else:
                    warnings.warn(f"Target module '{target_name}' is not nn.Linear. Skipping.")

            except AttributeError as e:
                warnings.warn(f"Could not find target module '{target_name}': {e}. Skipping.")
            except Exception as e:
                 warnings.warn(f"Failed to apply LoRA to '{target_name}': {e}. Skipping.")

        if replaced_count == 0 and self.lora_target_modules:
             warnings.warn("LoRA specified but no target modules were replaced. Check `lora_target_modules` names.")
        elif verbose:
             print(f"Successfully applied LoRA to {replaced_count} linear layers.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Passes input through the adapted model."""
        return self.model(*args, **kwargs)
    
    def reset_lora_parameters(self) -> None:
        """Resets all LoRA parameters (A to Kaiming, B to zeros)."""
        for lora_layer in self._lora_layers.values():
            if isinstance(lora_layer, LoRALinear):
                lora_layer.reset_lora_parameters()
        print("Reset all LoRA parameters.")
    
    def replace_self_attr_2_sam2(self):
        pass

    @staticmethod
    def set_trainable_para(model, original_linear=False):
        for module in model.modules():
            if isinstance(module, LoRALinear):
                for param in module.lora_A.parameters():
                    param.requires_grad = True
                for param in module.lora_B.parameters():
                    param.requires_grad = True
                if original_linear:
                    # 设置原始线性层为可训练
                    for param in module.original_linear.parameters():
                        param.requires_grad = True
                    # 设置 LoRA 参数也为可训练（这通常是默认的，但也可以显式设置）

                    # 如果有 bias，也要处理（如果存在的话）
                    if module.original_linear.bias is not None:
                        module.original_linear.bias.requires_grad = True
        trainable_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_para = sum(p.numel() for p in model.parameters())
        model.para_desc = f'可训练参数量: {trainable_para}/{total_para} ≈ {trainable_para/total_para*100:.2f}%'
        print(model.para_desc)
    
    @staticmethod
    def save_lora_weights(model, save_path):
        lora_state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                # 使用唯一标识名保存 lora_A 和 lora_B
                lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.weight
                lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.weight
        torch.save(lora_state_dict, save_path)
        print(f"LoRA weights saved successfully to {save_path}")
    
    @staticmethod
    def load_lora_weights(model, load_path):
        lora_state_dict = torch.load(load_path)
        model_state_dict = model.state_dict()

        # 将 LoRA 权重加载到对应的位置
        for key in lora_state_dict:
            if key in model_state_dict:
                model_state_dict[key].copy_(lora_state_dict[key])
            else:
                print(f"警告: 权重 {key} 没有在模型中找到，跳过加载。")
        print(f"LoRA weights loaded successfully from {load_path}")


class TransparentLoRAWrapper:
    """
    透明包装 LoRA 适配器的类，自动将未找到的属性和方法转发到原始模型
    """
    def __init__(self, lora_adapter):
        """
        初始化包装器
        
        Args:
            lora_adapter: 已经创建好的 LoRA 适配器实例
        """
        self._lora_adapter = lora_adapter
        self._original_model = lora_adapter.model
        
    def __getattr__(self, name):
        """
        当属性在当前对象中找不到时，尝试按以下顺序查找：
        1. 在 LoRA 适配器中查找
        2. 在原始模型中查找
        """
        # 首先尝试从 LoRA 适配器获取
        if hasattr(self._lora_adapter, name):
            attr = getattr(self._lora_adapter, name)
            return attr
            
        # 然后尝试从原始模型获取
        if hasattr(self._original_model, name):
            return getattr(self._original_model, name)
            
        # 如果都找不到，抛出 AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __call__(self, *args, **kwargs):
        """
        转发调用到 LoRA 适配器
        """
        return self._lora_adapter(*args, **kwargs)


def add_lora_to_sam2(
    model,  # SAM2 模型
    r: int = 4,
    lora_alpha: float = 1.0,
    lora_target_modules: List[str] = None,
    lora_dropout: float = 0.0,
    verbose: bool = False
):
    """
    为SAM2模型添加LoRA适配器，直接修改模型内部结构而非封装。
    
    Args:
        model: SAM2模型实例
        r: LoRA的秩
        lora_alpha: LoRA缩放因子
        lora_target_modules: 要应用LoRA的目标模块名称列表
        lora_dropout: LoRA路径的dropout概率
        verbose: 是否打印详细信息
        
    Returns:
        修改后的原始模型（添加了LoRA功能）
    """
    assert r >= 0, "LoRA rank 'r' must be non-negative."
    
    # 存储LoRA层的字典，添加为模型的属性
    model._lora_layers = nn.ModuleDict()
    
    # 冻结原始模型参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 应用LoRA到目标模块
    if r > 0 and lora_target_modules:
        replaced_count = 0
        for target_name in lora_target_modules:
            try:
                parent_module, attr_name = _get_module_by_name(model, target_name)
                original_module = getattr(parent_module, attr_name)

                if isinstance(original_module, nn.Linear):
                    lora_layer = LoRALinear(
                        original_module,
                        r,
                        lora_alpha,
                        lora_dropout
                    )
                    # 替换原始模块
                    setattr(parent_module, attr_name, lora_layer)
                    # 存储引用以便保存/加载
                    model._lora_layers[target_name.replace('.', '_')] = lora_layer
                    replaced_count += 1
                    if verbose:
                        print(f"Applied LoRA to: {target_name}")
                else:
                    warnings.warn(f"Target module '{target_name}' is not nn.Linear. Skipping.")
            except Exception as e:
                warnings.warn(f"Failed to apply LoRA to '{target_name}': {e}. Skipping.")
                
        if replaced_count == 0 and lora_target_modules:
            warnings.warn("LoRA specified but no target modules were replaced.")
        elif verbose:
            print(f"Successfully applied LoRA to {replaced_count} linear layers.")
    
    # 添加保存LoRA参数的方法
    def save_lora_parameters(self, filename: str) -> None:
        """保存只有LoRA参数的状态字典到文件"""
        assert filename.endswith(".pt") or filename.endswith('.pth'), \
            "Filename must end with .pt or .pth"

        lora_state_dict = {}
        for module_key, lora_layer in self._lora_layers.items():
            if isinstance(lora_layer, LoRALinear) and lora_layer.r > 0:
                original_name = module_key.replace('_', '.')
                lora_state_dict[f"{original_name}.lora_A.weight"] = lora_layer.lora_A.weight
                lora_state_dict[f"{original_name}.lora_B.weight"] = lora_layer.lora_B.weight

        if not lora_state_dict:
            warnings.warn("No LoRA parameters found to save.")
            return

        torch.save(lora_state_dict, filename)
        print(f"Saved LoRA parameters to {filename}")
    
    # 添加加载LoRA参数的方法
    def load_lora_parameters(self, filename: str, strict: bool = True) -> None:
        """从文件加载LoRA参数"""
        assert filename.endswith(".pt") or filename.endswith('.pth'), \
            "Filename must end with .pt or .pth"

        try:
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        except FileNotFoundError:
            print(f"Error: LoRA parameter file not found at {filename}")
            return
        except Exception as e:
            print(f"Error loading LoRA parameters from {filename}: {e}")
            return

        loaded_count = 0
        mismatched_keys = []

        for module_key, lora_layer in self._lora_layers.items():
            if isinstance(lora_layer, LoRALinear) and lora_layer.r > 0:
                original_name = module_key.replace('_', '.')
                key_a = f"{original_name}.lora_A.weight"
                key_b = f"{original_name}.lora_B.weight"

                if key_a in state_dict:
                    if lora_layer.lora_A.weight.shape == state_dict[key_a].shape:
                        lora_layer.lora_A.weight = Parameter(state_dict[key_a].to(lora_layer.lora_A.weight.device))
                        loaded_count += 1
                    else:
                        mismatched_keys.append(f"{key_a} (shape mismatch)")
                elif strict:
                    mismatched_keys.append(f"{key_a} (missing)")

                if key_b in state_dict:
                    if lora_layer.lora_B.weight.shape == state_dict[key_b].shape:
                        lora_layer.lora_B.weight = Parameter(state_dict[key_b].to(lora_layer.lora_B.weight.device))
                        loaded_count += 1
                    else:
                        mismatched_keys.append(f"{key_b} (shape mismatch)")
                elif strict:
                    mismatched_keys.append(f"{key_b} (missing)")

        if mismatched_keys and strict:
            raise RuntimeError("Failed to load LoRA parameters due to missing or mismatched keys.")
        
        if loaded_count > 0:
            print(f"Successfully loaded {loaded_count // 2} LoRA layer parameters")
        else:
            warnings.warn(f"No matching LoRA parameters found to load.")
    
    # 添加重置LoRA参数的方法
    def reset_lora_parameters(self) -> None:
        """重置所有LoRA参数"""
        for lora_layer in self._lora_layers.values():
            if isinstance(lora_layer, LoRALinear):
                lora_layer.reset_lora_parameters()
        print("Reset all LoRA parameters.")
    
    # 添加打印可训练参数的方法
    def print_trainable_parameters(self) -> None:
        """打印可训练参数数量和百分比"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable LoRA parameters: {trainable_params:,}")
        if total_params > 0:
            print(f"Percentage trainable: {100 * trainable_params / total_params:.4f}%")
    
    # 将方法绑定到模型实例
    model.save_lora_parameters = types.MethodType(save_lora_parameters, model)
    model.load_lora_parameters = types.MethodType(load_lora_parameters, model)
    model.reset_lora_parameters = types.MethodType(reset_lora_parameters, model)
    model.print_trainable_parameters = types.MethodType(print_trainable_parameters, model)
    
    return model


if __name__ == "__main__":
    # --- Example Usage with SAM2Base ---

    # Assume sam2_model is an instance of your SAM2Base class
    # sam2_model = SAM2Base(...) # Load or initialize your model
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "/database/wuyonghuang/hsam_code/sam2-main/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"    # 这个是安装的时候写的, 不是相对路径
    sam2 = build_sam2(model_cfg, checkpoint)

    # Define which linear layers to target with LoRA
    # You need to find the exact names by inspecting the model structure (print(sam2_model))
    target_modules = [
        # Example targets in the image encoder trunk blocks (e.g., first 2 blocks)
        "image_encoder.trunk.blocks.0.attn.qkv",
        "image_encoder.trunk.blocks.0.attn.proj",
        "image_encoder.trunk.blocks.0.mlp.layers.0", # First MLP layer
        "image_encoder.trunk.blocks.0.mlp.layers.1", # Second MLP layer
        "image_encoder.trunk.blocks.1.attn.qkv",
        "image_encoder.trunk.blocks.1.attn.proj",

        # Example targets in memory attention (e.g., first layer)
        "memory_attention.layers.0.self_attn.q_proj",
        "memory_attention.layers.0.self_attn.k_proj", # Apply to K? Optional.
        "memory_attention.layers.0.self_attn.v_proj",
        "memory_attention.layers.0.self_attn.out_proj",
        "memory_attention.layers.0.linear1",
        "memory_attention.layers.0.linear2",

        # Example targets in SAM mask decoder (e.g., first block self-attention)
        "sam_mask_decoder.transformer.layers.0.self_attn.q_proj",
        "sam_mask_decoder.transformer.layers.0.self_attn.v_proj",
        "sam_mask_decoder.transformer.layers.0.mlp.layers.0",
    ]
    target_modules = [name for name, _ in sam2.named_modules()]
    include = 'image_encoder'   # None
    exclude = None
    target_modules = [name for name in target_modules if include in name] if include else target_modules
    target_modules = [name for name in target_modules if exclude not in name] if exclude else target_modules

    # Create the LoRA adapter
    lora_sam2_model = LoRA_Adapter(
        model=sam2,
        r=8,  # Example rank
        lora_alpha=16, # Example alpha (often 2*r)
        lora_target_modules=target_modules,
        lora_dropout=0.05, # Example dropout
        verbose=True # Print which layers are replaced
    )

    # Verify trainable parameters
    lora_sam2_model.print_trainable_parameters()

    # Now you can train lora_sam2_model. Only the LoRA parameters will be updated.
    optimizer = torch.optim.AdamW(lora_sam2_model.parameters(), lr=1e-4) # Optimizer will only get trainable params

    # --- Dummy Forward Pass (requires knowing SAM2Base input format) ---
    # try:
    #     # Replace with actual input structure for SAM2Base
    #     dummy_input_image = torch.randn(1, 3, 1024, 1024) # Example image size
    #     dummy_other_inputs = ... # Add other necessary inputs for SAM2Base forward
    #     output = lora_sam2_model(dummy_input_image, ...) # Pass all required args
    #     print("Dummy forward pass successful.")
    # except Exception as e:
    #     print(f"Dummy forward pass failed: {e}")
    #     print("Please ensure the input format matches the SAM2Base model's forward signature.")


    # --- Saving and Loading LoRA weights ---
    # lora_sam2_model.save_lora_parameters("sam2_lora_weights_r8.pth")

    # --- To load later ---
    # sam2_model_new = SAM2Base(...) # Load the original base model again
    # lora_adapter_new = LoRA_Adapter(sam2_model_new, r=8, lora_target_modules=target_modules) # Re-apply adapter structure
    # lora_adapter_new.load_lora_parameters("sam2_lora_weights_r8.pth")
    # print("Loaded LoRA weights.")
    # Now lora_adapter_new has the trained LoRA weights loaded.

