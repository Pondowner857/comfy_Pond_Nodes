import torch
import gc
import comfy.model_management as mm

class WanVideoResourceCleaner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": ("*",),  # 接受任何类型的输入
                "clear_cache": ("BOOLEAN", {"default": True}),
                "force_gc": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("*",)  # 输出任何类型
    RETURN_NAMES = ("any_output",)
    FUNCTION = "clean_resources"
    CATEGORY = "🐳Pond/video"
    DESCRIPTION = "清理资源并传递任何类型的数据"
    
    def clean_resources(self, any_input, clear_cache, force_gc, unload_model):
        # 获取设备
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # 检查输入是否是 WanVideo 模型
        if hasattr(any_input, 'model') and hasattr(any_input.model, 'diffusion_model'):
            # 获取 transformer
            transformer = any_input.model.diffusion_model
            
            # 1. 清理 block swap 状态
            if hasattr(transformer, 'block_swap'):
                # 将所有块移回 offload device
                for name, param in transformer.named_parameters():
                    if param.device != offload_device:
                        param.data = param.data.to(offload_device)
            
            # 2. 清理缓存状态
            if hasattr(transformer, 'teacache_state'):
                transformer.teacache_state.clear_all()
            if hasattr(transformer, 'magcache_state'):
                transformer.magcache_state.clear_all()
            
            # 3. 清理 VAE 缓存
            if hasattr(transformer, 'model') and hasattr(transformer.model, 'clear_cache'):
                transformer.model.clear_cache()
        
        # 4. 清理 CUDA 缓存
        if clear_cache:
            mm.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # 5. 强制垃圾回收
        if force_gc:
            gc.collect()
        
        # 6. 完全卸载模型（可选）
        if unload_model:
            mm.unload_all_models()
            mm.cleanup_models()
            # 如果是模型，将 transformer 移到 CPU
            if hasattr(any_input, 'model') and hasattr(any_input.model, 'diffusion_model'):
                any_input.model.diffusion_model.to(offload_device)
        
        # 打印内存状态
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"GPU 内存 - 已分配: {allocated:.2f}GB, 已预留: {reserved:.2f}GB")
        
        # 返回原始输入，保持数据流
        return (any_input,)


class WanVideoResetBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": ("*",),  # 接受任何类型的输入
            },
        }
    
    RETURN_TYPES = ("*",)  # 输出任何类型
    RETURN_NAMES = ("any_output",)
    FUNCTION = "reset_block_swap"
    CATEGORY = "🐳Pond/video"
    DESCRIPTION = "重置 block swap 设置（如果输入是支持的模型类型）"
    
    def reset_block_swap(self, any_input):
        # 检查输入是否是支持的模型类型
        if hasattr(any_input, 'clone') and hasattr(any_input, 'model_options'):
            # 克隆模型以避免修改原始模型
            patcher = any_input.clone()
            
            # 清除 block_swap_args
            if 'transformer_options' in patcher.model_options:
                if 'block_swap_args' in patcher.model_options['transformer_options']:
                    del patcher.model_options['transformer_options']['block_swap_args']
            
            # 获取 transformer（如果存在）
            if hasattr(patcher.model, 'diffusion_model'):
                transformer = patcher.model.diffusion_model
                device = mm.get_torch_device()
                
                # 将所有参数移回主设备
                for name, param in transformer.named_parameters():
                    if param.device != device:
                        param.data = param.data.to(device)
                
                # 重置 block swap 相关属性
                if hasattr(transformer, 'use_non_blocking'):
                    transformer.use_non_blocking = False
            
            return (patcher,)
        else:
            # 如果不是支持的类型，直接返回原始输入
            print("输入不是支持的模型类型，直接传递")
            return (any_input,)


NODE_CLASS_MAPPINGS = {
    "WanVideoResourceCleaner": WanVideoResourceCleaner,
    "WanVideoResetBlockSwap": WanVideoResetBlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoResourceCleaner": "🐳WanVideo Resource Cleaner",
    "WanVideoResetBlockSwap": "🐳WanVideo Reset Block Swap",
}