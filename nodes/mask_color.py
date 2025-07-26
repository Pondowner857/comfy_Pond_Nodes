import torch
import numpy as np
from PIL import Image

class MaskColorReplace:
    """
    将遮罩的黑色部分替换为自定义颜色的节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "red": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
                "green": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
                "blue": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "replace_color"
    
    CATEGORY = "🐳Pond/mask"
    
    def replace_color(self, mask, red, green, blue, background_image=None, threshold=0.01):
        # 确保mask是正确的形状
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        batch_size = mask.shape[0]
        height = mask.shape[1]
        width = mask.shape[2]
        
        # 创建RGB图像
        result_images = []
        
        for i in range(batch_size):
            # 获取当前批次的mask
            current_mask = mask[i]
            
            # 如果提供了背景图像，使用它；否则创建白色背景
            if background_image is not None:
                # 确保背景图像与mask尺寸相同
                bg_img = background_image[i % background_image.shape[0]]
                if bg_img.shape[0] != height or bg_img.shape[1] != width:
                    # 调整背景图像大小
                    bg_pil = Image.fromarray((bg_img.cpu().numpy() * 255).astype(np.uint8))
                    bg_pil = bg_pil.resize((width, height), Image.Resampling.LANCZOS)
                    bg_img = torch.from_numpy(np.array(bg_pil).astype(np.float32) / 255.0)
                    if len(bg_img.shape) == 2:
                        bg_img = bg_img.unsqueeze(-1).repeat(1, 1, 3)
                result_img = bg_img.clone()
            else:
                # 创建白色背景
                result_img = torch.ones((height, width, 3), dtype=torch.float32)
            
            # 将遮罩转换为0-1范围
            mask_normalized = current_mask.float()
            
            # 创建颜色数组
            color = torch.tensor([red/255.0, green/255.0, blue/255.0], dtype=torch.float32)
            
            # 找到黑色区域（值小于阈值的区域）
            black_mask = mask_normalized < threshold
            
            # 将黑色区域替换为指定颜色
            for c in range(3):
                result_img[:, :, c][black_mask] = color[c]
            
            result_images.append(result_img)
        
        # 将结果堆叠成批次
        result_batch = torch.stack(result_images, dim=0)
        
        return (result_batch,)

class MaskColorReplaceAdvanced:
    """
    高级版本：支持渐变和混合模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "color_hex": ("STRING", {
                    "default": "#000000",
                    "multiline": False
                }),
                "blend_mode": (["replace", "multiply", "overlay", "soft_light"],),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "replace_color_advanced"
    
    CATEGORY = "🐳Pond/mask"
    
    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def apply_feather(self, mask, feather_amount):
        """应用羽化效果"""
        if feather_amount == 0:
            return mask
        
        from scipy.ndimage import gaussian_filter
        mask_np = mask.cpu().numpy()
        feathered = gaussian_filter(mask_np, sigma=feather_amount)
        return torch.from_numpy(feathered).to(mask.device)
    
    def replace_color_advanced(self, mask, color_hex, blend_mode, background_image=None, opacity=1.0, feather=0):
        # 转换颜色
        r, g, b = self.hex_to_rgb(color_hex)
        
        # 确保mask是正确的形状
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        batch_size = mask.shape[0]
        height = mask.shape[1]
        width = mask.shape[2]
        
        result_images = []
        
        for i in range(batch_size):
            current_mask = mask[i]
            
            # 应用羽化
            if feather > 0:
                current_mask = self.apply_feather(current_mask, feather)
            
            # 创建或获取背景
            if background_image is not None:
                bg_img = background_image[i % background_image.shape[0]]
                if bg_img.shape[0] != height or bg_img.shape[1] != width:
                    bg_pil = Image.fromarray((bg_img.cpu().numpy() * 255).astype(np.uint8))
                    bg_pil = bg_pil.resize((width, height), Image.Resampling.LANCZOS)
                    bg_img = torch.from_numpy(np.array(bg_pil).astype(np.float32) / 255.0)
                    if len(bg_img.shape) == 2:
                        bg_img = bg_img.unsqueeze(-1).repeat(1, 1, 3)
                result_img = bg_img.clone()
            else:
                result_img = torch.ones((height, width, 3), dtype=torch.float32)
            
            # 反转遮罩（使黑色区域为1，白色区域为0）
            inverted_mask = 1.0 - current_mask.float()
            
            # 创建颜色层
            color_layer = torch.zeros((height, width, 3), dtype=torch.float32)
            color_layer[:, :, 0] = r
            color_layer[:, :, 1] = g
            color_layer[:, :, 2] = b
            
            # 根据混合模式应用颜色
            if blend_mode == "replace":
                # 直接替换
                for c in range(3):
                    result_img[:, :, c] = result_img[:, :, c] * (1 - inverted_mask * opacity) + color_layer[:, :, c] * inverted_mask * opacity
            
            elif blend_mode == "multiply":
                # 正片叠底
                blended = result_img * color_layer
                for c in range(3):
                    result_img[:, :, c] = result_img[:, :, c] * (1 - inverted_mask * opacity) + blended[:, :, c] * inverted_mask * opacity
            
            elif blend_mode == "overlay":
                # 叠加
                def overlay_blend(base, blend):
                    return torch.where(base < 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))
                
                blended = overlay_blend(result_img, color_layer)
                for c in range(3):
                    result_img[:, :, c] = result_img[:, :, c] * (1 - inverted_mask * opacity) + blended[:, :, c] * inverted_mask * opacity
            
            elif blend_mode == "soft_light":
                # 柔光
                def soft_light_blend(base, blend):
                    return torch.where(blend < 0.5, 
                                     base * (2 * blend + base * (1 - 2 * blend)),
                                     base + (2 * blend - 1) * (torch.sqrt(base) - base))
                
                blended = soft_light_blend(result_img, color_layer)
                for c in range(3):
                    result_img[:, :, c] = result_img[:, :, c] * (1 - inverted_mask * opacity) + blended[:, :, c] * inverted_mask * opacity
            
            result_images.append(result_img)
        
        result_batch = torch.stack(result_images, dim=0)
        return (result_batch,)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskColorReplace": MaskColorReplace,
    "MaskColorReplaceAdvanced": MaskColorReplaceAdvanced,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskColorReplace": "🐳Mask_Color",
    "MaskColorReplaceAdvanced": "🐳Mask_Color_V2",
}