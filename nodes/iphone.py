import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
import cv2

class iPhoneFilterNode:
    """iPhone相机滤镜节点"""
    
    # iPhone滤镜预设配置
    IPHONE_FILTERS = {
        # iPhone 6-7 经典滤镜
        "iPhone6_Vivid": {"brightness": 1.1, "contrast": 1.2, "saturation": 1.3, "warmth": 0},
        "iPhone6_VividWarm": {"brightness": 1.1, "contrast": 1.15, "saturation": 1.25, "warmth": 15},
        "iPhone6_VividCool": {"brightness": 1.1, "contrast": 1.15, "saturation": 1.25, "warmth": -15},
        "iPhone6_Dramatic": {"brightness": 0.95, "contrast": 1.4, "saturation": 1.1, "warmth": 0},
        "iPhone6_DramaticWarm": {"brightness": 0.95, "contrast": 1.35, "saturation": 1.1, "warmth": 10},
        "iPhone6_DramaticCool": {"brightness": 0.95, "contrast": 1.35, "saturation": 1.1, "warmth": -10},
        "iPhone6_Mono": {"brightness": 1.0, "contrast": 1.2, "saturation": 0, "warmth": 0},
        "iPhone6_Silvertone": {"brightness": 1.05, "contrast": 1.15, "saturation": 0.1, "warmth": -5},
        "iPhone6_Noir": {"brightness": 0.9, "contrast": 1.5, "saturation": 0, "warmth": 0},
        
        # iPhone 8-X 增强滤镜
        "iPhone8_Natural": {"brightness": 1.05, "contrast": 1.1, "saturation": 1.1, "warmth": 5},
        "iPhone8_StudioLight": {"brightness": 1.15, "contrast": 1.05, "saturation": 1.05, "warmth": 0},
        "iPhone8_ContourLight": {"brightness": 1.0, "contrast": 1.3, "saturation": 1.0, "warmth": 0},
        "iPhone8_StageLight": {"brightness": 0.9, "contrast": 1.4, "saturation": 0.9, "warmth": 0},
        "iPhone8_StageLightMono": {"brightness": 0.9, "contrast": 1.5, "saturation": 0, "warmth": 0},
        
        # iPhone 11-12 色彩风格
        "iPhone11_Original": {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "warmth": 0},
        "iPhone11_Vibrant": {"brightness": 1.1, "contrast": 1.25, "saturation": 1.4, "warmth": 5},
        "iPhone11_Rich": {"brightness": 1.05, "contrast": 1.3, "saturation": 1.3, "warmth": 10},
        "iPhone11_Warm": {"brightness": 1.05, "contrast": 1.1, "saturation": 1.15, "warmth": 20},
        "iPhone11_Cool": {"brightness": 1.05, "contrast": 1.15, "saturation": 1.1, "warmth": -20},
        
        # iPhone 13-14 摄影风格
        "iPhone13_Standard": {"brightness": 1.0, "contrast": 1.05, "saturation": 1.05, "warmth": 0},
        "iPhone13_RichContrast": {"brightness": 0.98, "contrast": 1.35, "saturation": 1.2, "warmth": 5},
        "iPhone13_Vibrant": {"brightness": 1.08, "contrast": 1.2, "saturation": 1.35, "warmth": 3},
        "iPhone13_Warm": {"brightness": 1.03, "contrast": 1.1, "saturation": 1.1, "warmth": 25},
        "iPhone13_Cool": {"brightness": 1.03, "contrast": 1.15, "saturation": 1.05, "warmth": -25},
        
        # iPhone 15-16 专业风格
        "iPhone15_Natural": {"brightness": 1.02, "contrast": 1.08, "saturation": 1.08, "warmth": 2},
        "iPhone15_Radiant": {"brightness": 1.12, "contrast": 1.18, "saturation": 1.25, "warmth": 8},
        "iPhone15_Peaceful": {"brightness": 1.05, "contrast": 0.95, "saturation": 0.9, "warmth": 15},
        "iPhone15_Cozy": {"brightness": 1.08, "contrast": 1.05, "saturation": 1.15, "warmth": 30},
        "iPhone15_Dramatic": {"brightness": 0.92, "contrast": 1.45, "saturation": 1.15, "warmth": -5},
        
        "iPhone16_Amber": {"brightness": 1.05, "contrast": 1.15, "saturation": 1.2, "warmth": 35, "tint": "amber"},
        "iPhone16_Gold": {"brightness": 1.08, "contrast": 1.12, "saturation": 1.18, "warmth": 28, "tint": "gold"},
        "iPhone16_Rose": {"brightness": 1.06, "contrast": 1.08, "saturation": 1.15, "warmth": 18, "tint": "rose"},
        "iPhone16_Neutral": {"brightness": 1.0, "contrast": 1.06, "saturation": 1.0, "warmth": 0},
        "iPhone16_Cool": {"brightness": 1.02, "contrast": 1.18, "saturation": 1.05, "warmth": -30, "tint": "blue"},
        "iPhone16_Standard": {"brightness": 1.01, "contrast": 1.07, "saturation": 1.06, "warmth": 1},
        "iPhone16_Luminous": {"brightness": 1.15, "contrast": 1.12, "saturation": 1.22, "warmth": 5},
        "iPhone16_Quiet": {"brightness": 1.03, "contrast": 0.92, "saturation": 0.85, "warmth": 12},
        "iPhone16_Cozy": {"brightness": 1.06, "contrast": 1.03, "saturation": 1.12, "warmth": 32},
        "iPhone16_Earthly": {"brightness": 0.98, "contrast": 1.15, "saturation": 1.08, "warmth": 15, "tint": "earth"},
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        filter_list = ["Custom"] + list(cls.IPHONE_FILTERS.keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                "filter_preset": (filter_list,),
                "custom_brightness": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "custom_contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "custom_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "custom_warmth": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0}),
                "custom_highlights": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "custom_shadows": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "custom_vignette": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "custom_grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_filter"
    CATEGORY = "🐳Pond/image"

    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        # 假设tensor形状为 [batch, height, width, channels]
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # 转换为numpy数组
        img_np = tensor.cpu().numpy()
        
        # 确保值在0-255范围内
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # 创建PIL图像
        return Image.fromarray(img_np)
    
    def pil_to_tensor(self, pil_img):
        """将PIL图像转换回tensor"""
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        # 添加batch维度
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        return img_tensor
    
    def adjust_warmth(self, img, warmth):
        """调整色温"""
        if warmth == 0:
            return img
        
        img_np = np.array(img).astype(np.float32)
        
        # 色温调整通过调整红蓝通道
        if warmth > 0:
            # 暖色调：增加红色，减少蓝色
            img_np[:,:,0] = np.clip(img_np[:,:,0] * (1 + warmth/100), 0, 255)
            img_np[:,:,2] = np.clip(img_np[:,:,2] * (1 - warmth/200), 0, 255)
        else:
            # 冷色调：减少红色，增加蓝色
            img_np[:,:,0] = np.clip(img_np[:,:,0] * (1 + warmth/200), 0, 255)
            img_np[:,:,2] = np.clip(img_np[:,:,2] * (1 - warmth/100), 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def adjust_highlights_shadows(self, img, highlights, shadows):
        """调整高光和阴影"""
        img_np = np.array(img).astype(np.float32)
        
        # 转换到LAB色彩空间以更好地处理亮度
        img_cv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_cv)
        
        # 调整高光（亮部）
        if highlights != 0:
            mask_highlights = l > 128
            l[mask_highlights] = np.clip(l[mask_highlights] + highlights, 0, 255)
        
        # 调整阴影（暗部）
        if shadows != 0:
            mask_shadows = l <= 128
            l[mask_shadows] = np.clip(l[mask_shadows] + shadows, 0, 255)
        
        # 合并通道并转换回RGB
        img_lab = cv2.merge([l, a, b])
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_rgb)
    
    def add_vignette(self, img, strength):
        """添加暗角效果"""
        if strength == 0:
            return img
        
        img_np = np.array(img).astype(np.float32)
        h, w = img_np.shape[:2]
        
        # 创建径向渐变
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist / max_dist) * strength
        vignette = np.clip(vignette, 1-strength, 1)
        
        # 应用暗角
        for i in range(3):
            img_np[:,:,i] *= vignette
        
        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
    
    def add_grain(self, img, intensity):
        """添加胶片颗粒效果"""
        if intensity == 0:
            return img
        
        img_np = np.array(img).astype(np.float32)
        
        # 生成噪声
        noise = np.random.normal(0, intensity, img_np.shape)
        
        # 添加噪声
        img_np = img_np + noise
        
        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
    
    def apply_tint(self, img, tint_type):
        """应用色调"""
        if not tint_type:
            return img
        
        img_np = np.array(img).astype(np.float32)
        
        tint_adjustments = {
            "amber": [1.15, 1.05, 0.85],
            "gold": [1.12, 1.08, 0.88],
            "rose": [1.10, 0.95, 1.05],
            "blue": [0.92, 0.98, 1.12],
            "earth": [1.05, 1.00, 0.90],
        }
        
        if tint_type in tint_adjustments:
            factors = tint_adjustments[tint_type]
            for i in range(3):
                img_np[:,:,i] = np.clip(img_np[:,:,i] * factors[i], 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def apply_filter(self, image, filter_preset, custom_brightness, custom_contrast, 
                    custom_saturation, custom_warmth, custom_highlights, custom_shadows,
                    custom_vignette, custom_grain):
        """应用滤镜主函数"""
        
        # 转换图像
        pil_img = self.tensor_to_pil(image)
        
        # 确定滤镜参数
        if filter_preset == "Custom":
            params = {
                "brightness": custom_brightness,
                "contrast": custom_contrast,
                "saturation": custom_saturation,
                "warmth": custom_warmth,
                "highlights": custom_highlights,
                "shadows": custom_shadows,
                "vignette": custom_vignette,
                "grain": custom_grain
            }
        else:
            # 使用预设参数
            params = self.IPHONE_FILTERS.get(filter_preset, {}).copy()
            # 添加额外的自定义参数
            params.setdefault("highlights", custom_highlights)
            params.setdefault("shadows", custom_shadows)
            params.setdefault("vignette", custom_vignette)
            params.setdefault("grain", custom_grain)
        
        # 应用基础调整
        # 亮度
        if params.get("brightness", 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(params["brightness"])
        
        # 对比度
        if params.get("contrast", 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(params["contrast"])
        
        # 饱和度
        if params.get("saturation", 1.0) != 1.0:
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(params["saturation"])
        
        # 色温
        if params.get("warmth", 0) != 0:
            pil_img = self.adjust_warmth(pil_img, params["warmth"])
        
        # 高光和阴影
        if params.get("highlights", 0) != 0 or params.get("shadows", 0) != 0:
            pil_img = self.adjust_highlights_shadows(
                pil_img, 
                params.get("highlights", 0), 
                params.get("shadows", 0)
            )
        
        # 应用色调（如果有）
        if "tint" in params:
            pil_img = self.apply_tint(pil_img, params["tint"])
        
        # 暗角效果
        if params.get("vignette", 0) > 0:
            pil_img = self.add_vignette(pil_img, params["vignette"])
        
        # 颗粒效果
        if params.get("grain", 0) > 0:
            pil_img = self.add_grain(pil_img, params["grain"])
        
        # 转换回tensor
        return (self.pil_to_tensor(pil_img),)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "iPhoneFilter": iPhoneFilterNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "iPhoneFilter": "🐳iPhone"
}
