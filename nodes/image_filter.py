import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

class ImageFilterNode:
    """
    ComfyUI图像滤镜调节节点
    支持多种图像滤镜效果，包括亮度、对比度、饱和度、模糊、锐化等
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "亮度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "对比度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "饱和度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "锐度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "色调偏移": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "模糊半径": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "色温": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "伽马值": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "滤镜类型": (["无", "复古", "棕褐色", "灰度", "边缘增强", "浮雕"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filters"
    CATEGORY = "🐳Pond/image"

    def apply_filters(self, image, 亮度=1.0, 对比度=1.0, 饱和度=1.0, 
                     锐度=1.0, 色调偏移=0.0, 模糊半径=0, 色温=0.0, 
                     伽马值=1.0, 滤镜类型="无"):
        
        # 将tensor转换为PIL Image
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            # 从tensor转换为numpy数组
            img_array = image[i].cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            
            # 转换为PIL Image
            pil_image = Image.fromarray(img_array, mode='RGB')
            
            # 应用基础调整
            # 亮度调整
            if 亮度 != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(亮度)
            
            # 对比度调整
            if 对比度 != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(对比度)
            
            # 饱和度调整
            if 饱和度 != 1.0:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(饱和度)
            
            # 锐度调整
            if 锐度 != 1.0:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(锐度)
            
            # 色调调整
            if 色调偏移 != 0.0:
                pil_image = self.adjust_hue(pil_image, 色调偏移)
            
            # 模糊效果
            if 模糊半径 > 0:
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=模糊半径))
            
            # 色温调整
            if 色温 != 0.0:
                pil_image = self.adjust_temperature(pil_image, 色温)
            
            # Gamma校正
            if 伽马值 != 1.0:
                pil_image = self.adjust_gamma(pil_image, 伽马值)
            
            # 应用特殊滤镜
            if 滤镜类型 != "无":
                pil_image = self.apply_special_filter(pil_image, 滤镜类型)
            
            # 转换回tensor
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            result_images.append(img_array)
        
        # 组合批次
        result_tensor = torch.from_numpy(np.stack(result_images))
        
        return (result_tensor,)
    
    def adjust_hue(self, image, hue_shift):
        """调整色调"""
        # 转换为HSV
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 调整色调
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 360
        
        # 转换回RGB
        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(rgb)
    
    def adjust_temperature(self, image, temperature):
        """调整色温"""
        img_array = np.array(image).astype(np.float32)
        
        # 色温调整系数
        temp_scale = temperature / 100.0
        
        # 调整红色和蓝色通道
        img_array[:, :, 0] *= (1 + temp_scale * 0.3)  # 红色通道
        img_array[:, :, 2] *= (1 - temp_scale * 0.3)  # 蓝色通道
        
        # 限制范围
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def adjust_gamma(self, image, gamma):
        """Gamma校正"""
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # 应用gamma校正
        img_array = np.power(img_array, gamma)
        
        # 转换回0-255范围
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def apply_special_filter(self, image, filter_type):
        """应用特殊滤镜效果"""
        if filter_type == "复古":
            # 复古效果
            img_array = np.array(image).astype(np.float32)
            # 调整色调，增加黄色调
            img_array[:, :, 0] *= 1.1  # 红色
            img_array[:, :, 1] *= 1.0  # 绿色
            img_array[:, :, 2] *= 0.8  # 蓝色
            # 降低对比度
            img_array = (img_array - 128) * 0.8 + 128
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        
        elif filter_type == "棕褐色":
            # 棕褐色效果
            img_array = np.array(image).astype(np.float32)
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            
            tr = 0.393 * r + 0.769 * g + 0.189 * b
            tg = 0.349 * r + 0.686 * g + 0.168 * b
            tb = 0.272 * r + 0.534 * g + 0.131 * b
            
            img_array[:, :, 0] = tr
            img_array[:, :, 1] = tg
            img_array[:, :, 2] = tb
            
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        
        elif filter_type == "灰度":
            # 灰度效果
            return image.convert('L').convert('RGB')
        
        elif filter_type == "边缘增强":
            # 边缘增强
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        elif filter_type == "浮雕":
            # 浮雕效果
            return image.filter(ImageFilter.EMBOSS)
        
        return image


# 高级滤镜节点
class AdvancedImageFilterNode:
    """
    高级图像滤镜节点，提供更多专业滤镜效果
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "晕影强度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "色差": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "噪点强度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "胶片颗粒": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "泛光强度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "泛光阈值": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_advanced_filters"
    CATEGORY = "🐳Pond/image"
    
    def apply_advanced_filters(self, image, 晕影强度=0.0, 色差=0.0,
                              噪点强度=0.0, 胶片颗粒=0.0, 泛光强度=0.0, 泛光阈值=0.8):
        
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            img_array = image[i].cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            
            # 应用晕影效果
            if 晕影强度 > 0:
                img_array = self.apply_vignette(img_array, 晕影强度)
            
            # 应用色差效果
            if 色差 > 0:
                img_array = self.apply_chromatic_aberration(img_array, 色差)
            
            # 应用噪点
            if 噪点强度 > 0:
                img_array = self.apply_noise(img_array, 噪点强度)
            
            # 应用胶片颗粒
            if 胶片颗粒 > 0:
                img_array = self.apply_film_grain(img_array, 胶片颗粒)
            
            # 应用泛光效果
            if 泛光强度 > 0:
                img_array = self.apply_bloom(img_array, 泛光强度, 泛光阈值)
            
            # 转换回tensor
            img_array = img_array.astype(np.float32) / 255.0
            result_images.append(img_array)
        
        result_tensor = torch.from_numpy(np.stack(result_images))
        return (result_tensor,)
    
    def apply_vignette(self, image, strength):
        """应用晕影效果"""
        h, w = image.shape[:2]
        
        # 创建径向渐变
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # 创建晕影遮罩
        vignette = 1 - (dist / max_dist) * strength
        vignette = np.clip(vignette, 0, 1)
        vignette = vignette[:, :, np.newaxis]
        
        # 应用晕影
        result = image.astype(np.float32) * vignette
        return result.astype(np.uint8)
    
    def apply_chromatic_aberration(self, image, amount):
        """应用色差效果"""
        h, w = image.shape[:2]
        
        # 分离通道
        r, g, b = cv2.split(image)
        
        # 创建位移
        shift = int(amount)
        
        # 移动红色和蓝色通道
        M_r = np.float32([[1, 0, shift], [0, 1, 0]])
        M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
        
        r_shifted = cv2.warpAffine(r, M_r, (w, h))
        b_shifted = cv2.warpAffine(b, M_b, (w, h))
        
        # 合并通道
        result = cv2.merge([r_shifted, g, b_shifted])
        return result
    
    def apply_noise(self, image, amount):
        """应用噪点"""
        noise = np.random.normal(0, amount * 255, image.shape)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def apply_film_grain(self, image, intensity):
        """应用胶片颗粒效果"""
        h, w = image.shape[:2]
        
        # 创建颗粒纹理
        grain = np.random.normal(0, 1, (h, w))
        grain = cv2.GaussianBlur(grain, (3, 3), 0)
        grain = grain * intensity * 50
        
        # 应用到每个通道
        result = image.astype(np.float32)
        for c in range(3):
            result[:, :, c] += grain
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_bloom(self, image, intensity, threshold):
        """应用泛光效果"""
        # 提取亮部
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, bright_mask = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # 创建泛光
        bloom = cv2.bitwise_and(image, image, mask=bright_mask)
        bloom = cv2.GaussianBlur(bloom, (21, 21), 0)
        
        # 混合原图和泛光
        result = cv2.addWeighted(image, 1, bloom, intensity, 0)
        return np.clip(result, 0, 255).astype(np.uint8)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageFilterNode": ImageFilterNode,
    "AdvancedImageFilterNode": AdvancedImageFilterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFilterNode": "🐳滤镜调节",
    "AdvancedImageFilterNode": "🐳滤镜调节V2"
}