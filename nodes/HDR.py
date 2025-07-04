import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2
from scipy.ndimage import gaussian_filter
import colorsys

# 颜色分级节点
class ColorGradingNode:
    """
    专业颜色分级节点，提供电影级调色功能
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # 高光、中间调、阴影调整
                "高光_红": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "高光_绿": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "高光_蓝": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "中间调_红": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "中间调_绿": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "中间调_蓝": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "阴影_红": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "阴影_绿": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "阴影_蓝": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                # 色彩平衡
                "色彩平衡": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "色调分离强度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_grading"
    CATEGORY = "🐳Pond/颜色"

    def apply_color_grading(self, image, 高光_红, 高光_绿, 高光_蓝,
                           中间调_红, 中间调_绿, 中间调_蓝,
                           阴影_红, 阴影_绿, 阴影_蓝,
                           色彩平衡, 色调分离强度):
        
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            img_array = image[i].cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            
            # 创建亮度蒙版
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            
            # 定义高光、中间调、阴影区域
            highlights = np.power(gray, 2)
            shadows = 1 - np.power(1 - gray, 2)
            midtones = 1 - highlights - (1 - shadows)
            
            # 扩展蒙版到3通道
            highlights = np.stack([highlights] * 3, axis=2)
            midtones = np.stack([midtones] * 3, axis=2)
            shadows = np.stack([shadows] * 3, axis=2)
            
            # 应用颜色分级
            result = img_array.astype(np.float32) / 255.0
            
            # 高光调整
            result[:, :, 0] += highlights[:, :, 0] * 高光_红
            result[:, :, 1] += highlights[:, :, 1] * 高光_绿
            result[:, :, 2] += highlights[:, :, 2] * 高光_蓝
            
            # 中间调调整
            result[:, :, 0] += midtones[:, :, 0] * 中间调_红
            result[:, :, 1] += midtones[:, :, 1] * 中间调_绿
            result[:, :, 2] += midtones[:, :, 2] * 中间调_蓝
            
            # 阴影调整
            result[:, :, 0] += shadows[:, :, 0] * 阴影_红
            result[:, :, 1] += shadows[:, :, 1] * 阴影_绿
            result[:, :, 2] += shadows[:, :, 2] * 阴影_蓝
            
            # 色彩平衡
            if 色彩平衡 != 0:
                result = self.apply_color_balance(result, 色彩平衡)
            
            # 色调分离
            if 色调分离强度 > 0:
                result = self.apply_split_toning(result, 色调分离强度)
            
            # 限制范围并转换
            result = np.clip(result, 0, 1)
            result_images.append(result)
        
        result_tensor = torch.from_numpy(np.stack(result_images))
        return (result_tensor,)
    
    def apply_color_balance(self, image, balance):
        """应用色彩平衡"""
        # 调整红-青和黄-蓝平衡
        image[:, :, 0] *= (1 + balance * 0.1)  # 红
        image[:, :, 2] *= (1 - balance * 0.1)  # 蓝
        return image
    
    def apply_split_toning(self, image, strength):
        """应用色调分离效果"""
        # 为高光添加暖色调，阴影添加冷色调
        gray = np.mean(image, axis=2)
        
        # 高光暖色
        highlight_mask = (gray > 0.5).astype(np.float32) * strength
        image[:, :, 0] += highlight_mask * 0.1
        image[:, :, 1] += highlight_mask * 0.05
        
        # 阴影冷色
        shadow_mask = (gray < 0.5).astype(np.float32) * strength
        image[:, :, 2] += shadow_mask * 0.1
        
        return image


# HDR效果节点
class HDREffectNode:
    """
    HDR效果处理节点，增强图像动态范围
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "HDR强度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "色调映射": (["Reinhard", "Drago", "Mantiuk", "线性"],),
                "细节增强": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "局部对比度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "高光压缩": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "阴影提升": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_hdr_effect"
    CATEGORY = "🐳Pond/颜色"

    def apply_hdr_effect(self, image, HDR强度, 色调映射, 细节增强, 
                        局部对比度, 高光压缩, 阴影提升):
        
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            img_array = image[i].cpu().numpy()
            
            # 应用HDR处理
            hdr_result = self.process_hdr(img_array, HDR强度, 色调映射)
            
            # 细节增强
            if 细节增强 > 0:
                hdr_result = self.enhance_details(hdr_result, 细节增强)
            
            # 局部对比度调整
            if 局部对比度 > 0:
                hdr_result = self.enhance_local_contrast(hdr_result, 局部对比度)
            
            # 高光和阴影调整
            hdr_result = self.adjust_highlights_shadows(hdr_result, 高光压缩, 阴影提升)
            
            result_images.append(hdr_result)
        
        result_tensor = torch.from_numpy(np.stack(result_images))
        return (result_tensor,)
    
    def process_hdr(self, image, strength, tone_mapping):
        """HDR处理"""
        # 转换到float32
        img_float = image.astype(np.float32)
        
        # 计算亮度
        luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
        
        # 应用不同的色调映射算法
        if tone_mapping == "Reinhard":
            # Reinhard色调映射
            mapped_lum = luminance / (1.0 + luminance)
        elif tone_mapping == "Drago":
            # Drago对数映射
            bias = 0.85
            mapped_lum = np.log10(1 + luminance) / np.log10(1 + np.max(luminance))
            mapped_lum = np.power(mapped_lum, np.log(bias) / np.log(0.5))
        elif tone_mapping == "Mantiuk":
            # 简化的Mantiuk映射
            mapped_lum = luminance / (luminance + 1)
            mapped_lum = np.power(mapped_lum, 1.0 / 2.2)
        else:  # 线性
            mapped_lum = np.clip(luminance, 0, 1)
        
        # 计算缩放因子
        scale = np.where(luminance > 0, mapped_lum / luminance, 1)
        scale = scale[:, :, np.newaxis]
        
        # 应用映射
        result = img_float * scale
        
        # 混合原图和HDR结果
        result = img_float * (1 - strength) + result * strength
        
        return np.clip(result, 0, 1)
    
    def enhance_details(self, image, strength):
        """增强细节"""
        # 使用非锐化掩模增强细节
        img_uint8 = (image * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (0, 0), 3)
        
        # 计算细节层
        details = img_uint8.astype(np.float32) - blurred.astype(np.float32)
        
        # 增强细节
        enhanced = img_uint8.astype(np.float32) + details * strength * 2
        
        return np.clip(enhanced / 255.0, 0, 1)
    
    def enhance_local_contrast(self, image, strength):
        """增强局部对比度"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img_lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0 * strength + 1.0, tileGridSize=(8, 8))
        
        # 只对L通道应用
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        
        # 转换回RGB
        result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        return result.astype(np.float32) / 255.0
    
    def adjust_highlights_shadows(self, image, highlight_compression, shadow_lift):
        """调整高光和阴影"""
        # 计算亮度
        luminance = np.mean(image, axis=2)
        
        # 高光压缩
        if highlight_compression > 0:
            highlight_mask = np.clip((luminance - 0.7) / 0.3, 0, 1)
            compression = 1 - highlight_mask * highlight_compression * 0.5
            image = image * compression[:, :, np.newaxis]
        
        # 阴影提升
        if shadow_lift > 0:
            shadow_mask = np.clip((0.3 - luminance) / 0.3, 0, 1)
            lift = 1 + shadow_mask * shadow_lift * 0.5
            image = image * lift[:, :, np.newaxis]
        
        return np.clip(image, 0, 1)


# 皮肤美化节点
class SkinEnhancementNode:
    """
    智能皮肤美化节点，专门用于人像后期处理
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "磨皮强度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "美白程度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "红润度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "细节保留": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "去瑕疵": ("BOOLEAN", {"default": True}),
                "眼睛增强": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "牙齿美白": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_skin"
    CATEGORY = "🐳Pond/颜色"

    def enhance_skin(self, image, 磨皮强度, 美白程度, 红润度, 
                    细节保留, 去瑕疵, 眼睛增强, 牙齿美白):
        
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            img_array = image[i].cpu().numpy()
            img_uint8 = (img_array * 255).astype(np.uint8)
            
            # 检测皮肤区域
            skin_mask = self.detect_skin(img_uint8)
            
            # 磨皮处理
            if 磨皮强度 > 0:
                img_uint8 = self.smooth_skin(img_uint8, skin_mask, 磨皮强度, 细节保留)
            
            # 美白处理
            if 美白程度 > 0:
                img_uint8 = self.whiten_skin(img_uint8, skin_mask, 美白程度)
            
            # 增加红润度
            if 红润度 > 0:
                img_uint8 = self.add_blush(img_uint8, skin_mask, 红润度)
            
            # 去瑕疵
            if 去瑕疵:
                img_uint8 = self.remove_blemishes(img_uint8, skin_mask)
            
            # 眼睛增强
            if 眼睛增强 > 0:
                img_uint8 = self.enhance_eyes(img_uint8, 眼睛增强)
            
            # 牙齿美白
            if 牙齿美白 > 0:
                img_uint8 = self.whiten_teeth(img_uint8, 牙齿美白)
            
            result_images.append(img_uint8.astype(np.float32) / 255.0)
        
        result_tensor = torch.from_numpy(np.stack(result_images))
        return (result_tensor,)
    
    def detect_skin(self, image):
        """检测皮肤区域"""
        # 转换到YCrCb色彩空间
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # 定义皮肤色彩范围
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # 创建皮肤掩模
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        
        # 形态学操作优化掩模
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        return skin_mask.astype(np.float32) / 255.0
    
    def smooth_skin(self, image, skin_mask, strength, detail_preserve):
        """磨皮处理"""
        # 使用双边滤波保持边缘
        smooth = cv2.bilateralFilter(image, 
                                    int(15 * strength), 
                                    int(80 * strength), 
                                    int(80 * strength))
        
        # 保留细节
        if detail_preserve > 0:
            # 高通滤波提取细节
            blur = cv2.GaussianBlur(image, (21, 21), 0)
            detail = image.astype(np.float32) - blur.astype(np.float32)
            
            # 添加回部分细节
            smooth = smooth.astype(np.float32) + detail * detail_preserve
            smooth = np.clip(smooth, 0, 255).astype(np.uint8)
        
        # 应用蒙版
        skin_mask_3ch = np.stack([skin_mask] * 3, axis=2)
        result = image * (1 - skin_mask_3ch) + smooth * skin_mask_3ch
        
        return result.astype(np.uint8)
    
    def whiten_skin(self, image, skin_mask, strength):
        """美白皮肤"""
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 增加明度，降低饱和度
        skin_mask_expand = skin_mask[:, :, np.newaxis]
        hsv[:, :, 2] += skin_mask_expand[:, :, 0] * 30 * strength  # 明度
        hsv[:, :, 1] *= 1 - (skin_mask_expand[:, :, 0] * 0.3 * strength)  # 饱和度
        
        # 限制范围
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # 转换回RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return result
    
    def add_blush(self, image, skin_mask, strength):
        """增加红润度"""
        # 在脸颊区域增加红色
        result = image.copy()
        skin_mask_3ch = np.stack([skin_mask] * 3, axis=2)
        
        # 增加红色通道
        blush = image.astype(np.float32)
        blush[:, :, 0] += 20 * strength  # 红色
        blush[:, :, 1] += 10 * strength  # 略微增加绿色
        
        # 应用蒙版
        result = image * (1 - skin_mask_3ch * strength * 0.5) + blush * skin_mask_3ch * strength * 0.5
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def remove_blemishes(self, image, skin_mask):
        """去除瑕疵"""
        # 使用中值滤波去除小瑕疵
        result = image.copy()
        
        # 只在皮肤区域应用
        skin_area = cv2.medianBlur(image, 5)
        
        skin_mask_3ch = np.stack([skin_mask] * 3, axis=2)
        result = image * (1 - skin_mask_3ch * 0.3) + skin_area * skin_mask_3ch * 0.3
        
        return result.astype(np.uint8)
    
    def enhance_eyes(self, image, strength):
        """增强眼睛"""
        # 简化处理：增加对比度和锐度
        # 实际应用中需要眼睛检测
        result = image.copy()
        
        # 增加局部对比度和锐度
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * strength * 0.1
        
        sharpened = cv2.filter2D(image, -1, kernel)
        result = cv2.addWeighted(image, 1 - strength * 0.3, sharpened, strength * 0.3, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def whiten_teeth(self, image, strength):
        """美白牙齿"""
        # 简化处理：检测白色区域并增强
        # 实际应用中需要牙齿检测
        result = image.copy()
        
        # 检测接近白色的区域
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 轻微增加亮度
        white_area = image.astype(np.float32)
        white_area += 20 * strength
        
        # 应用蒙版
        white_mask_3ch = np.stack([white_mask] * 3, axis=2).astype(np.float32) / 255.0
        result = image * (1 - white_mask_3ch * strength * 0.5) + white_area * white_mask_3ch * strength * 0.5
        
        return np.clip(result, 0, 255).astype(np.uint8)


# 艺术效果节点
class ArtisticEffectsNode:
    """
    艺术效果节点，提供各种艺术风格化效果
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "效果类型": (["油画", "水彩", "素描", "漫画", "印象派", "点彩画", "版画", "马赛克"],),
                "效果强度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "笔触大小": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "色彩简化": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "纹理强度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "保留细节": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_artistic_effect"
    CATEGORY = "🐳Pond/颜色"

    def apply_artistic_effect(self, image, 效果类型, 效果强度, 笔触大小, 
                            色彩简化, 纹理强度, 保留细节):
        
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            img_array = image[i].cpu().numpy()
            img_uint8 = (img_array * 255).astype(np.uint8)
            
            if 效果类型 == "油画":
                result = self.oil_painting_effect(img_uint8, 笔触大小, 效果强度)
            elif 效果类型 == "水彩":
                result = self.watercolor_effect(img_uint8, 效果强度, 纹理强度)
            elif 效果类型 == "素描":
                result = self.pencil_sketch_effect(img_uint8, 效果强度, 保留细节)
            elif 效果类型 == "漫画":
                result = self.cartoon_effect(img_uint8, 色彩简化, 效果强度)
            elif 效果类型 == "印象派":
                result = self.impressionist_effect(img_uint8, 笔触大小, 效果强度)
            elif 效果类型 == "点彩画":
                result = self.pointillism_effect(img_uint8, 笔触大小, 效果强度)
            elif 效果类型 == "版画":
                result = self.engraving_effect(img_uint8, 效果强度)
            elif 效果类型 == "马赛克":
                result = self.mosaic_effect(img_uint8, 笔触大小, 效果强度)
            else:
                result = img_uint8
            
            result_images.append(result.astype(np.float32) / 255.0)
        
        result_tensor = torch.from_numpy(np.stack(result_images))
        return (result_tensor,)
    
    def oil_painting_effect(self, image, brush_size, strength):
        """油画效果 - 不依赖xphoto模块的实现"""
        h, w = image.shape[:2]
        
        # 1. 使用边缘保留滤波模拟油画的平滑效果
        # 多次应用双边滤波以增强效果
        result = image.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, brush_size * 2, 50, 50)
        
        # 2. 颜色量化，模拟油画的色块效果
        # 减少颜色数量
        div = 32  # 颜色级别
        result = result // div * div + div // 2
        
        # 3. 创建油画笔触纹理
        # 使用形态学操作创建笔触效果
        kernel_size = max(3, brush_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 对每个通道分别处理
        for i in range(3):
            result[:, :, i] = cv2.morphologyEx(result[:, :, i], cv2.MORPH_CLOSE, kernel)
        
        # 4. 添加轻微的纹理增强
        # 使用拉普拉斯算子检测边缘
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.absolute(edges)
        edges = np.uint8(np.clip(edges, 0, 255))
        
        # 将边缘添加回图像以增强纹理
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(result, 1.0, edges_3ch, 0.1, 0)
        
        # 5. 最终混合
        result = cv2.addWeighted(image, 1 - strength, result, strength, 0)
        
        return result
    
    def watercolor_effect(self, image, strength, texture_strength):
        """水彩效果"""
        # 边缘保留滤波
        result = cv2.bilateralFilter(image, 15, 80, 80)
        result = cv2.bilateralFilter(result, 15, 80, 80)
        
        # 创建水彩纹理
        # 使用随机噪声模拟水彩晕染
        h, w = image.shape[:2]
        texture = np.random.normal(0, 25 * texture_strength, (h, w, 3))
        
        # 应用纹理
        result = result.astype(np.float32) + texture
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 颜色简化
        result = cv2.edgePreservingFilter(result, flags=1, sigma_s=60, sigma_r=0.4)
        
        # 混合
        result = cv2.addWeighted(image, 1 - strength, result, strength, 0)
        
        return result
    
    def pencil_sketch_effect(self, image, strength, detail_preserve):
        """素描效果"""
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 反转
        inv_gray = 255 - gray
        
        # 高斯模糊
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        
        # 颜色减淡混合
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        
        # 保留部分细节
        if detail_preserve > 0:
            edges = cv2.Canny(gray, 50, 150)
            sketch = cv2.addWeighted(sketch, 1 - detail_preserve, edges, detail_preserve, 0)
        
        # 转换回三通道
        sketch_3ch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        # 混合
        result = cv2.addWeighted(image, 1 - strength, sketch_3ch, strength, 0)
        
        return result
    
    def cartoon_effect(self, image, num_colors, strength):
        """漫画效果"""
        # 边缘保留滤波
        smooth = cv2.bilateralFilter(image, 15, 80, 80)
        smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
        
        # 边缘检测
        gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 10)
        
        # 颜色量化
        if num_colors > 0:
            # K-means颜色聚类
            data = smooth.reshape((-1, 3))
            data = np.float32(data)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, 
                                           cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            quantized = centers[labels.flatten()]
            quantized = quantized.reshape(smooth.shape)
        else:
            quantized = smooth
        
        # 将边缘转换为三通道
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges = cv2.bitwise_not(edges)
        
        # 合并边缘和颜色
        cartoon = cv2.bitwise_and(quantized, edges)
        
        # 混合
        result = cv2.addWeighted(image, 1 - strength, cartoon, strength, 0)
        
        return result
    
    def impressionist_effect(self, image, brush_size, strength):
        """印象派效果"""
        h, w = image.shape[:2]
        result = np.zeros_like(image)
        
        # 创建随机笔触
        num_strokes = 1000
        for _ in range(num_strokes):
            # 随机位置
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            # 获取该点的颜色
            color = image[y, x]
            
            # 随机笔触大小和角度
            size = np.random.randint(brush_size, brush_size * 2)
            angle = np.random.randint(0, 360)
            
            # 绘制椭圆笔触
            axes = (size, size // 2)
            cv2.ellipse(result, (x, y), axes, angle, 0, 360, color.tolist(), -1)
        
        # 混合
        result = cv2.addWeighted(image, 1 - strength, result, strength, 0)
        
        return result
    
    def pointillism_effect(self, image, dot_size, strength):
        """点彩画效果"""
        h, w = image.shape[:2]
        result = np.ones_like(image) * 255  # 白色背景
        
        # 创建点阵
        step = dot_size * 2
        for y in range(0, h, step):
            for x in range(0, w, step):
                # 获取区域平均颜色
                roi = image[y:y+step, x:x+step]
                if roi.size > 0:
                    color = np.mean(roi, axis=(0, 1))
                    
                    # 绘制圆点
                    cv2.circle(result, (x + step//2, y + step//2), 
                             dot_size, color.tolist(), -1)
        
        # 混合
        result = cv2.addWeighted(image, 1 - strength, result, strength, 0)
        
        return result
    
    def engraving_effect(self, image, strength):
        """版画效果"""
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 创建线条纹理
        h, w = gray.shape
        texture = np.zeros_like(gray)
        
        # 水平线条
        for y in range(0, h, 2):
            texture[y, :] = 255
        
        # 根据亮度调制线条
        result = np.where(gray > 128, texture, 255 - texture)
        
        # 转换回三通道
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        # 混合
        result = cv2.addWeighted(image, 1 - strength, result, strength, 0)
        
        return result
    
    def mosaic_effect(self, image, block_size, strength):
        """马赛克效果"""
        h, w = image.shape[:2]
        result = image.copy()
        
        # 创建马赛克
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                # 获取块的平均颜色
                roi = image[y:y+block_size, x:x+block_size]
                if roi.size > 0:
                    color = np.mean(roi, axis=(0, 1))
                    result[y:y+block_size, x:x+block_size] = color
        
        # 混合
        result = cv2.addWeighted(image, 1 - strength, result, strength, 0)
        
        return result


# 选择性颜色调整节点
class SelectiveColorNode:
    """
    选择性颜色调整节点，可以针对特定颜色范围进行调整
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "目标颜色": (["红色", "黄色", "绿色", "青色", "蓝色", "品红", "白色", "黑色"],),
                "色相偏移": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "饱和度调整": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "明度调整": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "范围宽度": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 90.0, "step": 1.0}),
                "羽化程度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_selective_color"
    CATEGORY = "🐳Pond/颜色"

    def adjust_selective_color(self, image, 目标颜色, 色相偏移, 饱和度调整, 
                              明度调整, 范围宽度, 羽化程度):
        
        # 定义各颜色的中心色相值
        color_hues = {
            "红色": 0,
            "黄色": 60,
            "绿色": 120,
            "青色": 180,
            "蓝色": 240,
            "品红": 300,
            "白色": -1,  # 特殊处理
            "黑色": -2   # 特殊处理
        }
        
        batch_size = image.shape[0]
        result_images = []
        
        for i in range(batch_size):
            img_array = image[i].cpu().numpy()
            
            if 目标颜色 in ["白色", "黑色"]:
                # 基于亮度的选择
                result = self.adjust_by_luminance(
                    img_array, 目标颜色, 饱和度调整, 明度调整, 羽化程度
                )
            else:
                # 基于色相的选择
                target_hue = color_hues[目标颜色]
                result = self.adjust_by_hue(
                    img_array, target_hue, 色相偏移, 饱和度调整, 
                    明度调整, 范围宽度, 羽化程度
                )
            
            result_images.append(result)
        
        result_tensor = torch.from_numpy(np.stack(result_images))
        return (result_tensor,)
    
    def adjust_by_hue(self, image, target_hue, hue_shift, sat_adjust, 
                     val_adjust, range_width, feather):
        """基于色相的选择性调整"""
        # 转换到HSV
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 创建色相掩模
        hue_mask = self.create_hue_mask(hsv[:, :, 0], target_hue, range_width, feather)
        
        # 应用调整
        hsv[:, :, 0] = hsv[:, :, 0] + hue_mask * hue_shift
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + hue_mask * sat_adjust)
        hsv[:, :, 2] = hsv[:, :, 2] * (1 + hue_mask * val_adjust)
        
        # 限制范围
        hsv[:, :, 0] = hsv[:, :, 0] % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # 转换回RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32) / 255.0
    
    def adjust_by_luminance(self, image, target, sat_adjust, val_adjust, feather):
        """基于亮度的选择性调整"""
        # 计算亮度
        gray = np.mean(image, axis=2)
        
        # 创建亮度掩模
        if target == "白色":
            mask = self.create_luminance_mask(gray, 0.7, 1.0, feather)
        else:  # 黑色
            mask = self.create_luminance_mask(gray, 0.0, 0.3, feather)
        
        # 转换到HSV进行调整
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 应用调整
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + mask * sat_adjust)
        hsv[:, :, 2] = hsv[:, :, 2] * (1 + mask * val_adjust)
        
        # 限制范围
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # 转换回RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32) / 255.0
    
    def create_hue_mask(self, hue_channel, target_hue, range_width, feather):
        """创建色相选择掩模"""
        # 计算色相差异
        hue_diff = np.abs(hue_channel - target_hue)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # 处理色相环绕
        
        # 创建掩模
        half_width = range_width / 2
        mask = np.zeros_like(hue_channel)
        
        # 完全选中的区域
        mask[hue_diff <= half_width * (1 - feather)] = 1
        
        # 羽化区域
        feather_start = half_width * (1 - feather)
        feather_end = half_width
        feather_mask = (hue_diff > feather_start) & (hue_diff <= feather_end)
        mask[feather_mask] = 1 - (hue_diff[feather_mask] - feather_start) / (feather_end - feather_start)
        
        return mask
    
    def create_luminance_mask(self, luminance, min_val, max_val, feather):
        """创建亮度选择掩模"""
        mask = np.zeros_like(luminance)
        
        # 完全选中的区域
        range_size = max_val - min_val
        inner_min = min_val + range_size * feather * 0.5
        inner_max = max_val - range_size * feather * 0.5
        
        mask[(luminance >= inner_min) & (luminance <= inner_max)] = 1
        
        # 下羽化
        lower_feather = (luminance >= min_val) & (luminance < inner_min)
        mask[lower_feather] = (luminance[lower_feather] - min_val) / (inner_min - min_val)
        
        # 上羽化
        upper_feather = (luminance > inner_max) & (luminance <= max_val)
        mask[upper_feather] = 1 - (luminance[upper_feather] - inner_max) / (max_val - inner_max)
        
        return mask

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ColorGradingNode": ColorGradingNode,
    "HDREffectNode": HDREffectNode,
    "SkinEnhancementNode": SkinEnhancementNode,
    "ArtisticEffectsNode": ArtisticEffectsNode,
    "SelectiveColorNode": SelectiveColorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorGradingNode": "🐳色彩平衡",
    "HDREffectNode": "🐳HDR",
    "SkinEnhancementNode": "🐳人像美化",
    "ArtisticEffectsNode": "🐳艺术效果",
    "SelectiveColorNode": "🐳色彩范围"
}