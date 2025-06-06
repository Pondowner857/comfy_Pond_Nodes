import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import comfy.utils

class PixelizeNode:
    """
    将普通图像转换为像素风格
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "像素大小": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "缩放模式": (["保持原始尺寸", "缩放到像素网格"],),
                "抗锯齿": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "pixelize"
    CATEGORY = "🐳Pond/图像"
    
    def pixelize(self, 图像, 像素大小, 缩放模式, 抗锯齿):
        batch_size, height, width, channels = 图像.shape
        processed_images = []
        
        for i in range(batch_size):
            # 转换为PIL图像
            img_tensor = 图像[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            if 缩放模式 == "保持原始尺寸":
                # 保持原始尺寸，只是像素化
                # 计算像素化后的尺寸
                pixel_width = width // 像素大小
                pixel_height = height // 像素大小
                
                # 缩小到像素尺寸
                downsample_method = Image.LANCZOS if 抗锯齿 else Image.NEAREST
                img_small = img_pil.resize((pixel_width, pixel_height), downsample_method)
                
                # 放大回原始尺寸
                img_pixelated = img_small.resize((width, height), Image.NEAREST)
            else:
                # 缩放到像素网格（确保每个像素块都是完整的）
                pixel_width = width // 像素大小
                pixel_height = height // 像素大小
                new_width = pixel_width * 像素大小
                new_height = pixel_height * 像素大小
                
                # 先调整到网格尺寸
                img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
                
                # 像素化
                downsample_method = Image.LANCZOS if 抗锯齿 else Image.NEAREST
                img_small = img_resized.resize((pixel_width, pixel_height), downsample_method)
                img_pixelated = img_small.resize((new_width, new_height), Image.NEAREST)
            
            # 转换回tensor
            img_np = np.array(img_pixelated).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output,)

class SquarePixelCorrectionNode:
    """
    将像素图像中的非正方形像素校正为1:1的正方形
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "检测模式": (["自动检测", "手动设置"],),
                "像素宽度": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "像素高度": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "输出模式": (["拉伸图像", "添加边距", "裁剪图像"],),
                "对齐方式": (["居中", "左上", "右上", "左下", "右下"],),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("图像", "像素宽度", "像素高度")
    FUNCTION = "correct_pixels"
    CATEGORY = "🐳Pond/图像"
    
    def correct_pixels(self, 图像, 检测模式, 像素宽度, 像素高度, 输出模式, 对齐方式):
        batch_size, height, width, channels = 图像.shape
        processed_images = []
        
        for i in range(batch_size):
            # 转换为PIL图像
            img_tensor = 图像[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            # 自动检测像素大小
            if 检测模式 == "自动检测":
                detected_width, detected_height = self._detect_pixel_size(img_pil)
                if detected_width > 0 and detected_height > 0:
                    像素宽度 = detected_width
                    像素高度 = detected_height
            
            # 计算校正后的尺寸
            if 输出模式 == "拉伸图像":
                # 计算需要的拉伸比例
                if 像素宽度 > 像素高度:
                    # 需要垂直拉伸
                    scale_factor = 像素宽度 / 像素高度
                    new_width = width
                    new_height = int(height * scale_factor)
                else:
                    # 需要水平拉伸
                    scale_factor = 像素高度 / 像素宽度
                    new_width = int(width * scale_factor)
                    new_height = height
                
                img_corrected = img_pil.resize((new_width, new_height), Image.NEAREST)
                
            elif 输出模式 == "添加边距":
                # 计算需要添加的边距
                target_ratio = 1.0  # 目标是1:1
                current_ratio = 像素宽度 / 像素高度
                
                if current_ratio > target_ratio:
                    # 像素太宽，需要添加上下边距
                    new_height = int(height * current_ratio)
                    new_width = width
                    
                    # 创建新图像
                    img_corrected = Image.new(img_pil.mode, (new_width, new_height), (0, 0, 0))
                    
                    # 根据对齐方式放置原图
                    y_offset = self._calculate_offset(new_height - height, 对齐方式, 'vertical')
                    img_corrected.paste(img_pil, (0, y_offset))
                else:
                    # 像素太高，需要添加左右边距
                    new_width = int(width / current_ratio)
                    new_height = height
                    
                    # 创建新图像
                    img_corrected = Image.new(img_pil.mode, (new_width, new_height), (0, 0, 0))
                    
                    # 根据对齐方式放置原图
                    x_offset = self._calculate_offset(new_width - width, 对齐方式, 'horizontal')
                    img_corrected.paste(img_pil, (x_offset, 0))
                    
            else:  # 裁剪图像
                # 计算裁剪尺寸
                if 像素宽度 > 像素高度:
                    # 需要裁剪宽度
                    crop_ratio = 像素高度 / 像素宽度
                    new_width = int(width * crop_ratio)
                    new_height = height
                    
                    # 根据对齐方式计算裁剪位置
                    x_offset = self._calculate_offset(width - new_width, 对齐方式, 'horizontal')
                    img_corrected = img_pil.crop((x_offset, 0, x_offset + new_width, height))
                else:
                    # 需要裁剪高度
                    crop_ratio = 像素宽度 / 像素高度
                    new_width = width
                    new_height = int(height * crop_ratio)
                    
                    # 根据对齐方式计算裁剪位置
                    y_offset = self._calculate_offset(height - new_height, 对齐方式, 'vertical')
                    img_corrected = img_pil.crop((0, y_offset, width, y_offset + new_height))
            
            # 转换回tensor
            img_np = np.array(img_corrected).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output, 像素宽度, 像素高度)
    
    def _detect_pixel_size(self, img):
        """自动检测像素大小"""
        # 将图像转换为numpy数组
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # 检测水平方向的像素大小
        pixel_width = 1
        for w in range(1, min(width // 2, 64)):
            # 检查是否所有像素都是w的倍数宽度
            is_valid = True
            for x in range(0, width - w, w):
                # 检查像素块是否一致
                block = img_array[:, x:x+w]
                if not self._is_uniform_block(block):
                    is_valid = False
                    break
            if is_valid:
                pixel_width = w
                break
        
        # 检测垂直方向的像素大小
        pixel_height = 1
        for h in range(1, min(height // 2, 64)):
            # 检查是否所有像素都是h的倍数高度
            is_valid = True
            for y in range(0, height - h, h):
                # 检查像素块是否一致
                block = img_array[y:y+h, :]
                if not self._is_uniform_block(block):
                    is_valid = False
                    break
            if is_valid:
                pixel_height = h
                break
        
        return pixel_width, pixel_height
    
    def _is_uniform_block(self, block):
        """检查像素块是否均匀"""
        if block.size == 0:
            return False
        
        # 获取第一个像素的颜色
        first_pixel = block.flat[0:block.shape[-1]]
        
        # 检查所有像素是否相同
        return np.all(block == first_pixel)
    
    def _calculate_offset(self, total_offset, alignment, direction):
        """根据对齐方式计算偏移量"""
        if alignment == "居中":
            return total_offset // 2
        elif alignment == "左上":
            return 0
        elif alignment == "右上":
            return total_offset if direction == 'horizontal' else 0
        elif alignment == "左下":
            return 0 if direction == 'horizontal' else total_offset
        elif alignment == "右下":
            return total_offset
        return 0

class PartialPixelizeNode:
    """
    局部像素化节点，通过遮罩控制像素化区域
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "像素大小": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "混合强度": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "羽化半径": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "混合模式": (["正常", "叠加", "柔光", "强光"],),
                "反转遮罩": ("BOOLEAN", {"default": False}),
                "保持颜色": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "partial_pixelize"
    CATEGORY = "🐳Pond/图像"
    
    def partial_pixelize(self, 图像, 遮罩, 像素大小, 混合强度, 羽化半径, 混合模式, 反转遮罩, 保持颜色):
        batch_size, height, width, channels = 图像.shape
        processed_images = []
        
        for i in range(batch_size):
            # 转换为PIL图像
            img_tensor = 图像[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            # 处理遮罩
            if i < 遮罩.shape[0]:
                mask_tensor = 遮罩[i]
            else:
                mask_tensor = 遮罩[0]  # 如果遮罩数量不足，使用第一个
            
            mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
            
            # 确保遮罩尺寸匹配
            if mask_pil.size != (width, height):
                mask_pil = mask_pil.resize((width, height), Image.LANCZOS)
            
            # 反转遮罩
            if 反转遮罩:
                mask_np = 255 - np.array(mask_pil)
                mask_pil = Image.fromarray(mask_np, mode='L')
            
            # 应用羽化
            if 羽化半径 > 0:
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=羽化半径))
            
            # 创建像素化版本
            pixel_width = max(1, width // 像素大小)
            pixel_height = max(1, height // 像素大小)
            
            # 缩小图像
            img_small = img_pil.resize((pixel_width, pixel_height), Image.NEAREST)
            
            # 如果保持颜色，只像素化形状
            if 保持颜色:
                # 创建亮度图
                img_gray = img_pil.convert('L')
                gray_small = img_gray.resize((pixel_width, pixel_height), Image.NEAREST)
                gray_pixelated = gray_small.resize((width, height), Image.NEAREST)
                
                # 将像素化的亮度应用到原始颜色
                img_hsv = img_pil.convert('HSV')
                h, s, v = img_hsv.split()
                img_hsv = Image.merge('HSV', (h, s, gray_pixelated))
                img_pixelated = img_hsv.convert('RGB')
            else:
                # 标准像素化
                img_pixelated = img_small.resize((width, height), Image.NEAREST)
            
            # 应用混合模式
            if 混合模式 == "正常":
                img_blended = img_pixelated
            elif 混合模式 == "叠加":
                img_blended = self._overlay_blend(img_pil, img_pixelated)
            elif 混合模式 == "柔光":
                img_blended = self._soft_light_blend(img_pil, img_pixelated)
            elif 混合模式 == "强光":
                img_blended = self._hard_light_blend(img_pil, img_pixelated)
            
            # 根据遮罩和强度混合原图和像素化图像
            if 混合强度 < 1.0:
                # 调整遮罩强度
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                mask_np = mask_np * 混合强度
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
            
            # 使用遮罩合成
            img_result = Image.composite(img_blended, img_pil, mask_pil)
            
            # 转换回tensor
            img_np = np.array(img_result).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output,)
    
    def _overlay_blend(self, base, overlay):
        """叠加混合模式"""
        base_np = np.array(base).astype(np.float32) / 255.0
        overlay_np = np.array(overlay).astype(np.float32) / 255.0
        
        # 叠加公式
        result = np.where(base_np < 0.5,
                         2 * base_np * overlay_np,
                         1 - 2 * (1 - base_np) * (1 - overlay_np))
        
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result, mode='RGB')
    
    def _soft_light_blend(self, base, overlay):
        """柔光混合模式"""
        base_np = np.array(base).astype(np.float32) / 255.0
        overlay_np = np.array(overlay).astype(np.float32) / 255.0
        
        # 柔光公式
        result = np.where(overlay_np < 0.5,
                         base_np - (1 - 2 * overlay_np) * base_np * (1 - base_np),
                         base_np + (2 * overlay_np - 1) * (np.sqrt(base_np) - base_np))
        
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result, mode='RGB')
    
    def _hard_light_blend(self, base, overlay):
        """强光混合模式"""
        base_np = np.array(base).astype(np.float32) / 255.0
        overlay_np = np.array(overlay).astype(np.float32) / 255.0
        
        # 强光公式（与叠加相反）
        result = np.where(overlay_np < 0.5,
                         2 * base_np * overlay_np,
                         1 - 2 * (1 - base_np) * (1 - overlay_np))
        
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result, mode='RGB')

class PixelArtEnhanceNode:
    """
    像素艺术增强节点，提供更多像素处理选项
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "处理模式": (["像素化", "像素校正", "像素优化"],),
                "像素大小": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "颜色量化": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "抖动": ("BOOLEAN", {"default": False}),
                "保持锐利": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "enhance"
    CATEGORY = "🐳Pond/图像"
    
    def enhance(self, 图像, 处理模式, 像素大小, 颜色量化, 抖动, 保持锐利):
        batch_size, height, width, channels = 图像.shape
        processed_images = []
        
        for i in range(batch_size):
            # 转换为PIL图像
            img_tensor = 图像[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            if 处理模式 == "像素化":
                # 标准像素化处理
                pixel_width = width // 像素大小
                pixel_height = height // 像素大小
                
                # 应用颜色量化
                if 颜色量化 > 0:
                    img_pil = img_pil.quantize(colors=颜色量化, dither=Image.FLOYDSTEINBERG if 抖动 else Image.NONE)
                    img_pil = img_pil.convert('RGB')
                
                # 缩小
                img_small = img_pil.resize((pixel_width, pixel_height), Image.NEAREST if 保持锐利 else Image.LANCZOS)
                
                # 放大
                img_processed = img_small.resize((width, height), Image.NEAREST)
                
            elif 处理模式 == "像素校正":
                # 检测并校正非正方形像素
                # 这里简化处理，直接按照高度进行缩放
                img_processed = img_pil.resize((width, width), Image.NEAREST)
                
            else:  # 像素优化
                # 优化像素艺术（去除模糊，增强边缘）
                # 增强锐度
                if 保持锐利:
                    enhancer = ImageEnhance.Sharpness(img_pil)
                    img_pil = enhancer.enhance(2.0)
                
                # 应用最近邻采样确保像素清晰
                img_processed = img_pil
                
                # 颜色量化
                if 颜色量化 > 0:
                    img_processed = img_processed.quantize(colors=颜色量化, dither=Image.FLOYDSTEINBERG if 抖动 else Image.NONE)
                    img_processed = img_processed.convert('RGB')
            
            # 转换回tensor
            img_np = np.array(img_processed).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "Pixelize": PixelizeNode,
    "SquarePixelCorrection": SquarePixelCorrectionNode,
    "PartialPixelize": PartialPixelizeNode,
    "PixelArtEnhance": PixelArtEnhanceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixelize": "🐳像素化",
    "SquarePixelCorrection": "🐳像素校正",
    "PartialPixelize": "🐳局部像素化",
    "PixelArtEnhance": "🐳像素增强"
}