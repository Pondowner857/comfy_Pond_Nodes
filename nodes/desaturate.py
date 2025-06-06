import torch
import numpy as np

class DesaturateImage:
    """
    图像去色节点 - 模拟Photoshop的去色效果
    将彩色图像转换为灰度图像（保持RGB格式）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "去色方法": (["亮度", "平均", "最大值", "最小值"], {
                    "default": "亮度"
                }),
                "去色强度": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "desaturate"
    CATEGORY = "🐳Pond/颜色"
    
    def desaturate(self, image, 去色方法="亮度", 去色强度=1.0):
        """
        执行去色操作
        
        Args:
            image: 输入图像张量 (B, H, W, C)
            去色方法: 去色方法
            去色强度: 混合因子，控制去色强度
        
        Returns:
            去色后的图像
        """
        # 确保输入是正确的格式
        batch_size, height, width, channels = image.shape
        
        # 克隆图像以避免修改原始数据
        result = image.clone()
        
        if 去色方法 == "亮度":
            # 使用ITU-R BT.709标准的亮度权重（类似Photoshop）
            # 这些权重考虑了人眼对不同颜色的敏感度
            gray = 0.2126 * image[:, :, :, 0] + \
                   0.7152 * image[:, :, :, 1] + \
                   0.0722 * image[:, :, :, 2]
        
        elif 去色方法 == "平均":
            # 简单平均法
            gray = (image[:, :, :, 0] + image[:, :, :, 1] + image[:, :, :, 2]) / 3.0
        
        elif 去色方法 == "最大值":
            # 使用最大通道值
            gray = torch.max(image[:, :, :, :3], dim=3)[0]
        
        elif 去色方法 == "最小值":
            # 使用最小通道值
            gray = torch.min(image[:, :, :, :3], dim=3)[0]
        
        # 将灰度值扩展到所有通道
        gray = gray.unsqueeze(3)
        
        # 应用到RGB通道
        for i in range(3):
            result[:, :, :, i] = gray[:, :, :, 0]
        
        # 如果有Alpha通道，保持不变
        if channels == 4:
            result[:, :, :, 3] = image[:, :, :, 3]
        
        # 根据混合因子混合原图和去色结果
        if 去色强度 < 1.0:
            result = image * (1 - 去色强度) + result * 去色强度
        
        return (result,)


class DesaturateImageAdvanced:
    """
    高级图像去色节点 - 提供更多控制选项
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "R": ("FLOAT", {
                    "default": 0.2126,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "G": ("FLOAT", {
                    "default": 0.7152,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "B": ("FLOAT", {
                    "default": 0.0722,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "归一化": ("BOOLEAN", {"default": True}),
                "去色强度": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "desaturate_advanced"
    CATEGORY = "🐳Pond/颜色"
    
    def desaturate_advanced(self, image, R=0.2126, G=0.7152, 
                          B=0.0722, 归一化=True, 去色强度=1.0):
        """
        使用自定义权重执行去色操作
        """
        batch_size, height, width, channels = image.shape
        result = image.clone()
        
        # 归一化权重
        if 归一化:
            total_weight = R + G + B
            if total_weight > 0:
                R /= total_weight
                G /= total_weight
                B /= total_weight
        
        # 计算灰度值
        gray = (R * image[:, :, :, 0] + 
                G * image[:, :, :, 1] + 
                B * image[:, :, :, 2])
        
        # 将灰度值扩展到所有通道
        gray = gray.unsqueeze(3)
        
        # 应用到RGB通道
        for i in range(3):
            result[:, :, :, i] = gray[:, :, :, 0]
        
        # 保持Alpha通道
        if channels == 4:
            result[:, :, :, 3] = image[:, :, :, 3]
        
        # 混合原图和结果
        if 去色强度 < 1.0:
            result = image * (1 - 去色强度) + result * 去色强度
        
        return (result,)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "DesaturateImage": DesaturateImage,
    "DesaturateImageAdvanced": DesaturateImageAdvanced,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "DesaturateImage": "🐳图像去色",
    "DesaturateImageAdvanced": "🐳图像去色(V2)",
}