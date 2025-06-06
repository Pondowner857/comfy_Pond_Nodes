import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class MaskRemoveNode:
    """
    ComfyUI节点：根据遮罩移除图像背景
    保留白色区域，移除黑色区域
    输出原图大小和裁剪后的图像
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "遮罩": ("MASK",),
                "边缘细化类型": (["无", "高斯模糊", "形态学平滑", "边缘羽化"],),
                "细化强度": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "裁剪边距": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("原尺寸图像", "裁剪图像", "使用遮罩")
    FUNCTION = "remove_background"
    CATEGORY = "🐳Pond/mask"
    
    def remove_background(self, 图像, 遮罩, 边缘细化类型, 细化强度, 裁剪边距):
        """
        根据遮罩移除图像背景，并对边缘进行细化处理
        
        Args:
            图像: 输入图像张量 (B, H, W, C)
            遮罩: 遮罩张量 (B, H, W) 或 (H, W)
            边缘细化类型: 边缘处理方式
            细化强度: 处理强度
            裁剪边距: 裁剪时的额外边距
            
        Returns:
            tuple: (原尺寸处理后的图像, 裁剪后的图像, 使用的遮罩)
        """
        # 确保输入是torch张量
        if not isinstance(图像, torch.Tensor):
            图像 = torch.tensor(图像)
        if not isinstance(遮罩, torch.Tensor):
            遮罩 = torch.tensor(遮罩)
        
        # 获取图像尺寸
        if len(图像.shape) == 4:  # (B, H, W, C)
            batch_size, height, width, channels = 图像.shape
        elif len(图像.shape) == 3:  # (H, W, C)
            图像 = 图像.unsqueeze(0)  # 添加批次维度
            batch_size, height, width, channels = 图像.shape
        else:
            raise ValueError("图像格式不正确，应为 (B, H, W, C) 或 (H, W, C)")
        
        # 处理遮罩尺寸
        if len(遮罩.shape) == 2:  # (H, W)
            遮罩 = 遮罩.unsqueeze(0)  # 添加批次维度 (B, H, W)
        elif len(遮罩.shape) == 3:  # (B, H, W)
            pass
        else:
            raise ValueError("遮罩格式不正确，应为 (H, W) 或 (B, H, W)")
        
        # 确保遮罩和图像尺寸匹配
        if 遮罩.shape[-2:] != (height, width):
            # 调整遮罩尺寸
            遮罩 = torch.nn.functional.interpolate(
                遮罩.unsqueeze(1).float(), 
                size=(height, width), 
                mode='nearest'
            ).squeeze(1)
        
        # 确保遮罩值在0-1范围内
        遮罩 = torch.clamp(遮罩, 0, 1)
        
        # 边缘细化处理
        if 边缘细化类型 != "无" and 细化强度 > 0:
            遮罩 = self.refine_mask_edges(遮罩, 边缘细化类型, 细化强度)
        
        # 创建原尺寸结果图像
        原尺寸图像 = 图像.clone()
        
        # 应用遮罩：保留白色区域(1)，移除黑色区域(0)
        for b in range(batch_size):
            for c in range(channels):
                if c < 3:  # RGB通道
                    原尺寸图像[b, :, :, c] = 图像[b, :, :, c] * 遮罩[b]
                else:  # Alpha通道
                    if channels == 4:
                        原尺寸图像[b, :, :, c] = 遮罩[b]
        
        # 如果原图像没有Alpha通道，添加Alpha通道
        if channels == 3:
            alpha_channel = 遮罩.unsqueeze(-1)  # (B, H, W, 1)
            原尺寸图像 = torch.cat([原尺寸图像, alpha_channel], dim=-1)
        
        # 创建裁剪图像列表
        裁剪图像列表 = []
        
        for b in range(batch_size):
            # 找到遮罩中非零区域的边界
            mask_b = 遮罩[b]
            非零位置 = torch.where(mask_b > 0)
            
            if len(非零位置[0]) > 0:  # 如果有非零区域
                # 计算边界框
                y_min = 非零位置[0].min().item()
                y_max = 非零位置[0].max().item()
                x_min = 非零位置[1].min().item()
                x_max = 非零位置[1].max().item()
                
                # 添加边距
                y_min = max(0, y_min - 裁剪边距)
                y_max = min(height - 1, y_max + 裁剪边距)
                x_min = max(0, x_min - 裁剪边距)
                x_max = min(width - 1, x_max + 裁剪边距)
                
                # 裁剪图像
                裁剪部分 = 原尺寸图像[b, y_min:y_max+1, x_min:x_max+1, :]
                裁剪图像列表.append(裁剪部分)
            else:
                # 如果没有非零区域，返回一个小的透明图像
                小图像 = torch.zeros(1, 1, 原尺寸图像.shape[-1], device=图像.device)
                裁剪图像列表.append(小图像)
        
        # 找到最大的裁剪尺寸，以便创建统一大小的批次
        max_h = max(img.shape[0] for img in 裁剪图像列表)
        max_w = max(img.shape[1] for img in 裁剪图像列表)
        
        # 创建统一大小的裁剪图像批次
        裁剪图像批次 = torch.zeros(batch_size, max_h, max_w, 原尺寸图像.shape[-1], device=图像.device)
        
        for b, img in enumerate(裁剪图像列表):
            h, w = img.shape[:2]
            # 将裁剪图像放在左上角
            裁剪图像批次[b, :h, :w, :] = img
        
        # 确保输出格式正确
        原尺寸图像 = torch.clamp(原尺寸图像, 0, 1)
        裁剪图像批次 = torch.clamp(裁剪图像批次, 0, 1)
        使用遮罩 = torch.clamp(遮罩, 0, 1)
        
        return (原尺寸图像, 裁剪图像批次, 使用遮罩)
    
    def refine_mask_edges(self, mask, refine_type, strength):
        """
        对遮罩边缘进行细化处理
        
        Args:
            mask: 遮罩张量
            refine_type: 细化类型
            strength: 细化强度
            
        Returns:
            refined_mask: 细化后的遮罩
        """
        if refine_type == "高斯模糊":
            return self.gaussian_blur_refine(mask, strength)
        elif refine_type == "形态学平滑":
            return self.morphological_refine(mask, strength)
        elif refine_type == "边缘羽化":
            return self.feather_edges(mask, strength)
        else:
            return mask
    
    def gaussian_blur_refine(self, mask, strength):
        """高斯模糊边缘细化"""
        # 计算模糊核大小
        kernel_size = int(strength * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 高斯模糊参数
        sigma = strength * 2.0
        
        # 添加通道维度进行模糊
        mask_blur = mask.unsqueeze(1).float()  # (B, 1, H, W)
        
        # 创建高斯核
        coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # 分别进行水平和垂直模糊
        kernel_1d = g.view(1, 1, -1, 1)
        mask_blur = F.conv2d(mask_blur, kernel_1d, padding=(kernel_size//2, 0))
        
        kernel_1d = g.view(1, 1, 1, -1)
        mask_blur = F.conv2d(mask_blur, kernel_1d, padding=(0, kernel_size//2))
        
        return mask_blur.squeeze(1)
    
    def morphological_refine(self, mask, strength):
        """形态学平滑处理"""
        kernel_size = int(strength * 3) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建形态学核
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) / (kernel_size * kernel_size)
        mask_morph = mask.unsqueeze(1).float()
        
        # 先腐蚀再膨胀（开运算）
        mask_morph = F.conv2d(mask_morph, kernel, padding=kernel_size//2)
        mask_morph = torch.clamp(mask_morph, 0, 1)
        
        # 再膨胀再腐蚀（闭运算）
        mask_morph = 1 - F.conv2d(1 - mask_morph, kernel, padding=kernel_size//2)
        mask_morph = torch.clamp(mask_morph, 0, 1)
        
        return mask_morph.squeeze(1)
    
    def feather_edges(self, mask, strength):
        """边缘羽化处理（纯PyTorch实现）"""
        # 使用多次高斯模糊实现边缘羽化效果
        feather_radius = max(1, int(strength * 5))
        
        # 创建多个不同强度的模糊版本
        mask_float = mask.unsqueeze(1).float()
        blurred_masks = []
        
        for i in range(1, feather_radius + 1):
            blur_strength = i * 0.5
            kernel_size = int(blur_strength * 4) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            sigma = blur_strength
            coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            coords -= kernel_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            
            # 水平模糊
            kernel_1d = g.view(1, 1, -1, 1)
            blurred = F.conv2d(mask_float, kernel_1d, padding=(kernel_size//2, 0))
            
            # 垂直模糊
            kernel_1d = g.view(1, 1, 1, -1)
            blurred = F.conv2d(blurred, kernel_1d, padding=(0, kernel_size//2))
            
            blurred_masks.append(blurred)
        
        # 混合不同强度的模糊结果
        if blurred_masks:
            # 使用最后一个（最强）模糊作为基础
            result = blurred_masks[-1]
            # 保持原始边缘的一些锐度
            result = mask_float * 0.3 + result * 0.7
        else:
            result = mask_float
        
        return result.squeeze(1)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "遮罩移除": MaskRemoveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "遮罩移除": "🐳遮罩移除"
}
