import torch
import numpy as np
from PIL import Image
import random

class RealisticNoiseNode:
    """
    将AI生成图像的噪点转换为更真实的相机噪点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "噪点强度": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "光子噪声": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "热噪声": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "读取噪声": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "颜色噪声": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "ISO模拟": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 12800,
                    "step": 100,
                    "display": "slider"
                }),
                "保留细节": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "种子": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32-1,
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_realistic_noise"
    CATEGORY = "🐳Pond/image"
    
    def add_realistic_noise(self, image, 噪点强度, 光子噪声, 热噪声, 读取噪声, 颜色噪声, ISO模拟, 保留细节, 种子=-1):
        # 设置随机种子
        if 种子 != -1:
            torch.manual_seed(种子)
            np.random.seed(种子)
            random.seed(种子)
        
        # 获取图像维度
        batch_size, height, width, channels = image.shape
        device = image.device
        
        # 复制图像以避免修改原始数据
        noisy_image = image.clone()
        
        # ISO增益模拟
        iso_factor = ISO模拟 / 100.0
        overall_strength = 噪点强度 * iso_factor * 0.1
        
        # 1. 添加光子噪声（Shot Noise）- 泊松分布
        if 光子噪声 > 0:
            # 模拟光子噪声，在暗部更明显
            luminance = torch.mean(noisy_image, dim=3, keepdim=True)
            # 暗部噪声更强
            dark_mask = 1.0 - luminance
            
            # 使用泊松分布的近似（高斯）
            photon_noise = torch.randn_like(noisy_image) * 光子噪声 * overall_strength
            photon_noise = photon_noise * dark_mask * 2.0
            noisy_image = noisy_image + photon_noise
        
        # 2. 添加热噪声（Thermal Noise）- 固定模式噪声
        if 热噪声 > 0:
            # 创建固定模式噪声
            thermal_pattern = torch.randn(1, height, width, 1, device=device) * 热噪声 * overall_strength * 0.5
            thermal_pattern = thermal_pattern.expand(batch_size, -1, -1, channels)
            
            # 添加轻微的时间变化
            temporal_variation = torch.randn_like(noisy_image) * 热噪声 * overall_strength * 0.1
            thermal_noise = thermal_pattern + temporal_variation
            
            noisy_image = noisy_image + thermal_noise
        
        # 3. 添加读取噪声（Read Noise）- 高斯分布
        if 读取噪声 > 0:
            read_noise = torch.randn_like(noisy_image) * 读取噪声 * overall_strength * 0.8
            noisy_image = noisy_image + read_noise
        
        # 4. 添加颜色噪声（Color Noise）
        if 颜色噪声 > 0:
            # 为每个颜色通道添加不同强度的噪声
            color_noise = torch.randn_like(noisy_image)
            # R通道噪声稍强
            color_noise[:, :, :, 0] *= 颜色噪声 * overall_strength * 1.2
            # G通道噪声标准
            color_noise[:, :, :, 1] *= 颜色噪声 * overall_strength * 1.0
            # B通道噪声最强（模拟传感器特性）
            color_noise[:, :, :, 2] *= 颜色噪声 * overall_strength * 1.4
            
            noisy_image = noisy_image + color_noise
        
        # 5. 模拟传感器响应非线性
        # 在高光区域减少噪声（模拟传感器饱和）
        highlights = torch.clamp(luminance - 0.8, 0, 1) * 5.0
        noisy_image = torch.lerp(noisy_image, image, highlights)
        
        # 6. 应用细节保留
        if 保留细节 > 0:
            # 使用边缘检测保留细节
            # 简单的边缘检测
            dx = torch.abs(image[:, 1:, :, :] - image[:, :-1, :, :])
            dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
            
            # 创建边缘掩码
            edge_mask = torch.zeros_like(image)
            edge_mask[:, 1:, :, :] += dx
            edge_mask[:, :-1, :, :] += dx
            edge_mask[:, :, 1:, :] += dy
            edge_mask[:, :, :-1, :] += dy
            
            edge_mask = torch.clamp(edge_mask * 5.0, 0, 1)
            
            # 在边缘区域混合原始图像以保留细节
            noisy_image = torch.lerp(noisy_image, image, edge_mask * 保留细节)
        
        # 7. 添加轻微的椒盐噪声（Salt and Pepper）
        if random.random() < 0.3:  # 30%概率添加椒盐噪声
            salt_pepper_prob = 0.001 * overall_strength
            salt_mask = torch.rand_like(noisy_image[:, :, :, 0]) < salt_pepper_prob
            pepper_mask = torch.rand_like(noisy_image[:, :, :, 0]) < salt_pepper_prob
            
            salt_mask = salt_mask.unsqueeze(-1).expand(-1, -1, -1, channels)
            pepper_mask = pepper_mask.unsqueeze(-1).expand(-1, -1, -1, channels)
            
            noisy_image[salt_mask] = 1.0
            noisy_image[pepper_mask] = 0.0
        
        # 8. 最终处理
        # 添加轻微的高斯模糊以模拟传感器的低通滤波效果
        if overall_strength > 0.5:
            # 简单的3x3高斯模糊
            kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=device) / 16.0
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            
            # 对每个通道应用模糊
            blurred = torch.zeros_like(noisy_image)
            for c in range(channels):
                channel = noisy_image[:, :, :, c:c+1].permute(0, 3, 1, 2)
                blurred_channel = torch.nn.functional.conv2d(channel, kernel, padding=1)
                blurred[:, :, :, c] = blurred_channel.permute(0, 2, 3, 1).squeeze(-1)
            
            # 轻微混合模糊效果
            blur_strength = min(0.3, overall_strength * 0.2)
            noisy_image = torch.lerp(noisy_image, blurred, blur_strength)
        
        # 裁剪到有效范围
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return (noisy_image,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 如果种子是-1（随机），则每次都标记为已更改
        if kwargs.get('种子', -1) == -1:
            return float("NaN")
        return None


# 用于注册节点
NODE_CLASS_MAPPINGS = {
    "RealisticNoiseNode": RealisticNoiseNode
}

# 节点在UI中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "RealisticNoiseNode": "🐳噪点调节"
}