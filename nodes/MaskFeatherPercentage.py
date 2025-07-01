import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class MaskFeatherPercentageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "羽化百分比": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "保持锐利边缘": (["是", "否"], {"default": "否"})
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "feather_mask_percentage"
    CATEGORY = "🐳Pond/遮罩"

    def normalize_mask(self, mask):
        # 处理输入掩码的维度，确保为 (1, H, W)
        if len(mask.shape) == 2:  # (H, W)
            mask = mask.unsqueeze(0)  # (1, H, W)
        elif len(mask.shape) == 3:  # (B, H, W) 或 (1, H, W)
            if mask.shape[0] > 1:
                mask = mask[0:1]  # 只取第一个
        elif len(mask.shape) == 4:  # (B, C, H, W) 或 (B, H, W, C)
            if mask.shape[1] == 1:  # (B, 1, H, W)
                mask = mask.squeeze(1)[0:1]  # (1, H, W)
            elif mask.shape[3] == 1:  # (B, H, W, 1)
                mask = mask.squeeze(3)[0:1]  # (1, H, W)
            else:
                raise ValueError(f"不支持的掩码形状: {mask.shape}")
        
        return mask

    def feather_mask_percentage(self, mask, 羽化百分比, 保持锐利边缘):
        """基于百分比羽化遮罩边缘"""
        # 规范化掩码为 (1, H, W) 格式
        mask = self.normalize_mask(mask)
        
        # 如果百分比为0，直接返回原始掩码
        if 羽化百分比 <= 0.1:
            return (mask,)
        
        # 转为numpy以便使用更高级的图像处理
        cpu_mask = mask.cpu().numpy()[0]  # 获取为(H, W)格式
        height, width = cpu_mask.shape
        
        # 计算基于百分比的羽化半径
        # 使用遮罩的较小维度作为参考
        reference_dimension = min(height, width)
        feather_radius = int(reference_dimension * 羽化百分比 / 100.0)
        
        # 确保羽化半径至少为1像素
        feather_radius = max(1, feather_radius)
        
        # 创建二值掩码以获取边缘
        binary_mask = (cpu_mask > 0.5).astype(np.float32)
        
        if 保持锐利边缘 == "是":
            # 计算距离变换
            # 对前景和背景分别计算，然后合并
            dist_fg = ndimage.distance_transform_edt(binary_mask)
            dist_bg = ndimage.distance_transform_edt(1.0 - binary_mask)
            
            # 计算邻近边界的区域（用于羽化）
            edge_region = np.logical_and(dist_fg <= feather_radius, binary_mask > 0.5)
            
            # 将距离转换为羽化值 (线性映射)
            feathered_mask = binary_mask.copy()
            feathered_mask[edge_region] = dist_fg[edge_region] / feather_radius
            
            # 确保数值在0-1范围内
            feathered_mask = np.clip(feathered_mask, 0.0, 1.0)
        else:
            # 使用高斯模糊进行羽化
            # 首先对二值掩码应用高斯模糊
            sigma = feather_radius / 2.0  # 高斯核的标准差
            feathered_mask = ndimage.gaussian_filter(binary_mask, sigma=sigma)
            
            # 确保原始区域内的值仍然接近1
            feathered_mask = np.maximum(feathered_mask, binary_mask * 0.99)
            
            # 确保数值在0-1范围内
            feathered_mask = np.clip(feathered_mask, 0.0, 1.0)
        
        # 转回PyTorch格式
        result_mask = torch.from_numpy(feathered_mask).float().unsqueeze(0)
        
        return (result_mask,)

NODE_CLASS_MAPPINGS = {"MaskFeatherPercentageNode": MaskFeatherPercentageNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskFeatherPercentageNode": "🐳遮罩百分比羽化"}