import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class MaskRegionExpandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "左": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "上": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "右": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "下": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "扩展区域": (["黑色区域", "白色区域"], {"default": "黑色区域"}),
                "边缘平滑": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "使用渐变": (["否", "是"], {"default": "否"})
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand_mask_region"
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

    def expand_mask_region(self, mask, 左, 上, 右, 下, 扩展区域, 边缘平滑=0, 使用渐变="否"):
        """使用膨胀操作扩展遮罩中的特定区域，并提供边缘平滑选项"""
        # 规范化掩码为 (1, H, W) 格式
        mask = self.normalize_mask(mask)
        
        # 如果没有需要扩展的方向，直接返回原始掩码
        if 左 == 0 and 上 == 0 and 右 == 0 and 下 == 0:
            return (mask,)
        
        # 确定是扩展黑色区域还是白色区域
        if 扩展区域 == "黑色区域":
            # 反转掩码（1变0，0变1）
            work_mask = 1.0 - mask
        else:
            work_mask = mask.clone()
        
        # 转为numpy以便使用更高级的图像处理
        cpu_mask = work_mask.cpu().numpy()[0]  # 获取为(H, W)格式
        height, width = cpu_mask.shape
        
        # 创建用于扩展的掩码和距离图
        expanded_mask = cpu_mask.copy()
        
        # 创建一个距离变换图，用于渐变效果
        if 使用渐变 == "是":
            # 计算二值掩码（阈值为0.5）
            binary_mask = (cpu_mask > 0.5).astype(np.uint8)
            # 生成距离变换
            distance_map = ndimage.distance_transform_edt(1 - binary_mask)
        
        # 处理左右方向（水平扩展）
        if 左 > 0:
            # 从右到左扫描每一行
            for y in range(height):
                # 找到该行第一个非零像素
                for x in range(width):
                    if cpu_mask[y, x] > 0.5:
                        # 向左扩展
                        start = max(0, x - 左)
                        if 使用渐变 == "是":
                            # 使用线性渐变填充
                            for i in range(start, x):
                                # 计算距离百分比
                                distance_percent = (x - i) / 左 if 左 > 0 else 0
                                # 应用渐变效果，距离越远值越小
                                expanded_mask[y, i] = max(expanded_mask[y, i], 1.0 - distance_percent)
                        else:
                            # 硬边界填充
                            expanded_mask[y, start:x] = 1
                        break
        
        if 右 > 0:
            # 从左到右扫描每一行
            for y in range(height):
                # 找到该行最后一个非零像素
                for x in range(width-1, -1, -1):
                    if cpu_mask[y, x] > 0.5:
                        # 向右扩展
                        end = min(width, x + 右 + 1)
                        if 使用渐变 == "是":
                            # 使用线性渐变填充
                            for i in range(x+1, end):
                                # 计算距离百分比
                                distance_percent = (i - x) / 右 if 右 > 0 else 0
                                # 应用渐变效果，距离越远值越小
                                expanded_mask[y, i] = max(expanded_mask[y, i], 1.0 - distance_percent)
                        else:
                            # 硬边界填充
                            expanded_mask[y, x+1:end] = 1
                        break
        
        # 处理上下方向（垂直扩展）
        if 下 > 0:  # 下表示向图像底部扩展（实际是增加y值）
            # 从上到下扫描每一列
            for x in range(width):
                # 找到该列最后一个非零像素
                for y in range(height-1, -1, -1):
                    if cpu_mask[y, x] > 0.5:
                        # 向下扩展
                        end = min(height, y + 下 + 1)
                        if 使用渐变 == "是":
                            # 使用线性渐变填充
                            for i in range(y+1, end):
                                # 计算距离百分比
                                distance_percent = (i - y) / 下 if 下 > 0 else 0
                                # 应用渐变效果
                                expanded_mask[i, x] = max(expanded_mask[i, x], 1.0 - distance_percent)
                        else:
                            # 硬边界填充
                            expanded_mask[y+1:end, x] = 1
                        break
        
        if 上 > 0:  # 上表示向图像顶部扩展（实际是减少y值）
            # 从下到上扫描每一列
            for x in range(width):
                # 找到该列第一个非零像素
                for y in range(height):
                    if cpu_mask[y, x] > 0.5:
                        # 向上扩展
                        start = max(0, y - 上)
                        if 使用渐变 == "是":
                            # 使用线性渐变填充
                            for i in range(start, y):
                                # 计算距离百分比
                                distance_percent = (y - i) / 上 if 上 > 0 else 0
                                # 应用渐变效果
                                expanded_mask[i, x] = max(expanded_mask[i, x], 1.0 - distance_percent)
                        else:
                            # 硬边界填充
                            expanded_mask[start:y, x] = 1
                        break
        
        # 应用边缘平滑（高斯模糊）
        if 边缘平滑 > 0:
            # 对扩展区域应用高斯模糊
            expanded_mask = ndimage.gaussian_filter(expanded_mask, sigma=边缘平滑/3)
            
            # 确保原始掩码区域不受影响
            if 使用渐变 != "是":  # 渐变模式已经修改了原始区域，所以不需要这一步
                expanded_mask = np.maximum(expanded_mask, cpu_mask)
        
        # 转回PyTorch格式
        result_mask = torch.from_numpy(expanded_mask).float().unsqueeze(0)
        
        # 如果是扩展黑色区域，需要再次反转掩码
        if 扩展区域 == "黑色区域":
            result_mask = 1.0 - result_mask
        
        return (result_mask,)

NODE_CLASS_MAPPINGS = {"MaskRegionExpandNode": MaskRegionExpandNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskRegionExpandNode": "🐳遮罩区域扩展"}