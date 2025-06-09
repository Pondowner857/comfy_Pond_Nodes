import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

class ImageAlignByMask:
    """
    ComfyUI插件：基于遮罩对齐的图像定位
    根据遮罩对齐方式，同步调整对应的图像位置和尺寸
    扩展区域填充白色、黑色或透明
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基准遮罩": ("MASK",),
                "遮罩2": ("MASK",),
                "图像2": ("IMAGE",),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐", 
                           "左上对齐", "右上对齐", "左下对齐", "右下对齐"], 
                          {"default": "居中对齐"}),
            },
            "optional": {
                "X轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "Y轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "填充颜色": (["白色", "黑色", "透明"], {"default": "白色"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("定位后图像", "基准遮罩", "对齐后遮罩")
    FUNCTION = "align_image_by_mask"
    CATEGORY = "🐳Pond/image"
    OUTPUT_NODE = False
    
    def get_mask_bounds(self, mask: torch.Tensor) -> Tuple[int, int, int, int]:
        """获取遮罩中非零区域的边界"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        coords = torch.nonzero(mask > 0.01)
        
        if coords.numel() == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        
        min_y, min_x = coords.min(dim=0)[0]
        max_y, max_x = coords.max(dim=0)[0]
        
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)
    
    def calculate_alignment_transform(self, base_bounds: Tuple[int, int, int, int], 
                                    mask2_bounds: Tuple[int, int, int, int], 
                                    base_size: Tuple[int, int],
                                    mask2_size: Tuple[int, int],
                                    alignment: str, 
                                    offset_x: int = 0, 
                                    offset_y: int = 0) -> Tuple[int, int]:
        """计算对齐所需的变换参数"""
        base_h, base_w = base_size
        mask2_h, mask2_w = mask2_size
        base_x, base_y, base_w_content, base_h_content = base_bounds
        mask2_x, mask2_y, mask2_w_content, mask2_h_content = mask2_bounds
        
        # 根据对齐方式计算偏移
        if alignment == "居中对齐":
            base_center_x = base_x + base_w_content // 2
            base_center_y = base_y + base_h_content // 2
            mask2_center_x = mask2_x + mask2_w_content // 2
            mask2_center_y = mask2_y + mask2_h_content // 2
            
            place_x = base_center_x - mask2_center_x
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "左对齐":
            place_x = base_x - mask2_x
            base_center_y = base_y + base_h_content // 2
            mask2_center_y = mask2_y + mask2_h_content // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "右对齐":
            place_x = (base_x + base_w_content) - (mask2_x + mask2_w_content)
            base_center_y = base_y + base_h_content // 2
            mask2_center_y = mask2_y + mask2_h_content // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "上对齐":
            base_center_x = base_x + base_w_content // 2
            mask2_center_x = mask2_x + mask2_w_content // 2
            place_x = base_center_x - mask2_center_x
            place_y = base_y - mask2_y
            
        elif alignment == "下对齐":
            base_center_x = base_x + base_w_content // 2
            mask2_center_x = mask2_x + mask2_w_content // 2
            place_x = base_center_x - mask2_center_x
            place_y = (base_y + base_h_content) - (mask2_y + mask2_h_content)
            
        elif alignment == "左上对齐":
            place_x = base_x - mask2_x
            place_y = base_y - mask2_y
            
        elif alignment == "右上对齐":
            place_x = (base_x + base_w_content) - (mask2_x + mask2_w_content)
            place_y = base_y - mask2_y
            
        elif alignment == "左下对齐":
            place_x = base_x - mask2_x
            place_y = (base_y + base_h_content) - (mask2_y + mask2_h_content)
            
        elif alignment == "右下对齐":
            place_x = (base_x + base_w_content) - (mask2_x + mask2_w_content)
            place_y = (base_y + base_h_content) - (mask2_y + mask2_h_content)
        
        # 应用用户偏移
        place_x += offset_x
        place_y += offset_y
        
        return place_x, place_y
    
    def apply_transform_to_image(self, image: torch.Tensor, 
                               place_x: int, place_y: int,
                               target_height: int, target_width: int,
                               fill_color: str) -> torch.Tensor:
        """将变换应用到图像"""
        batch, h, w, c = image.shape
        
        # 设置填充值
        if fill_color == "白色":
            fill_value = 1.0
        elif fill_color == "黑色":
            fill_value = 0.0
        else:  # 透明
            fill_value = 0.0
            # 如果是透明，确保有alpha通道
            if c == 3:
                alpha = torch.ones((batch, h, w, 1), dtype=image.dtype, device=image.device)
                image = torch.cat([image, alpha], dim=3)
                c = 4
        
        # 创建输出图像
        output = torch.full((batch, target_height, target_width, c), 
                          fill_value, dtype=image.dtype, device=image.device)
        
        # 如果是透明模式，设置alpha通道
        if fill_color == "透明" and c == 4:
            output[:, :, :, 3] = 0.0  # 背景透明
        
        # 计算复制区域
        src_x_start = max(0, -place_x)
        src_y_start = max(0, -place_y)
        src_x_end = min(w, target_width - place_x)
        src_y_end = min(h, target_height - place_y)
        
        dst_x_start = max(0, place_x)
        dst_y_start = max(0, place_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # 复制图像内容
        if src_x_end > src_x_start and src_y_end > src_y_start:
            output[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                image[:, src_y_start:src_y_end, src_x_start:src_x_end]
            
            # 如果是透明模式，设置复制区域的alpha为1
            if fill_color == "透明" and c == 4:
                output[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, 3] = 1.0
        
        return output
    
    def apply_transform_to_mask(self, mask: torch.Tensor,
                              place_x: int, place_y: int,
                              target_height: int, target_width: int) -> torch.Tensor:
        """将变换应用到遮罩"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # 创建输出遮罩
        output = torch.zeros((target_height, target_width), dtype=mask.dtype, device=mask.device)
        
        h, w = mask.shape
        
        # 计算复制区域
        src_x_start = max(0, -place_x)
        src_y_start = max(0, -place_y)
        src_x_end = min(w, target_width - place_x)
        src_y_end = min(h, target_height - place_y)
        
        dst_x_start = max(0, place_x)
        dst_y_start = max(0, place_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # 复制遮罩内容
        if src_x_end > src_x_start and src_y_end > src_y_start:
            output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                mask[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return output
    
    def align_image_by_mask(self, 基准遮罩, 遮罩2, 图像2, 对齐方式, 
                           X轴偏移=0, Y轴偏移=0, 填充颜色="白色"):
        """
        主函数：根据遮罩对齐方式调整图像
        """
        # 获取基准遮罩尺寸
        if len(基准遮罩.shape) > 2:
            base_h, base_w = 基准遮罩.shape[1:3]
        else:
            base_h, base_w = 基准遮罩.shape
        
        # 获取遮罩2尺寸
        if len(遮罩2.shape) > 2:
            mask2_h, mask2_w = 遮罩2.shape[1:3]
        else:
            mask2_h, mask2_w = 遮罩2.shape
        
        # 获取内容边界
        base_bounds = self.get_mask_bounds(基准遮罩)
        mask2_bounds = self.get_mask_bounds(遮罩2)
        
        print(f"基准遮罩尺寸: {base_h}x{base_w}")
        print(f"基准遮罩内容边界: x={base_bounds[0]}, y={base_bounds[1]}, w={base_bounds[2]}, h={base_bounds[3]}")
        print(f"遮罩2尺寸: {mask2_h}x{mask2_w}")
        print(f"遮罩2内容边界: x={mask2_bounds[0]}, y={mask2_bounds[1]}, w={mask2_bounds[2]}, h={mask2_bounds[3]}")
        print(f"对齐方式: {对齐方式}")
        
        # 计算对齐变换
        place_x, place_y = self.calculate_alignment_transform(
            base_bounds, mask2_bounds,
            (base_h, base_w), (mask2_h, mask2_w),
            对齐方式, X轴偏移, Y轴偏移
        )
        
        print(f"计算得到的偏移: x={place_x}, y={place_y}")
        
        # 应用变换到图像2
        aligned_image2 = self.apply_transform_to_image(
            图像2, place_x, place_y, base_h, base_w, 填充颜色
        )
        
        # 应用变换到遮罩2
        aligned_mask2 = self.apply_transform_to_mask(
            遮罩2, place_x, place_y, base_h, base_w
        )
        
        # 确保遮罩输出维度正确
        if len(基准遮罩.shape) == 3:
            if len(aligned_mask2.shape) == 2:
                aligned_mask2 = aligned_mask2.unsqueeze(0)
        
        print(f"输出图像尺寸: {aligned_image2.shape}")
        print(f"输出遮罩尺寸: {aligned_mask2.shape}")
        
        return (aligned_image2, 基准遮罩, aligned_mask2)


class ImageAlignByMaskBatch:
    """
    批量版本：支持多个图像同时对齐
    """
    
    def __init__(self):
        self.aligner = ImageAlignByMask()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基准遮罩": ("MASK",),
                "遮罩2": ("MASK",),
                "图像2": ("IMAGE",),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐", 
                           "左上对齐", "右上对齐", "左下对齐", "右下对齐"], 
                          {"default": "居中对齐"}),
            },
            "optional": {
                "遮罩3": ("MASK",),
                "图像3": ("IMAGE",),
                "遮罩4": ("MASK",),
                "图像4": ("IMAGE",),
                "X轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "Y轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "填充颜色": (["白色", "黑色", "透明"], {"default": "白色"}),
                "合并模式": (["分别输出", "合并输出"], {"default": "分别输出"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("定位图像2", "定位图像3", "定位图像4", "合并遮罩")
    FUNCTION = "align_multiple_images"
    CATEGORY = "🐳Pond/image"
    
    def merge_images_with_masks(self, images: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """使用遮罩合并多个图像"""
        if len(images) == 0:
            return None
            
        result = images[0].clone()
        
        for i in range(1, len(images)):
            if i < len(masks):
                mask = masks[i]
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0).unsqueeze(-1)
                elif len(mask.shape) == 3:
                    mask = mask.unsqueeze(-1)
                
                # 使用遮罩混合图像
                result = result * (1 - mask) + images[i] * mask
        
        return result
    
    def align_multiple_images(self, 基准遮罩, 遮罩2, 图像2, 对齐方式,
                            遮罩3=None, 图像3=None, 
                            遮罩4=None, 图像4=None,
                            X轴偏移=0, Y轴偏移=0, 填充颜色="白色", 合并模式="分别输出"):
        """对齐多个图像"""
        
        # 收集所有需要对齐的图像和遮罩对
        mask_image_pairs = [(遮罩2, 图像2)]
        if 遮罩3 is not None and 图像3 is not None:
            mask_image_pairs.append((遮罩3, 图像3))
        if 遮罩4 is not None and 图像4 is not None:
            mask_image_pairs.append((遮罩4, 图像4))
        
        # 对齐所有图像
        aligned_images = []
        aligned_masks = [基准遮罩]
        
        # 对齐每个图像
        for mask, image in mask_image_pairs:
            aligned_img, _, aligned_mask = self.aligner.align_image_by_mask(
                基准遮罩, mask, image, 对齐方式, X轴偏移, Y轴偏移, 填充颜色
            )
            aligned_images.append(aligned_img)
            aligned_masks.append(aligned_mask)
        
        # 创建合并遮罩
        merged_mask = aligned_masks[0].clone()
        for mask in aligned_masks[1:]:
            merged_mask = torch.maximum(merged_mask, mask)
        
        # 如果是合并输出模式，合并所有图像
        if 合并模式 == "合并输出" and len(aligned_images) > 1:
            # 使用对应的遮罩合并图像
            masks_for_merge = aligned_masks[1:]  # 跳过基准遮罩
            merged_image = self.merge_images_with_masks(aligned_images, masks_for_merge)
            # 将第一个输出替换为合并的图像
            aligned_images[0] = merged_image
        
        # 填充空输出
        empty_image = torch.zeros_like(aligned_images[0])
        while len(aligned_images) < 3:
            aligned_images.append(empty_image)
        
        return tuple(aligned_images[:3] + [merged_mask])


NODE_CLASS_MAPPINGS = {
    "ImageAlignByMask": ImageAlignByMask,
    "ImageAlignByMaskBatch": ImageAlignByMaskBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAlignByMask": "🐳图像扩展",
    "ImageAlignByMaskBatch": "🐳图像扩展(V2)"
}