import torch
import numpy as np
import cv2
from typing import Tuple, List

class MaskToBBoxCropper:
    """
    从遮罩提取边界框 - 输出完整遮罩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "BBOX_LIST")
    RETURN_NAMES = ("preview_image", "mask", "bboxes")
    FUNCTION = "extract_bbox"
    CATEGORY = "🐳Pond/bbox"
    OUTPUT_IS_LIST = (False, False, False)

    def extract_bbox(self, image, mask):
        """
        从遮罩提取边界框
        """
        # 处理输入维度
        if image.dim() == 4:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # 处理遮罩维度 - 保留原始遮罩
        if mask.dim() == 4:
            mask_tensor = mask[0]
            mask_np = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        elif mask.dim() == 3:
            mask_tensor = mask
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            mask_tensor = mask.unsqueeze(0)
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        # 确保图像是RGB格式
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        # 二值化遮罩
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes_list = []
        
        # 创建预览图像
        preview_img = image_np.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # 处理每个轮廓
        for idx, contour in enumerate(contours):
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 添加到列表
            bboxes_list.append([x, y, x + w, y + h])
            
            # 在预览图上绘制边界框
            color = colors[idx % len(colors)]
            cv2.rectangle(preview_img, (x, y), (x + w, y + h), color, 2)
            # 添加编号标签
            cv2.putText(preview_img, f"{idx}", (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 如果没有找到轮廓，返回整个区域
        if not contours:
            bboxes_list.append([0, 0, image_np.shape[1], image_np.shape[0]])
        
        # 转换预览图像为tensor
        preview_tensor = torch.from_numpy(preview_img.astype(np.float32) / 255.0).unsqueeze(0)
        
        # 返回原始遮罩（不是列表）
        return (preview_tensor, mask_tensor, bboxes_list)


class CropByBBox:
    """
    根据边界框裁剪图像，支持四向独立扩展和羽化
    默认裁剪所有bbox并合并到一张图
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": ("BBOX_LIST",),
                "bbox_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99,
                    "step": 1,
                    "display": "number"
                }),
                "expand_top": ("INT", {
                    "default": 0,
                    "min": -9999,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "expand_bottom": ("INT", {
                    "default": 0,
                    "min": -9999,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "expand_left": ("INT", {
                    "default": 0,
                    "min": -9999,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "expand_right": ("INT", {
                    "default": 0,
                    "min": -9999,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA")
    RETURN_NAMES = ("cropped_image", "crop_mask", "crop_data")
    FUNCTION = "crop_image"
    CATEGORY = "🐳Pond/bbox"

    def crop_image(self, image, bbox, bbox_index=-1, expand_top=0, expand_bottom=0, expand_left=0, expand_right=0, feather=0):
        """
        根据边界框裁剪图像
        bbox_index = -1 时裁剪所有bbox并合并
        bbox_index >= 0 时裁剪指定的单个bbox
        """
        # 处理图像输入
        if image.dim() == 4:
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # 确保是RGB
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        
        # 确保bbox是列表
        if not isinstance(bbox, list):
            bbox = [bbox]
        
        # 如果bbox_index为-1，处理所有bbox
        if bbox_index == -1:
            # 计算所有bbox的最小外接矩形
            if len(bbox) > 0:
                min_x = min([b[0] for b in bbox])
                min_y = min([b[1] for b in bbox])
                max_x = max([b[2] for b in bbox])
                max_y = max([b[3] for b in bbox])
                
                # 应用扩展
                min_x = max(0, min_x - expand_left)
                min_y = max(0, min_y - expand_top)
                max_x = min(img_np.shape[1], max_x + expand_right)
                max_y = min(img_np.shape[0], max_y + expand_bottom)
                
                # 保存裁剪坐标
                crop_coords = [min_x, min_y, max_x, max_y]
                
                # 创建合并的裁剪图像和遮罩
                cropped = img_np[min_y:max_y, min_x:max_x].copy()
                h, w = max_y - min_y, max_x - min_x
                crop_mask = np.zeros((h, w), dtype=np.float32)
                
                # 在遮罩中绘制所有bbox区域
                for single_bbox in bbox:
                    x1, y1, x2, y2 = single_bbox[:4]
                    # 转换到裁剪后的坐标系
                    x1 = max(0, x1 - min_x + expand_left)
                    y1 = max(0, y1 - min_y + expand_top)
                    x2 = min(w, x2 - min_x + expand_left)
                    y2 = min(h, y2 - min_y + expand_top)
                    
                    # 填充该区域
                    crop_mask[y1:y2, x1:x2] = 1.0
                
                # 应用羽化到整个遮罩
                if feather > 0 and h > feather*2 and w > feather*2:
                    # 使用距离变换创建羽化
                    crop_mask_uint8 = (crop_mask * 255).astype(np.uint8)
                    dist_transform = cv2.distanceTransform(crop_mask_uint8, cv2.DIST_L2, 5)
                    if dist_transform.max() > 0:
                        crop_mask = np.minimum(dist_transform / feather, 1.0)
                    
                    # 高斯模糊平滑
                    if feather > 2:
                        kernel_size = min(feather * 2 + 1, 51)
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        crop_mask = cv2.GaussianBlur(crop_mask, (kernel_size, kernel_size), feather/3)
            else:
                # 没有bbox时返回整个图像
                cropped = img_np.copy()
                crop_mask = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
                crop_coords = [0, 0, img_np.shape[1], img_np.shape[0]]
        
        else:
            # 处理单个bbox
            if len(bbox) > bbox_index:
                single_bbox = bbox[bbox_index]
            else:
                single_bbox = bbox[-1] if bbox else [0, 0, img_np.shape[1], img_np.shape[0]]
            
            # 获取坐标
            try:
                x1, y1, x2, y2 = single_bbox[:4]
                x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
            except:
                x1, y1, x2, y2 = 0, 0, img_np.shape[1], img_np.shape[0]
            
            # 应用扩展
            x1 = max(0, x1 - expand_left)
            y1 = max(0, y1 - expand_top)
            x2 = min(img_np.shape[1], x2 + expand_right)
            y2 = min(img_np.shape[0], y2 + expand_bottom)
            
            # 保存裁剪坐标
            crop_coords = [x1, y1, x2, y2]
            
            # 裁剪
            cropped = img_np[y1:y2, x1:x2].copy()
            h, w = y2 - y1, x2 - x1
            crop_mask = np.ones((h, w), dtype=np.float32)
            
            # 应用羽化
            if feather > 0 and h > feather*2 and w > feather*2:
                for y in range(h):
                    for x in range(w):
                        dist_to_edge = min(x, w - 1 - x, y, h - 1 - y)
                        if dist_to_edge < feather:
                            crop_mask[y, x] = dist_to_edge / feather
                
                if feather > 2:
                    kernel_size = min(feather * 2 + 1, 51)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    crop_mask = cv2.GaussianBlur(crop_mask, (kernel_size, kernel_size), feather/3)
        
        # 转换为tensor
        cropped_tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0)
        if cropped_tensor.dim() == 2:
            cropped_tensor = cropped_tensor.unsqueeze(-1).repeat(1, 1, 3)
        cropped_tensor = cropped_tensor.unsqueeze(0)
        
        mask_tensor = torch.from_numpy(crop_mask)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return (cropped_tensor, mask_tensor, crop_coords)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskToBBoxCropper": MaskToBBoxCropper,
    "CropByBBox": CropByBBox,
}

# 节点显示名称映射  
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToBBoxCropper": "🐳遮罩到bbox",
    "CropByBBox": "🐳bbox裁剪",
}