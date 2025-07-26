import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YoloImagePasteNode:
    """
    与YOLO检测节点配套的拼接节点
    将处理后的图像粘贴回原始位置
    支持列表输入，输出单张合成图像
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"display": "原始图像"}),
                "paste_images": ("IMAGE", {"display": "粘贴图像列表"}),
                "bboxes": ("BBOXES", {"display": "边界框"}),
                "paste_mode": (["全部粘贴", "指定索引", "循环使用"], {
                    "default": "全部粘贴",
                    "display": "粘贴模式"
                }),
                "target_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1,
                    "display": "目标索引"
                }),
                "feather_amount": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "display": "羽化程度(像素)"
                }),
                "blend_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "混合透明度"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("拼接图像", "合成遮罩")
    INPUT_IS_LIST = {"paste_images": True}  # 标记paste_images接收列表
    FUNCTION = "paste_images"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "将处理后的图像列表粘贴回YOLO检测的原始位置，支持羽化混合避免明显边缘。"

    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # 从 (C, H, W) 或 (H, W, C) 转换为 (H, W, C)
        if tensor.shape[0] in [1, 3, 4] and tensor.shape[0] < tensor.shape[1]:
            tensor = tensor.permute(1, 2, 0)
        
        # 转换为numpy
        img_np = tensor.cpu().numpy()
        
        # 确保值在0-255范围内
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # 处理通道数
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        
        return Image.fromarray(img_np, 'RGB')

    def pil_to_tensor(self, pil_image):
        """将PIL图像转换为tensor"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(np_image)

    def create_advanced_feather_mask(self, size, feather_amount):
        """创建高级羽化遮罩，边缘更自然"""
        width, height = size
        
        # 创建基础遮罩
        mask = Image.new('L', (width, height), 0)
        
        if feather_amount <= 0:
            # 没有羽化，返回纯白遮罩
            return Image.new('L', (width, height), 255)
        
        # 创建一个更小的白色核心区域
        feather_pixels = min(feather_amount, min(width, height) // 2 - 1)
        
        if width > 2 * feather_pixels and height > 2 * feather_pixels:
            # 创建内部白色区域
            inner_width = width - 2 * feather_pixels
            inner_height = height - 2 * feather_pixels
            
            # 使用numpy创建渐变
            mask_array = np.zeros((height, width), dtype=np.float32)
            
            # 填充中心区域
            mask_array[feather_pixels:height-feather_pixels, 
                      feather_pixels:width-feather_pixels] = 255
            
            # 创建渐变边缘
            for i in range(feather_pixels):
                alpha = (i + 1) / feather_pixels
                # 上边
                mask_array[i, feather_pixels:width-feather_pixels] = 255 * alpha
                # 下边
                mask_array[height-1-i, feather_pixels:width-feather_pixels] = 255 * alpha
                # 左边
                mask_array[feather_pixels:height-feather_pixels, i] = 255 * alpha
                # 右边
                mask_array[feather_pixels:height-feather_pixels, width-1-i] = 255 * alpha
            
            # 处理四个角落 - 使用圆形渐变
            for y in range(feather_pixels):
                for x in range(feather_pixels):
                    # 左上角
                    dist = np.sqrt((feather_pixels - x) ** 2 + (feather_pixels - y) ** 2)
                    alpha = max(0, 1 - dist / feather_pixels)
                    mask_array[y, x] = 255 * alpha
                    
                    # 右上角
                    dist = np.sqrt((x + 1) ** 2 + (feather_pixels - y) ** 2)
                    alpha = max(0, 1 - dist / feather_pixels)
                    mask_array[y, width - feather_pixels + x] = 255 * alpha
                    
                    # 左下角
                    dist = np.sqrt((feather_pixels - x) ** 2 + (y + 1) ** 2)
                    alpha = max(0, 1 - dist / feather_pixels)
                    mask_array[height - feather_pixels + y, x] = 255 * alpha
                    
                    # 右下角
                    dist = np.sqrt((x + 1) ** 2 + (y + 1) ** 2)
                    alpha = max(0, 1 - dist / feather_pixels)
                    mask_array[height - feather_pixels + y, width - feather_pixels + x] = 255 * alpha
            
            # 转换为PIL图像
            mask = Image.fromarray(mask_array.astype(np.uint8))
            
            # 额外的高斯模糊使过渡更平滑
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_pixels * 0.3))
        else:
            # 图像太小，使用简单的渐变
            mask = Image.new('L', (width, height), 128)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_pixels))
        
        return mask

    def parse_bboxes(self, bboxes):
        """解析各种格式的边界框数据"""
        bboxes_list = []
        
        logger.info(f"解析边界框，原始类型: {type(bboxes)}")
        
        # 处理嵌套列表的情况
        if isinstance(bboxes, list):
            if len(bboxes) == 1 and isinstance(bboxes[0], list):
                # [[bbox1, bbox2, ...]] 格式
                first_elem = bboxes[0]
                if all(isinstance(item, (list, tuple)) and len(item) == 4 for item in first_elem):
                    bboxes_list = [list(bbox) for bbox in first_elem]
                    logger.info(f"解包边界框列表，得到 {len(bboxes_list)} 个边界框")
                else:
                    bboxes_list = bboxes
            else:
                # 直接是边界框列表
                for bbox in bboxes:
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        bboxes_list.append(list(bbox))
                    elif isinstance(bbox, torch.Tensor):
                        bboxes_list.append(bbox.tolist())
        elif isinstance(bboxes, torch.Tensor):
            if bboxes.dim() == 2:
                bboxes_list = bboxes.tolist()
            elif bboxes.dim() == 1:
                bboxes_list = [bboxes.tolist()]
        
        logger.info(f"解析后得到 {len(bboxes_list)} 个边界框")
        return bboxes_list

    def paste_single_image(self, base_pil, paste_pil, bbox, feather_amount, blend_alpha):
        """使用PIL将单个图像粘贴到指定位置"""
        width, height = base_pil.size
        
        # 解析边界框（假设是像素坐标）
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 确保坐标在有效范围内
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        target_width = x2 - x1
        target_height = y2 - y1
        
        if target_width <= 0 or target_height <= 0:
            logger.warning(f"无效的边界框: [{x1},{y1},{x2},{y2}]")
            return base_pil
        
        logger.info(f"粘贴到区域: [{x1},{y1},{x2},{y2}] (尺寸: {target_width}x{target_height})")
        
        # 调整粘贴图像大小
        paste_resized = paste_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 获取原始区域
        original_region = base_pil.crop((x1, y1, x2, y2))
        
        if feather_amount > 0:
            # 创建高级羽化遮罩
            feather_mask = self.create_advanced_feather_mask((target_width, target_height), feather_amount)
            
            # 颜色匹配 - 在边缘区域匹配颜色
            edge_size = min(10, feather_amount // 2)
            if edge_size > 2:
                # 获取边缘区域的平均颜色
                edge_region = original_region.resize((edge_size * 2, edge_size * 2), Image.Resampling.LANCZOS)
                edge_array = np.array(edge_region).mean(axis=(0, 1))
                
                paste_array = np.array(paste_resized)
                paste_edge = paste_array.copy()
                
                # 只在边缘区域获取颜色
                if paste_array.shape[0] > edge_size * 2 and paste_array.shape[1] > edge_size * 2:
                    paste_edge = np.concatenate([
                        paste_array[:edge_size, :].flatten(),
                        paste_array[-edge_size:, :].flatten(),
                        paste_array[:, :edge_size].flatten(),
                        paste_array[:, -edge_size:].flatten()
                    ]).reshape(-1, 3).mean(axis=0)
                else:
                    paste_edge = paste_array.mean(axis=(0, 1))
                
                # 计算颜色偏移
                color_shift = edge_array - paste_edge
                
                # 应用颜色调整，但只在边缘区域
                adjusted_paste = paste_array.astype(np.float32)
                
                # 创建颜色调整遮罩（边缘强，中心弱）
                color_mask = 1 - np.array(feather_mask) / 255.0
                color_mask = np.stack([color_mask] * 3, axis=-1)
                
                # 应用颜色调整
                adjusted_paste += color_shift * color_mask * 0.5  # 0.5是调整强度
                adjusted_paste = np.clip(adjusted_paste, 0, 255).astype(np.uint8)
                
                paste_resized = Image.fromarray(adjusted_paste)
            
            # 使用遮罩混合
            blended = Image.composite(paste_resized, original_region, feather_mask)
            
            # 如果blend_alpha < 1，进一步混合
            if blend_alpha < 1.0:
                blended = Image.blend(original_region, blended, blend_alpha)
            
            # 粘贴回原图
            base_pil.paste(blended, (x1, y1))
        else:
            # 直接粘贴或简单混合
            if blend_alpha < 1.0:
                blended = Image.blend(original_region, paste_resized, blend_alpha)
                base_pil.paste(blended, (x1, y1))
            else:
                base_pil.paste(paste_resized, (x1, y1))
        
        return base_pil

    def paste_images(self, original_image, paste_images, bboxes, paste_mode, 
                    target_index, feather_amount, blend_alpha):
        """执行图像拼接"""
        
        # 处理参数（可能是列表）
        if isinstance(paste_mode, list):
            paste_mode = paste_mode[0]
        if isinstance(target_index, list):
            target_index = target_index[0]
        if isinstance(feather_amount, list):
            feather_amount = feather_amount[0]
        if isinstance(blend_alpha, list):
            blend_alpha = blend_alpha[0]
        
        # 处理原始图像
        if isinstance(original_image, list):
            original_image = original_image[0]
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        
        # 转换原始图像为PIL
        base_pil = self.tensor_to_pil(original_image)
        width, height = base_pil.size
        
        # 创建工作副本
        result_pil = base_pil.copy()
        
        # 处理粘贴图像列表
        if not isinstance(paste_images, list):
            paste_images = [paste_images]
        
        # 解析边界框
        bboxes_list = self.parse_bboxes(bboxes)
        
        num_paste_images = len(paste_images)
        num_bboxes = len(bboxes_list)
        
        logger.info(f"\n粘贴参数:")
        logger.info(f"- 粘贴图像数量: {num_paste_images}")
        logger.info(f"- 边界框数量: {num_bboxes}")
        logger.info(f"- 羽化程度: {feather_amount}")
        logger.info(f"- 混合透明度: {blend_alpha}")
        
        # 创建累积遮罩（用于输出）
        mask_np = np.zeros((height, width), dtype=np.float32)
        
        # 根据模式执行粘贴
        if paste_mode == "指定索引":
            if target_index < num_paste_images and target_index < num_bboxes:
                paste_img = paste_images[target_index]
                bbox = bboxes_list[target_index]
                
                # 转换为PIL
                paste_pil = self.tensor_to_pil(paste_img)
                
                # 执行粘贴
                result_pil = self.paste_single_image(
                    result_pil, paste_pil, bbox, feather_amount, blend_alpha
                )
                
                # 更新遮罩
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1.0
        
        elif paste_mode == "循环使用":
            for i in range(num_bboxes):
                paste_idx = i % num_paste_images
                paste_img = paste_images[paste_idx]
                bbox = bboxes_list[i]
                
                logger.info(f"\n粘贴第 {i+1}/{num_bboxes} 个区域（使用图像 {paste_idx+1}）")
                
                # 转换为PIL
                paste_pil = self.tensor_to_pil(paste_img)
                
                # 执行粘贴
                result_pil = self.paste_single_image(
                    result_pil, paste_pil, bbox, feather_amount, blend_alpha
                )
                
                # 更新遮罩
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1.0
        
        else:  # 全部粘贴
            max_items = min(num_paste_images, num_bboxes)
            
            for i in range(max_items):
                paste_img = paste_images[i]
                bbox = bboxes_list[i]
                
                logger.info(f"\n粘贴第 {i+1}/{max_items} 个图像")
                logger.info(f"边界框: {bbox}")
                
                # 转换为PIL
                paste_pil = self.tensor_to_pil(paste_img)
                
                # 执行粘贴
                result_pil = self.paste_single_image(
                    result_pil, paste_pil, bbox, feather_amount, blend_alpha
                )
                
                # 更新遮罩
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1.0
        
        # 将结果转换回tensor
        result_tensor = self.pil_to_tensor(result_pil).unsqueeze(0)
        
        # 转换遮罩为tensor
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        logger.info(f"\n粘贴完成")
        logger.info(f"输出图像尺寸: {result_tensor.shape}")
        logger.info(f"输出遮罩尺寸: {mask_tensor.shape}")
        
        return (result_tensor, mask_tensor)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloImagePasteNode": YoloImagePasteNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloImagePasteNode": "🐳YOLO图像拼接"
}