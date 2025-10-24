import torch
import numpy as np
from PIL import Image, ImageFilter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YoloV11ImagePasteNode:
    """YOLOv11专用图像粘贴节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"display": "原始图像"}),
                "paste_images": ("IMAGE", {"display": "粘贴图像列表"}),
                "bboxes": ("BBOXES", {"display": "边界框"}),
                "paste_mode": (["全部粘贴", "指定索引", "循环使用", "智能混合"], {
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
                }),
                "color_match": ("BOOLEAN", {
                    "default": True,
                    "display": "颜色匹配"
                }),
                "edge_blend": ("BOOLEAN", {
                    "default": True,
                    "display": "边缘混合"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("拼接图像", "合成遮罩", "处理信息")
    INPUT_IS_LIST = {"paste_images": True}
    FUNCTION = "paste_images_v11"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "YOLOv11专用粘贴节点，支持智能颜色匹配和边缘混合"

    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        if tensor.shape[0] in [1, 3, 4] and tensor.shape[0] < tensor.shape[1]:
            tensor = tensor.permute(1, 2, 0)
        
        img_np = tensor.cpu().numpy()
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
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
        """创建高级羽化遮罩"""
        width, height = size
        
        if feather_amount <= 0:
            return Image.new('L', (width, height), 255)
        
        feather_pixels = min(feather_amount, min(width, height) // 2 - 1)
        
        if width > 2 * feather_pixels and height > 2 * feather_pixels:
            mask_array = np.zeros((height, width), dtype=np.float32)
            
            # 填充中心区域
            mask_array[feather_pixels:height-feather_pixels, 
                      feather_pixels:width-feather_pixels] = 255
            
            # 创建渐变边缘
            for i in range(feather_pixels):
                alpha = (i + 1) / feather_pixels
                mask_array[i, feather_pixels:width-feather_pixels] = 255 * alpha
                mask_array[height-1-i, feather_pixels:width-feather_pixels] = 255 * alpha
                mask_array[feather_pixels:height-feather_pixels, i] = 255 * alpha
                mask_array[feather_pixels:height-feather_pixels, width-1-i] = 255 * alpha
            
            # 处理四个角落
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
            
            mask = Image.fromarray(mask_array.astype(np.uint8))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_pixels * 0.3))
        else:
            mask = Image.new('L', (width, height), 128)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_pixels))
        
        return mask

    def parse_bboxes(self, bboxes):
        """解析边界框数据"""
        bboxes_list = []
        
        if isinstance(bboxes, list):
            if len(bboxes) == 1 and isinstance(bboxes[0], list):
                first_elem = bboxes[0]
                if all(isinstance(item, (list, tuple)) and len(item) == 4 for item in first_elem):
                    bboxes_list = [list(bbox) for bbox in first_elem]
                else:
                    bboxes_list = bboxes
            else:
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
        
        return bboxes_list

    def apply_color_matching(self, paste_img, original_region):
        """应用颜色匹配"""
        # 获取边缘区域的平均颜色
        edge_size = 10
        
        orig_array = np.array(original_region)
        paste_array = np.array(paste_img)
        
        # 计算原始区域边缘的平均颜色
        orig_edge = np.concatenate([
            orig_array[:edge_size, :].flatten(),
            orig_array[-edge_size:, :].flatten(),
            orig_array[:, :edge_size].flatten(),
            orig_array[:, -edge_size:].flatten()
        ]).reshape(-1, 3).mean(axis=0)
        
        # 计算粘贴图像边缘的平均颜色
        paste_edge = np.concatenate([
            paste_array[:edge_size, :].flatten(),
            paste_array[-edge_size:, :].flatten(),
            paste_array[:, :edge_size].flatten(),
            paste_array[:, -edge_size:].flatten()
        ]).reshape(-1, 3).mean(axis=0)
        
        # 计算颜色偏移
        color_shift = orig_edge - paste_edge
        
        # 应用颜色调整
        adjusted = paste_array.astype(np.float32) + color_shift * 0.3
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return Image.fromarray(adjusted)

    def paste_single_image_v11(self, base_pil, paste_pil, bbox, feather_amount, 
                               blend_alpha, color_match, edge_blend):
        """粘贴单个图像"""
        width, height = base_pil.size
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        target_width = x2 - x1
        target_height = y2 - y1
        
        if target_width <= 0 or target_height <= 0:
            return base_pil
        
        # 调整粘贴图像大小
        paste_resized = paste_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 获取原始区域
        original_region = base_pil.crop((x1, y1, x2, y2))
        
        # 颜色匹配
        if color_match:
            paste_resized = self.apply_color_matching(paste_resized, original_region)
        
        # 边缘混合
        if edge_blend and feather_amount > 0:
            feather_mask = self.create_advanced_feather_mask((target_width, target_height), feather_amount)
            blended = Image.composite(paste_resized, original_region, feather_mask)
            
            if blend_alpha < 1.0:
                blended = Image.blend(original_region, blended, blend_alpha)
            
            base_pil.paste(blended, (x1, y1))
        else:
            if blend_alpha < 1.0:
                blended = Image.blend(original_region, paste_resized, blend_alpha)
                base_pil.paste(blended, (x1, y1))
            else:
                base_pil.paste(paste_resized, (x1, y1))
        
        return base_pil

    def paste_images_v11(self, original_image, paste_images, bboxes, paste_mode, 
                        target_index, feather_amount, blend_alpha, color_match, edge_blend):
        """执行YOLOv11图像粘贴"""
        
        # 处理参数
        if isinstance(paste_mode, list):
            paste_mode = paste_mode[0]
        if isinstance(target_index, list):
            target_index = target_index[0]
        if isinstance(feather_amount, list):
            feather_amount = feather_amount[0]
        if isinstance(blend_alpha, list):
            blend_alpha = blend_alpha[0]
        if isinstance(color_match, list):
            color_match = color_match[0]
        if isinstance(edge_blend, list):
            edge_blend = edge_blend[0]
        
        # 处理原始图像
        if isinstance(original_image, list):
            original_image = original_image[0]
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        
        # 转换为PIL
        base_pil = self.tensor_to_pil(original_image)
        width, height = base_pil.size
        
        result_pil = base_pil.copy()
        
        # 处理粘贴图像列表
        if not isinstance(paste_images, list):
            paste_images = [paste_images]
        
        # 解析边界框
        bboxes_list = self.parse_bboxes(bboxes)
        
        num_paste_images = len(paste_images)
        num_bboxes = len(bboxes_list)
        
        # 创建累积遮罩
        mask_np = np.zeros((height, width), dtype=np.float32)
        
        processed_count = 0
        
        # 根据模式执行粘贴
        if paste_mode == "智能混合":
            # 智能混合模式：根据置信度和位置自动调整
            for i in range(min(num_paste_images, num_bboxes)):
                paste_img = paste_images[i]
                bbox = bboxes_list[i]
                
                paste_pil = self.tensor_to_pil(paste_img)
                
                # 根据位置调整混合参数
                dynamic_alpha = blend_alpha * (0.8 + 0.2 * (i / max(num_bboxes, 1)))
                
                result_pil = self.paste_single_image_v11(
                    result_pil, paste_pil, bbox, feather_amount, 
                    dynamic_alpha, color_match, edge_blend
                )
                
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1.0
                
                processed_count += 1
                
        elif paste_mode == "指定索引":
            if target_index < num_paste_images and target_index < num_bboxes:
                paste_img = paste_images[target_index]
                bbox = bboxes_list[target_index]
                
                paste_pil = self.tensor_to_pil(paste_img)
                
                result_pil = self.paste_single_image_v11(
                    result_pil, paste_pil, bbox, feather_amount, 
                    blend_alpha, color_match, edge_blend
                )
                
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1.0
                
                processed_count = 1
                
        elif paste_mode == "循环使用":
            for i in range(num_bboxes):
                paste_idx = i % num_paste_images
                paste_img = paste_images[paste_idx]
                bbox = bboxes_list[i]
                
                paste_pil = self.tensor_to_pil(paste_img)
                
                result_pil = self.paste_single_image_v11(
                    result_pil, paste_pil, bbox, feather_amount, 
                    blend_alpha, color_match, edge_blend
                )
                
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1.0
                
                processed_count += 1
                
        else:  # 全部粘贴
            max_items = min(num_paste_images, num_bboxes)
            
            for i in range(max_items):
                paste_img = paste_images[i]
                bbox = bboxes_list[i]
                
                paste_pil = self.tensor_to_pil(paste_img)
                
                result_pil = self.paste_single_image_v11(
                    result_pil, paste_pil, bbox, feather_amount, 
                    blend_alpha, color_match, edge_blend
                )
                
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1.0
                
                processed_count += 1
        
        # 转换结果
        result_tensor = self.pil_to_tensor(result_pil).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        # 生成处理信息
        info_str = f"YOLOv11 粘贴完成\n"
        info_str += f"处理了 {processed_count} 个区域\n"
        info_str += f"模式: {paste_mode}\n"
        info_str += f"羽化: {feather_amount}px | 透明度: {blend_alpha}\n"
        info_str += f"颜色匹配: {'是' if color_match else '否'} | 边缘混合: {'是' if edge_blend else '否'}"
        
        return (result_tensor, mask_tensor, info_str)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloV11ImagePasteNode": YoloV11ImagePasteNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloV11ImagePasteNode": "🐳YOLOv11图像粘贴"
}