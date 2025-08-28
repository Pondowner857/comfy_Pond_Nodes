import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

class AdvancedMaskImageComposite:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE", {"tooltip": "背景图像 - 作为底层的图像"}),  
                "subject_image": ("IMAGE", {"tooltip": "主体图像 - 要拼接的图像"}),     
                "subject_mask": ("MASK", {"tooltip": "主体遮罩 - 用于抠取主体的遮罩"}),
                "position_mask": ("MASK", {"tooltip": "位置遮罩 - 白色区域表示拼接位置"}),       
                "alignment": (["中", "上", "下", "左", "右"], {
                    "default": "中",
                    "tooltip": "对齐方式 - 在目标区域内的对齐位置"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "缩放因子 - 控制主体相对于目标区域的大小"
                }),
            }
        }
    
    # 添加输入参数的中文显示名称
    @classmethod
    def INPUT_NAMES(cls):
        return {
            "background_image": "背景图像",
            "subject_image": "主体图像", 
            "subject_mask": "主体遮罩",
            "position_mask": "位置遮罩",
            "alignment": "对齐方式",
            "scale_factor": "缩放因子",
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("拼接图像", "最终遮罩")
    FUNCTION = "advanced_composite"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False

    
    def advanced_composite(self, background_image, subject_image, subject_mask, position_mask, alignment, scale_factor):
        """
        执行遮罩图像拼接的主要函数
        """
        # 转换tensor到numpy数组
        bg_img = self.tensor_to_numpy(background_image)
        subj_img = self.tensor_to_numpy(subject_image)
        subj_mask_np = self.mask_tensor_to_numpy(subject_mask)
        pos_mask_np = self.mask_tensor_to_numpy(position_mask)
        
        # 验证尺寸匹配
        if not self.validate_dimensions(bg_img, pos_mask_np, subj_img, subj_mask_np):
            # 如果尺寸不匹配，进行智能调整
            bg_img, pos_mask_np, subj_img, subj_mask_np = self.smart_resize(
                bg_img, pos_mask_np, subj_img, subj_mask_np
            )
        
        # 步骤1：用主体遮罩抠取主体
        extracted_subject = self.extract_subject(subj_img, subj_mask_np)
        
        # 步骤2：分析拼接位置遮罩，获取目标区域
        target_bbox = self.get_position_bbox(pos_mask_np)
        if target_bbox is None:
            # 如果没有检测到白色区域，返回原背景
            return (self.numpy_to_tensor(bg_img), self.numpy_to_mask_tensor(pos_mask_np))
        
        # 步骤3：将抠取的主体缩放到目标尺寸（适应模式）
        scaled_subject, scaled_mask = self.scale_subject_to_target(
            extracted_subject, subj_mask_np, target_bbox, alignment, scale_factor, bg_img.shape
        )
        
        # 步骤4：执行最终拼接
        result_img = self.blend_images(bg_img, scaled_subject, scaled_mask)
        
        # 转换回tensor格式
        result_tensor = self.numpy_to_tensor(result_img)
        final_mask_tensor = self.numpy_to_mask_tensor(scaled_mask)
        
        return (result_tensor, final_mask_tensor)
    
    def validate_dimensions(self, bg_img, pos_mask, subj_img, subj_mask):
        """验证输入尺寸是否符合要求"""
        bg_h, bg_w = bg_img.shape[:2]
        pos_h, pos_w = pos_mask.shape[:2]
        subj_h, subj_w = subj_img.shape[:2]
        mask_h, mask_w = subj_mask.shape[:2]
        
        bg_match = (bg_h == pos_h and bg_w == pos_w)
        subj_match = (subj_h == mask_h and subj_w == mask_w)
        
        return bg_match and subj_match
    
    def smart_resize(self, bg_img, pos_mask, subj_img, subj_mask):
        """智能调整尺寸以符合要求"""
        # 以背景图像尺寸为基准调整位置遮罩
        bg_h, bg_w = bg_img.shape[:2]
        pos_mask_resized = cv2.resize(pos_mask, (bg_w, bg_h))
        
        # 以主体图像尺寸为基准调整主体遮罩
        subj_h, subj_w = subj_img.shape[:2]
        subj_mask_resized = cv2.resize(subj_mask, (subj_w, subj_h))
        
        return bg_img, pos_mask_resized, subj_img, subj_mask_resized
    
    def extract_subject(self, subject_img, subject_mask):
        """使用遮罩从主体图像中抠取主体"""
        # 确保遮罩有正确的维度
        if len(subject_mask.shape) == 2:
            mask_3d = np.expand_dims(subject_mask, axis=2)
            mask_3d = np.repeat(mask_3d, 3, axis=2)
        else:
            mask_3d = subject_mask
        
        # 抠取主体，保持透明背景
        extracted = subject_img * mask_3d
        
        return extracted
    
    def get_position_bbox(self, position_mask):
        """从位置遮罩中获取白色区域的边界框"""
        # 二值化遮罩
        binary_mask = (position_mask > 0.5).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return {
            'x': x, 'y': y, 'width': w, 'height': h,
            'x2': x + w, 'y2': y + h
        }
    
    def get_subject_bbox(self, mask):
        """获取主体的边界框"""
        binary_mask = (mask > 0.1).astype(np.uint8)
        coords = np.column_stack(np.where(binary_mask > 0))
        
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return {
            'x': x_min, 'y': y_min,
            'width': x_max - x_min + 1,
            'height': y_max - y_min + 1,
            'x2': x_max + 1, 'y2': y_max + 1
        }
    
    def scale_subject_to_target(self, extracted_subject, subject_mask, target_bbox, alignment, scale_factor, bg_shape):
        """将抠取的主体缩放到目标区域（适应模式）"""
        target_w, target_h = target_bbox['width'], target_bbox['height']
        target_x, target_y = target_bbox['x'], target_bbox['y']
        bg_h, bg_w = bg_shape[:2]
        
        # 获取主体的有效区域
        subject_bbox = self.get_subject_bbox(subject_mask)
        if subject_bbox is None:
            # 如果没有有效主体区域，返回和背景相同尺寸的空图像
            empty_img = np.zeros((bg_h, bg_w, 3), dtype=np.float32)
            empty_mask = np.zeros((bg_h, bg_w), dtype=np.float32)
            return empty_img, empty_mask
        
        # 裁剪主体到有效区域
        cropped_subject = extracted_subject[
            subject_bbox['y']:subject_bbox['y2'],
            subject_bbox['x']:subject_bbox['x2']
        ]
        cropped_mask = subject_mask[
            subject_bbox['y']:subject_bbox['y2'],
            subject_bbox['x']:subject_bbox['x2']
        ]
        
        # 计算初始缩放比例（适应目标区域）并应用缩放因子
        subj_h, subj_w = cropped_subject.shape[:2]
        scale_w = target_w / subj_w * scale_factor
        scale_h = target_h / subj_h * scale_factor
        scale = min(scale_w, scale_h)
        
        # 计算缩放后的尺寸
        new_w = int(subj_w * scale)
        new_h = int(subj_h * scale)
        
        # 创建和背景图像相同尺寸的最终图像和遮罩
        final_subject = np.zeros((bg_h, bg_w, 3), dtype=np.float32)
        final_mask = np.zeros((bg_h, bg_w), dtype=np.float32)
        
        # 根据对齐方式计算放置位置
        if alignment == "中":
            placement_x = target_x + (target_w - new_w) // 2
            placement_y = target_y + (target_h - new_h) // 2
        elif alignment == "上":
            placement_x = target_x + (target_w - new_w) // 2
            placement_y = target_y
        elif alignment == "下":
            placement_x = target_x + (target_w - new_w) // 2
            placement_y = target_y + target_h - new_h
        elif alignment == "左":
            placement_x = target_x
            placement_y = target_y + (target_h - new_h) // 2
        elif alignment == "右":
            placement_x = target_x + target_w - new_w
            placement_y = target_y + (target_h - new_h) // 2
        else:  # 默认居中
            placement_x = target_x + (target_w - new_w) // 2
            placement_y = target_y + (target_h - new_h) // 2
        
        # 检查是否会超出边界，如果会，则需要进一步缩小
        if placement_x < 0 or placement_y < 0 or placement_x + new_w > bg_w or placement_y + new_h > bg_h:
            # 调整位置到合法范围
            if placement_x < 0:
                placement_x = 0
            if placement_y < 0:
                placement_y = 0
            
            # 计算可用空间
            max_w = bg_w - placement_x
            max_h = bg_h - placement_y
            
            # 如果当前尺寸超出可用空间，重新计算缩放
            if new_w > max_w or new_h > max_h:
                extra_scale_w = max_w / new_w
                extra_scale_h = max_h / new_h
                extra_scale = min(extra_scale_w, extra_scale_h)
                
                # 应用额外缩放
                new_w = int(new_w * extra_scale)
                new_h = int(new_h * extra_scale)
        
        # 确保位置在有效范围内
        placement_x = max(0, min(placement_x, bg_w - new_w))
        placement_y = max(0, min(placement_y, bg_h - new_h))
        
        # 缩放主体和遮罩到最终尺寸
        scaled_subject = cv2.resize(cropped_subject, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 放置主体（现在保证不会超出边界）
        end_x = placement_x + new_w
        end_y = placement_y + new_h
        
        final_subject[placement_y:end_y, placement_x:end_x] = scaled_subject
        final_mask[placement_y:end_y, placement_x:end_x] = scaled_mask
        
        return final_subject, final_mask
    
    def tensor_to_numpy(self, tensor):
        """将ComfyUI的图像tensor转换为numpy数组"""
        if len(tensor.shape) == 4:  # batch dimension
            tensor = tensor[0]
        
        # 从CHW或HWC格式转换为HWC
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # CHW格式
            tensor = tensor.permute(1, 2, 0)
        
        # 转换为numpy并确保数据类型
        img = tensor.cpu().numpy()
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # 确保值在0-1范围内
        img = np.clip(img, 0, 1)
        
        return img
    
    def mask_tensor_to_numpy(self, mask_tensor):
        """将遮罩tensor转换为numpy数组"""
        if len(mask_tensor.shape) == 3:  # 移除batch dimension
            mask_tensor = mask_tensor[0]
        
        mask = mask_tensor.cpu().numpy()
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        # 确保值在0-1范围内
        mask = np.clip(mask, 0, 1)
        
        return mask
    
    def numpy_to_tensor(self, img):
        """将numpy数组转换为ComfyUI的图像tensor格式"""
        # 确保是HWC格式
        if len(img.shape) == 2:  # 灰度图
            img = np.expand_dims(img, axis=2)
        
        # 转换为tensor
        tensor = torch.from_numpy(img).float()
        
        # 添加batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def numpy_to_mask_tensor(self, mask):
        """将numpy遮罩数组转换为ComfyUI的遮罩tensor格式"""
        # 确保是2D格式（移除多余的通道维度）
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # 转换为tensor
        tensor = torch.from_numpy(mask).float()
        
        # 添加batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def blend_images(self, background, subject, mask):
        """根据遮罩混合图像（正常模式）"""
        # 确保所有图像尺寸一致
        bg_h, bg_w = background.shape[:2]
        
        # 如果subject或mask尺寸不匹配，调整为背景图像尺寸
        if subject.shape[:2] != (bg_h, bg_w):
            subject = cv2.resize(subject, (bg_w, bg_h))
        
        if mask.shape[:2] != (bg_h, bg_w):
            mask = cv2.resize(mask, (bg_w, bg_h))
        
        # 确保遮罩有正确的维度
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        
        # 使用遮罩混合背景和主体图像
        result = background * (1 - mask) + subject * mask
        
        # 确保值在有效范围内
        result = np.clip(result, 0, 1)
        
        return result


# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "AdvancedMaskImageComposite": AdvancedMaskImageComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedMaskImageComposite": "🐳遮罩图像拼接",
}