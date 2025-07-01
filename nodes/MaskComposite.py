import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
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
                "scale_mode": (["拉伸", "适应", "填充"], {
                    "default": "适应",
                    "tooltip": "缩放模式：拉伸=直接拉伸到目标尺寸，适应=保持比例适应目标，填充=保持比例填充目标"
                }),
                "alignment": (["居中", "左上", "右上", "左下", "右下"], {
                    "default": "居中",
                    "tooltip": "对齐方式 - 在目标区域内的对齐位置"
                }),
                "edge_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "边缘模糊 - 数值越大边缘越柔和"
                }),  
                "blend_mode": (["正常", "叠加", "滤色", "覆盖"], {
                    "default": "正常",
                    "tooltip": "混合模式：正常=直接覆盖，叠加=相乘效果，滤色=增亮效果，覆盖=对比增强"
                }),
                "feather_edge": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "边缘羽化 - 是否对边缘进行羽化处理，让过渡更自然"
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
            "scale_mode": "缩放模式",
            "alignment": "对齐方式",
            "edge_blur": "边缘模糊",
            "blend_mode": "混合模式",
            "feather_edge": "边缘羽化"
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("拼接图像", "最终遮罩")
    FUNCTION = "advanced_composite"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    # 添加描述信息
    DESCRIPTION = """
遮罩图像拼接节点 - 智能将主体图像拼接到背景图像的指定位置

使用方法：
1. 连接背景图像和主体图像
2. 提供主体遮罩来抠取主体
3. 提供位置遮罩来指定拼接位置（白色区域）
4. 调整缩放模式和对齐方式
5. 使用边缘处理让拼接更自然

提示：
- 位置遮罩的白色区域决定拼接位置
- 边缘模糊和羽化可以让拼接更自然
- 不同的混合模式适合不同场景
"""
    
    def advanced_composite(self, background_image, subject_image, subject_mask, position_mask, 
                          scale_mode, alignment, edge_blur, blend_mode, feather_edge):
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
            print("⚠️ 警告：位置遮罩中未检测到有效的白色区域，返回原背景图像")
            return (self.numpy_to_tensor(bg_img), self.numpy_to_tensor(pos_mask_np))
        
        # 步骤3：将抠取的主体缩放到目标尺寸
        scaled_subject, scaled_mask = self.scale_subject_to_target(
            extracted_subject, subj_mask_np, target_bbox, scale_mode, alignment, bg_img.shape
        )
        
        # 步骤4：应用边缘处理
        if feather_edge or edge_blur > 0:
            scaled_mask = self.apply_edge_processing(scaled_mask, edge_blur, feather_edge)
        
        # 步骤5：执行最终拼接
        result_img = self.blend_images(bg_img, scaled_subject, scaled_mask, blend_mode)
        
        # 转换回tensor格式
        result_tensor = self.numpy_to_tensor(result_img)
        final_mask_tensor = self.numpy_to_tensor(scaled_mask)
        
        return (result_tensor, final_mask_tensor)
    
    def validate_dimensions(self, bg_img, pos_mask, subj_img, subj_mask):
        """验证输入尺寸是否符合要求"""
        bg_h, bg_w = bg_img.shape[:2]
        pos_h, pos_w = pos_mask.shape[:2]
        subj_h, subj_w = subj_img.shape[:2]
        mask_h, mask_w = subj_mask.shape[:2]
        
        bg_match = (bg_h == pos_h and bg_w == pos_w)
        subj_match = (subj_h == mask_h and subj_w == mask_w)
        
        if not bg_match:
            print(f"📐 背景图像尺寸 ({bg_w}x{bg_h}) 与位置遮罩尺寸 ({pos_w}x{pos_h}) 不匹配")
        if not subj_match:
            print(f"📐 主体图像尺寸 ({subj_w}x{subj_h}) 与主体遮罩尺寸 ({mask_w}x{mask_h}) 不匹配")
        
        return bg_match and subj_match
    
    def smart_resize(self, bg_img, pos_mask, subj_img, subj_mask):
        """智能调整尺寸以符合要求"""
        print("🔧 正在自动调整尺寸...")
        
        # 以背景图像尺寸为基准调整位置遮罩
        bg_h, bg_w = bg_img.shape[:2]
        pos_mask_resized = cv2.resize(pos_mask, (bg_w, bg_h))
        
        # 以主体图像尺寸为基准调整主体遮罩
        subj_h, subj_w = subj_img.shape[:2]
        subj_mask_resized = cv2.resize(subj_mask, (subj_w, subj_h))
        
        print("✅ 尺寸调整完成")
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
        
        print(f"📍 检测到拼接位置：x={x}, y={y}, 宽度={w}, 高度={h}")
        
        return {
            'x': x, 'y': y, 'width': w, 'height': h,
            'x2': x + w, 'y2': y + h
        }
    
    def scale_subject_to_target(self, extracted_subject, subject_mask, target_bbox, scale_mode, alignment, bg_shape):
        """将抠取的主体缩放到目标区域"""
        target_w, target_h = target_bbox['width'], target_bbox['height']
        bg_h, bg_w = bg_shape[:2]
        
        # 获取主体的有效区域
        subject_bbox = self.get_subject_bbox(subject_mask)
        if subject_bbox is None:
            # 如果没有有效主体区域，返回和背景相同尺寸的空图像
            print("⚠️ 警告：未检测到有效的主体区域")
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
        
        print(f"🎯 缩放模式：{scale_mode}")
        
        # 根据缩放模式处理
        if scale_mode == "拉伸":
            # 直接拉伸到目标尺寸
            scaled_subject = cv2.resize(cropped_subject, (target_w, target_h))
            scaled_mask = cv2.resize(cropped_mask, (target_w, target_h))
        elif scale_mode == "适应":
            # 保持宽高比，适应目标尺寸
            scaled_subject, scaled_mask = self.scale_with_aspect_ratio(
                cropped_subject, cropped_mask, target_w, target_h, "fit"
            )
        else:  # 填充
            # 保持宽高比，填充目标尺寸
            scaled_subject, scaled_mask = self.scale_with_aspect_ratio(
                cropped_subject, cropped_mask, target_w, target_h, "fill"
            )
        
        # 创建和背景图像相同尺寸的最终图像和遮罩
        final_subject = np.zeros((bg_h, bg_w, 3), dtype=np.float32)
        final_mask = np.zeros((bg_h, bg_w), dtype=np.float32)
        
        # 根据对齐方式计算放置位置
        placement_x, placement_y = self.calculate_placement(
            target_bbox, scaled_subject.shape, alignment
        )
        
        print(f"📐 对齐方式：{alignment}，放置位置：({placement_x}, {placement_y})")
        
        # 添加边界检查和安全裁剪
        scaled_h, scaled_w = scaled_subject.shape[:2]
        
        # 确保放置位置不会超出背景图像边界
        placement_x = max(0, min(placement_x, bg_w - 1))
        placement_y = max(0, min(placement_y, bg_h - 1))
        
        # 计算实际可以放置的区域大小
        available_w = bg_w - placement_x
        available_h = bg_h - placement_y
        
        # 如果缩放后的图像超出可用空间，需要裁剪
        actual_w = min(scaled_w, available_w)
        actual_h = min(scaled_h, available_h)
        
        # 如果需要裁剪，从缩放图像的中心开始裁剪
        if actual_w < scaled_w or actual_h < scaled_h:
            crop_start_x = max(0, (scaled_w - actual_w) // 2)
            crop_start_y = max(0, (scaled_h - actual_h) // 2)
            
            scaled_subject_cropped = scaled_subject[
                crop_start_y:crop_start_y + actual_h,
                crop_start_x:crop_start_x + actual_w
            ]
            scaled_mask_cropped = scaled_mask[
                crop_start_y:crop_start_y + actual_h,
                crop_start_x:crop_start_x + actual_w
            ]
        else:
            scaled_subject_cropped = scaled_subject
            scaled_mask_cropped = scaled_mask
        
        # 安全地放置主体
        end_x = placement_x + actual_w
        end_y = placement_y + actual_h
        
        try:
            final_subject[placement_y:end_y, placement_x:end_x] = scaled_subject_cropped
            final_mask[placement_y:end_y, placement_x:end_x] = scaled_mask_cropped
        except ValueError as e:
            print(f"🚨 拼接警告: {e}")
            print(f"目标区域: [{placement_y}:{end_y}, {placement_x}:{end_x}] = ({end_y-placement_y}, {end_x-placement_x})")
            print(f"源图像尺寸: {scaled_subject_cropped.shape}")
            # 如果仍然出错，使用更保守的方法
            min_h = min(end_y - placement_y, scaled_subject_cropped.shape[0])
            min_w = min(end_x - placement_x, scaled_subject_cropped.shape[1])
            final_subject[placement_y:placement_y+min_h, placement_x:placement_x+min_w] = scaled_subject_cropped[:min_h, :min_w]
            final_mask[placement_y:placement_y+min_h, placement_x:placement_x+min_w] = scaled_mask_cropped[:min_h, :min_w]
        
        return final_subject, final_mask
    
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
    
    def scale_with_aspect_ratio(self, img, mask, target_w, target_h, mode):
        """保持宽高比的缩放"""
        img_h, img_w = img.shape[:2]
        
        # 计算缩放比例
        scale_w = target_w / img_w
        scale_h = target_h / img_h
        
        if mode == "fit":
            scale = min(scale_w, scale_h)
        else:  # fill
            scale = max(scale_w, scale_h)
        
        # 计算新尺寸
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # 缩放
        scaled_img = cv2.resize(img, (new_w, new_h))
        scaled_mask = cv2.resize(mask, (new_w, new_h))
        
        # 如果是fit模式且尺寸小于目标，需要居中放置
        if mode == "fit" and (new_w < target_w or new_h < target_h):
            final_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            final_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
            
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            
            final_img[start_y:start_y+new_h, start_x:start_x+new_w] = scaled_img
            final_mask[start_y:start_y+new_h, start_x:start_x+new_w] = scaled_mask
            
            return final_img, final_mask
        
        # 如果是fill模式且尺寸大于目标，需要裁剪
        elif mode == "fill" and (new_w > target_w or new_h > target_h):
            start_x = max(0, (new_w - target_w) // 2)
            start_y = max(0, (new_h - target_h) // 2)
            
            # 确保裁剪区域不超出图像边界
            end_x = min(start_x + target_w, new_w)
            end_y = min(start_y + target_h, new_h)
            actual_w = end_x - start_x
            actual_h = end_y - start_y
            
            cropped_img = scaled_img[start_y:end_y, start_x:end_x]
            cropped_mask = scaled_mask[start_y:end_y, start_x:end_x]
            
            # 如果裁剪后尺寸不足，用零填充
            if actual_w < target_w or actual_h < target_h:
                final_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
                final_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
                final_img[:actual_h, :actual_w] = cropped_img
                final_mask[:actual_h, :actual_w] = cropped_mask
                return final_img, final_mask
            
            return cropped_img, cropped_mask
        
        return scaled_img, scaled_mask
    
    def calculate_placement(self, target_bbox, subject_shape, alignment):
        """计算主体在目标区域的放置位置"""
        target_x, target_y = target_bbox['x'], target_bbox['y']
        target_w, target_h = target_bbox['width'], target_bbox['height']
        subj_h, subj_w = subject_shape[:2]
        
        if alignment == "居中":
            x = target_x + max(0, (target_w - subj_w) // 2)
            y = target_y + max(0, (target_h - subj_h) // 2)
        elif alignment == "左上":
            x, y = target_x, target_y
        elif alignment == "右上":
            x = target_x + max(0, target_w - subj_w)
            y = target_y
        elif alignment == "左下":
            x = target_x
            y = target_y + max(0, target_h - subj_h)
        else:  # 右下
            x = target_x + max(0, target_w - subj_w)
            y = target_y + max(0, target_h - subj_h)
        
        return max(0, x), max(0, y)
    
    def apply_edge_processing(self, mask, blur_radius, feather_edge):
        """应用边缘处理效果"""
        processed_mask = mask.copy()
        
        if feather_edge:
            # 羽化边缘
            print("🎨 应用边缘羽化...")
            processed_mask = self.feather_mask_edges(processed_mask)
        
        if blur_radius > 0:
            # 边缘模糊
            print(f"🎨 应用边缘模糊 (半径: {blur_radius})...")
            processed_mask = self.apply_edge_blur(processed_mask, blur_radius)
        
        return processed_mask
    
    def feather_mask_edges(self, mask):
        """羽化遮罩边缘"""
        # 使用形态学操作创建羽化效果
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 创建距离变换
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        
        # 归一化距离变换
        if dist_transform.max() > 0:
            feathered = dist_transform / dist_transform.max()
            # 应用平滑曲线
            feathered = np.power(feathered, 0.5)
        else:
            feathered = mask
        
        return feathered.astype(np.float32)
    
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
        """将numpy数组转换为ComfyUI的tensor格式"""
        # 确保是HWC格式
        if len(img.shape) == 2:  # 灰度图
            img = np.expand_dims(img, axis=2)
        
        # 转换为tensor
        tensor = torch.from_numpy(img).float()
        
        # 添加batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def apply_edge_blur(self, mask, blur_radius):
        """对遮罩边缘应用模糊效果"""
        # 将mask转换为0-255范围用于OpenCV处理
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 高斯模糊
        kernel_size = max(3, int(blur_radius * 2) | 1)  # 确保是奇数
        blurred_mask = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), blur_radius)
        
        # 转换回0-1范围
        blurred_mask = blurred_mask.astype(np.float32) / 255.0
        
        return blurred_mask
    
    def blend_images(self, background, subject, mask, blend_mode):
        """根据遮罩和混合模式混合图像"""
        # 确保所有图像尺寸一致
        bg_h, bg_w = background.shape[:2]
        
        # 如果subject或mask尺寸不匹配，调整为背景图像尺寸
        if subject.shape[:2] != (bg_h, bg_w):
            print(f"🔧 调整主体图像尺寸: {subject.shape[:2]} -> ({bg_h}, {bg_w})")
            subject = cv2.resize(subject, (bg_w, bg_h))
        
        if mask.shape[:2] != (bg_h, bg_w):
            print(f"🔧 调整遮罩尺寸: {mask.shape[:2]} -> ({bg_h}, {bg_w})")
            mask = cv2.resize(mask, (bg_w, bg_h))
        
        # 确保遮罩有正确的维度
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        
        print(f"🎨 应用混合模式：{blend_mode}")
        
        # 根据混合模式处理
        if blend_mode == "正常":
            blended = subject
        elif blend_mode == "叠加":
            blended = background * subject
        elif blend_mode == "滤色":
            blended = 1 - (1 - background) * (1 - subject)
        elif blend_mode == "覆盖":
            # 覆盖混合模式
            blended = np.where(background < 0.5,
                             2 * background * subject,
                             1 - 2 * (1 - background) * (1 - subject))
        else:
            blended = subject
        
        # 使用遮罩混合背景和处理后的主体图像
        result = background * (1 - mask) + blended * mask
        
        # 确保值在有效范围内
        result = np.clip(result, 0, 1)
        
        print("✅ 图像拼接完成！")
        
        return result


class MaskBasedImageComposite:
    """
    ComfyUI自定义节点：基于遮罩的图像拼接
    根据遮罩将主体图像拼接到背景图像上，支持边缘模糊控制
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE", {"tooltip": "背景图像 - Background Image"}),  
                "subject_image": ("IMAGE", {"tooltip": "主体图像 - Subject Image"}),     
                "range_mask": ("MASK", {"tooltip": "拼接范围遮罩 - Range Mask"}),         
                "subject_mask": ("MASK", {"tooltip": "主体形状遮罩 - Subject Mask"}),       
                "edge_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "边缘模糊程度 - Edge Blur Amount"
                }),  
                "blend_mode": (["normal", "multiply", "screen", "overlay"], {
                    "default": "normal",
                    "tooltip": "混合模式 - Blend Mode"
                }),  
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("拼接图像",)
    FUNCTION = "composite_images"
    CATEGORY = "🐳Pond/mask"
    
    def composite_images(self, background_image, subject_image, range_mask, subject_mask, edge_blur, blend_mode):
        """
        执行图像拼接的主要函数
        """
        # 转换tensor到numpy数组
        bg_img = self.tensor_to_numpy(background_image)
        subj_img = self.tensor_to_numpy(subject_image)
        range_mask_np = self.mask_tensor_to_numpy(range_mask)
        subject_mask_np = self.mask_tensor_to_numpy(subject_mask)
        
        # 获取所有输入的尺寸，找到最大尺寸
        bg_h, bg_w = bg_img.shape[:2]
        subj_h, subj_w = subj_img.shape[:2]
        range_h, range_w = range_mask_np.shape[:2]
        subject_h, subject_w = subject_mask_np.shape[:2]
        
        # 计算最大画布尺寸
        max_h = max(bg_h, subj_h, range_h, subject_h)
        max_w = max(bg_w, subj_w, range_w, subject_w)
        
        # 将所有图像和遮罩居中对齐到最大画布
        bg_img_aligned = self.center_align_image(bg_img, max_h, max_w)
        subj_img_aligned = self.center_align_image(subj_img, max_h, max_w)
        range_mask_aligned = self.center_align_mask(range_mask_np, max_h, max_w)
        subject_mask_aligned = self.center_align_mask(subject_mask_np, max_h, max_w)
        
        # 处理遮罩组合
        # range_mask定义拼接的总体范围
        # subject_mask定义在该范围内的具体形状
        combined_mask = range_mask_aligned * subject_mask_aligned
        
        # 边缘模糊处理
        if edge_blur > 0:
            combined_mask = self.apply_edge_blur(combined_mask, edge_blur)
        
        # 执行图像混合
        result_img = self.blend_images(bg_img_aligned, subj_img_aligned, combined_mask, blend_mode)
        
        # 转换回tensor格式
        result_tensor = self.numpy_to_tensor(result_img)
        
        return (result_tensor,)
    
    def center_align_image(self, img, target_h, target_w):
        """将图像居中对齐到目标尺寸"""
        current_h, current_w = img.shape[:2]
        
        # 如果已经是目标尺寸，直接返回
        if current_h == target_h and current_w == target_w:
            return img
        
        # 创建目标尺寸的画布，填充黑色
        if len(img.shape) == 3:  # 彩色图像
            canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        else:  # 灰度图像
            canvas = np.zeros((target_h, target_w), dtype=img.dtype)
        
        # 计算居中位置
        start_y = (target_h - current_h) // 2
        start_x = (target_w - current_w) // 2
        end_y = start_y + current_h
        end_x = start_x + current_w
        
        # 将原图像放到画布中心
        canvas[start_y:end_y, start_x:end_x] = img
        
        return canvas
    
    def center_align_mask(self, mask, target_h, target_w):
        """将遮罩居中对齐到目标尺寸"""
        current_h, current_w = mask.shape[:2]
        
        # 如果已经是目标尺寸，直接返回
        if current_h == target_h and current_w == target_w:
            return mask
        
        # 创建目标尺寸的画布，填充0（黑色遮罩）
        canvas = np.zeros((target_h, target_w), dtype=mask.dtype)
        
        # 计算居中位置
        start_y = (target_h - current_h) // 2
        start_x = (target_w - current_w) // 2
        end_y = start_y + current_h
        end_x = start_x + current_w
        
        # 将原遮罩放到画布中心
        canvas[start_y:end_y, start_x:end_x] = mask
        
        return canvas

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
        """将numpy数组转换为ComfyUI的tensor格式"""
        # 确保是HWC格式
        if len(img.shape) == 2:  # 灰度图
            img = np.expand_dims(img, axis=2)
        
        # 转换为tensor
        tensor = torch.from_numpy(img).float()
        
        # 添加batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def apply_edge_blur(self, mask, blur_radius):
        """对遮罩边缘应用模糊效果"""
        # 将mask转换为0-255范围用于OpenCV处理
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 高斯模糊
        kernel_size = max(3, int(blur_radius * 2) | 1)  # 确保是奇数
        blurred_mask = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), blur_radius)
        
        # 转换回0-1范围
        blurred_mask = blurred_mask.astype(np.float32) / 255.0
        
        return blurred_mask
    
    def blend_images(self, background, subject, mask, blend_mode):
        """根据遮罩和混合模式混合图像"""
        # 确保遮罩有正确的维度
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        
        # 根据混合模式处理
        if blend_mode == "normal":
            blended = subject
        elif blend_mode == "multiply":
            blended = background * subject
        elif blend_mode == "screen":
            blended = 1 - (1 - background) * (1 - subject)
        elif blend_mode == "overlay":
            # 覆盖混合模式
            blended = np.where(background < 0.5,
                             2 * background * subject,
                             1 - 2 * (1 - background) * (1 - subject))
        else:
            blended = subject
        
        # 使用遮罩混合背景和处理后的主体图像
        result = background * (1 - mask) + blended * mask
        
        # 确保值在有效范围内
        result = np.clip(result, 0, 1)
        
        return result



# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "AdvancedMaskImageComposite": AdvancedMaskImageComposite,
    "MaskBasedImageComposite": MaskBasedImageComposite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedMaskImageComposite": "🐳遮罩图像拼接",
    "MaskBasedImageComposite": "🎭 遮罩图像拼接 (Mask Image Composite)"
}