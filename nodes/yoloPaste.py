import torch
import torchvision.transforms.functional as TF
import numpy as np

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
                "blend_mode": (["覆盖", "混合", "遮罩混合"], {
                    "default": "覆盖",
                    "display": "混合模式"
                }),
                "blend_alpha": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "display": "混合透明度"
                }),
                "feather_amount": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "羽化程度"
                })
            },
            "optional": {
                "mask": ("MASK", {"display": "遮罩"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("拼接图像", "合成遮罩")
    INPUT_IS_LIST = {"paste_images": True}  # 标记paste_images接收列表
    FUNCTION = "paste_images"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "将处理后的图像列表粘贴回YOLO检测的原始位置，输出单张合成图像。支持接收裁剪节点的列表输出。"

    def create_feathered_mask(self, height, width, bbox, feather_amount):
        """创建羽化遮罩"""
        mask = np.zeros((height, width), dtype=np.float32)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 确保坐标在有效范围内
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        if x2 > x1 and y2 > y1:
            # 创建基础遮罩
            mask[y1:y2, x1:x2] = 1.0
            
            if feather_amount > 0:
                # 应用高斯模糊实现羽化
                try:
                    import cv2
                    kernel_size = feather_amount * 2 + 1
                    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), feather_amount)
                except ImportError:
                    print("警告: 未安装OpenCV，无法应用羽化效果")
        
        return mask

    def resize_and_paste(self, original_img, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask=None):
        """将图像调整大小并粘贴到指定位置"""
        # 确保输入图像维度正确
        if len(original_img.shape) == 4:
            original_img = original_img[0]
        if len(paste_img.shape) == 4:
            paste_img = paste_img[0]
        
        # 获取原图尺寸
        height, width = original_img.shape[:2]
        
        # 解析边界框
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        target_width = x2 - x1
        target_height = y2 - y1
        
        if target_width <= 0 or target_height <= 0:
            print(f"警告: 无效的粘贴区域 [{x1},{y1},{x2},{y2}]")
            return original_img, torch.zeros((height, width), dtype=torch.float32)
        
        # 调整粘贴图像大小
        paste_tensor = paste_img.permute(2, 0, 1)  # HWC -> CHW
        resized_paste = TF.resize(
            paste_tensor, 
            [target_height, target_width],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        ).permute(1, 2, 0)  # CHW -> HWC
        
        # 执行粘贴
        result_img = original_img.clone()
        
        if blend_mode == "覆盖":
            result_img[y1:y2, x1:x2, :] = resized_paste
        elif blend_mode == "混合":
            original_region = result_img[y1:y2, x1:x2, :]
            blended = original_region * (1 - blend_alpha) + resized_paste * blend_alpha
            result_img[y1:y2, x1:x2, :] = blended
        elif blend_mode == "遮罩混合":
            # 创建羽化遮罩
            feather_mask = self.create_feathered_mask(height, width, bbox, feather_amount)
            feather_mask_tensor = torch.from_numpy(feather_mask).float()
            
            # 创建临时图像用于混合
            temp_img = original_img.clone()
            temp_img[y1:y2, x1:x2, :] = resized_paste
            
            # 应用遮罩混合
            for c in range(3):  # RGB通道
                result_img[:, :, c] = original_img[:, :, c] * (1 - feather_mask_tensor) + \
                                     temp_img[:, :, c] * feather_mask_tensor
        
        # 创建输出遮罩
        output_mask = np.zeros((height, width), dtype=np.float32)
        output_mask[y1:y2, x1:x2] = 1.0
        output_mask_tensor = torch.from_numpy(output_mask).float()
        
        return result_img, output_mask_tensor

    def paste_images(self, original_image, paste_images, bboxes, paste_mode, 
                    target_index, blend_mode, blend_alpha, feather_amount, mask=None):
        """执行图像拼接 - 将多个图像粘贴到一张原图上"""
        
        # 处理原始图像输入（可能是列表）
        if isinstance(original_image, list):
            # 如果是列表，使用第一张图像
            original_image = original_image[0]
        
        # 确保原始图像是4维张量
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        
        # 使用第一张原图作为基底
        base_image = original_image[0].clone()
        height, width = base_image.shape[:2]
        
        # 初始化累积遮罩
        cumulative_mask = torch.zeros((height, width), dtype=torch.float32)
        
        # 处理粘贴图像列表
        if not isinstance(paste_images, list):
            paste_images = [paste_images]
        
        # 验证输入
        if not paste_images:
            print("错误：没有提供粘贴图像")
            empty_mask = torch.zeros((1, original_image.shape[1], original_image.shape[2]), dtype=torch.float32)
            return (original_image.unsqueeze(0) if len(original_image.shape) == 3 else original_image[:1], empty_mask)
        
        # 处理边界框数据
        if isinstance(bboxes, torch.Tensor):
            bboxes_list = bboxes.tolist()
        else:
            bboxes_list = bboxes
        
        num_paste_images = len(paste_images)
        num_bboxes = len(bboxes_list)
        
        print(f"粘贴图像数量: {num_paste_images}, 边界框数量: {num_bboxes}")
        
        try:
            if paste_mode == "指定索引":
                # 指定索引模式：只粘贴指定索引的图像
                if target_index < num_paste_images and target_index < num_bboxes:
                    paste_img = paste_images[target_index]
                    # 确保图像维度正确
                    if isinstance(paste_img, torch.Tensor) and len(paste_img.shape) == 4:
                        paste_img = paste_img[0]
                    bbox = bboxes_list[target_index]
                    base_image, paste_mask = self.resize_and_paste(
                        base_image, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                    )
                    cumulative_mask = torch.maximum(cumulative_mask, paste_mask)
                    print(f"使用指定索引 {target_index} 进行粘贴")
                else:
                    print(f"警告: 指定索引 {target_index} 超出范围")
            
            elif paste_mode == "循环使用":
                # 循环使用模式：如果图像少于边界框，循环使用图像
                for i in range(num_bboxes):
                    paste_idx = i % num_paste_images
                    paste_img = paste_images[paste_idx]
                    # 确保图像维度正确
                    if isinstance(paste_img, torch.Tensor) and len(paste_img.shape) == 4:
                        paste_img = paste_img[0]
                    bbox = bboxes_list[i]
                    base_image, paste_mask = self.resize_and_paste(
                        base_image, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                    )
                    cumulative_mask = torch.maximum(cumulative_mask, paste_mask)
                print(f"循环使用模式: 粘贴了 {num_bboxes} 个区域")
            
            else:  # 全部粘贴模式（默认）
                # 全部粘贴：按顺序粘贴所有可用的图像
                max_items = min(num_paste_images, num_bboxes)
                
                for i in range(max_items):
                    paste_img = paste_images[i]
                    # 确保图像维度正确
                    if isinstance(paste_img, torch.Tensor) and len(paste_img.shape) == 4:
                        paste_img = paste_img[0]
                    bbox = bboxes_list[i]
                    base_image, paste_mask = self.resize_and_paste(
                        base_image, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                    )
                    cumulative_mask = torch.maximum(cumulative_mask, paste_mask)
                
                print(f"全部粘贴模式: 粘贴了 {max_items} 个图像")
                
                # 如果边界框多于图像，给出提示
                if num_bboxes > num_paste_images:
                    print(f"提示: 有 {num_bboxes - num_paste_images} 个边界框没有对应的粘贴图像")
                elif num_paste_images > num_bboxes:
                    print(f"提示: 有 {num_paste_images - num_bboxes} 个粘贴图像没有使用")
            
            # 添加批次维度并返回单张图像
            final_image = base_image.unsqueeze(0)
            # MASK格式应该是 (batch, height, width)，不需要通道维度
            final_mask = cumulative_mask.unsqueeze(0)
            
            return (final_image, final_mask)
            
        except Exception as e:
            print(f"图像拼接过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回原图作为fallback
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            # 确保返回的是tensor而不是列表
            if isinstance(original_image, list):
                return (original_image[0].unsqueeze(0) if len(original_image[0].shape) == 3 else original_image[0], empty_mask)
            else:
                return (original_image[:1], empty_mask)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloImagePasteNode": YoloImagePasteNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloImagePasteNode": "🐳YOLO图像拼接"
}