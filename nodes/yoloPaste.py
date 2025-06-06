import torch
import torchvision.transforms.functional as TF
import numpy as np

class YoloImagePasteNode:
    """
    与YOLO检测节点配套的拼接节点
    将处理后的图像粘贴回原始位置
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"display": "原始图像"}),
                "paste_images": ("IMAGE", {"display": "粘贴图像"}),
                "bboxes": ("BBOXES", {"display": "边界框"}),
                "paste_mode": (["自动匹配", "指定索引", "全部替换"], {
                    "default": "自动匹配",
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
    FUNCTION = "paste_images"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "将处理后的图像粘贴回YOLO检测的原始位置，支持多种混合模式"

    def create_feathered_mask(self, height, width, bbox, feather_amount):
        """创建羽化遮罩"""
        mask = np.zeros((height, width), dtype=np.float32)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 创建基础遮罩
        mask[y1:y2, x1:x2] = 1.0
        
        if feather_amount > 0:
            # 应用高斯模糊实现羽化
            import cv2
            kernel_size = feather_amount * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), feather_amount)
        
        return mask

    def resize_and_paste(self, original_img, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask=None):
        """将图像调整大小并粘贴到指定位置"""
        height, width = original_img.shape[1], original_img.shape[2]
        
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
            
            # 应用遮罩混合
            for c in range(3):  # RGB通道
                result_img[:, :, c] = original_img[:, :, c] * (1 - feather_mask_tensor) + \
                                     result_img[:, :, c] * feather_mask_tensor
        
        # 创建输出遮罩
        output_mask = np.zeros((height, width), dtype=np.float32)
        output_mask[y1:y2, x1:x2] = 1.0
        output_mask_tensor = torch.from_numpy(output_mask).float()
        
        return result_img, output_mask_tensor

    def paste_images(self, original_image, paste_images, bboxes, paste_mode, 
                    target_index, blend_mode, blend_alpha, feather_amount, mask=None):
        """执行图像拼接"""
        # 确保输入是4维张量
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        if len(paste_images.shape) == 3:
            paste_images = paste_images.unsqueeze(0)
        
        batch_size = original_image.shape[0]
        num_paste_images = paste_images.shape[0]
        
        # 处理边界框数据
        if isinstance(bboxes, torch.Tensor):
            bboxes_list = bboxes.tolist()
        else:
            bboxes_list = bboxes
        
        num_bboxes = len(bboxes_list)
        
        print(f"原图批次: {batch_size}, 粘贴图像: {num_paste_images}, 边界框: {num_bboxes}")
        
        try:
            result_images = []
            result_masks = []
            
            for b in range(batch_size):
                current_img = original_image[b]
                current_mask = torch.zeros((current_img.shape[1], current_img.shape[2]), dtype=torch.float32)
                
                if paste_mode == "指定索引":
                    # 指定索引模式：只粘贴指定索引的图像
                    if target_index < num_paste_images and target_index < num_bboxes:
                        paste_img = paste_images[target_index]
                        bbox = bboxes_list[target_index]
                        current_img, paste_mask = self.resize_and_paste(
                            current_img, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                        )
                        current_mask = torch.maximum(current_mask, paste_mask)
                        print(f"使用指定索引 {target_index} 进行粘贴")
                    else:
                        print(f"警告: 指定索引 {target_index} 超出范围")
                
                elif paste_mode == "全部替换":
                    # 全部替换模式：替换所有检测到的区域
                    max_items = min(num_paste_images, num_bboxes)
                    
                    # 如果粘贴图像少于边界框，循环使用
                    for i in range(num_bboxes):
                        paste_idx = i % num_paste_images
                        paste_img = paste_images[paste_idx]
                        bbox = bboxes_list[i]
                        current_img, paste_mask = self.resize_and_paste(
                            current_img, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                        )
                        current_mask = torch.maximum(current_mask, paste_mask)
                    
                    print(f"全部替换模式: 粘贴了 {num_bboxes} 个区域")
                
                else:  # 自动匹配模式
                    # 自动匹配：按顺序粘贴所有可用的图像
                    max_items = min(num_paste_images, num_bboxes)
                    
                    for i in range(max_items):
                        paste_img = paste_images[i]
                        bbox = bboxes_list[i]
                        current_img, paste_mask = self.resize_and_paste(
                            current_img, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                        )
                        current_mask = torch.maximum(current_mask, paste_mask)
                    
                    print(f"自动匹配模式: 粘贴了 {max_items} 个图像")
                
                result_images.append(current_img.unsqueeze(0))
                result_masks.append(current_mask.unsqueeze(0).unsqueeze(-1))
            
            # 合并所有批次
            final_result = torch.cat(result_images, dim=0)
            final_mask = torch.cat(result_masks, dim=0)
            
            return (final_result, final_mask)
            
        except Exception as e:
            print(f"图像拼接过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回原图作为fallback
            empty_mask = torch.zeros((batch_size, original_image.shape[1], original_image.shape[2], 1), dtype=torch.float32)
            return (original_image, empty_mask)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloImagePasteNode": YoloImagePasteNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloImagePasteNode": "🐳YOLO图像拼接"
}