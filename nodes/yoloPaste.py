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
                "blend_mode": (["羽化混合", "覆盖", "普通混合"], {
                    "default": "羽化混合",
                    "display": "混合模式"
                }),
                "feather_amount": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "display": "羽化程度"
                }),
                "color_match": ("BOOLEAN", {
                    "default": True,
                    "display": "颜色匹配"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("拼接图像", "合成遮罩")
    INPUT_IS_LIST = {"paste_images": True}  # 标记paste_images接收列表
    FUNCTION = "paste_images"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "将处理后的图像列表粘贴回YOLO检测的原始位置，支持羽化混合避免明显边缘。"

    def create_feather_mask(self, height, width, bbox, feather_size):
        """创建带羽化边缘的遮罩"""
        x1, y1, x2, y2 = bbox
        
        # 创建基础遮罩
        mask = torch.zeros((height, width), dtype=torch.float32)
        
        # 计算内部区域（完全不透明）
        inner_x1 = x1 + feather_size
        inner_y1 = y1 + feather_size
        inner_x2 = x2 - feather_size
        inner_y2 = y2 - feather_size
        
        # 确保内部区域有效
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            mask[inner_y1:inner_y2, inner_x1:inner_x2] = 1.0
        
        # 创建渐变边缘
        if feather_size > 0:
            # 顶部边缘
            for i in range(feather_size):
                alpha = i / feather_size
                y = y1 + i
                if y < height and inner_x2 > inner_x1:
                    mask[y, inner_x1:inner_x2] = alpha
            
            # 底部边缘
            for i in range(feather_size):
                alpha = i / feather_size
                y = y2 - i - 1
                if y >= 0 and inner_x2 > inner_x1:
                    mask[y, inner_x1:inner_x2] = alpha
            
            # 左侧边缘
            for i in range(feather_size):
                alpha = i / feather_size
                x = x1 + i
                if x < width:
                    mask[y1:y2, x] = alpha
            
            # 右侧边缘
            for i in range(feather_size):
                alpha = i / feather_size
                x = x2 - i - 1
                if x >= 0:
                    mask[y1:y2, x] = alpha
            
            # 角落处理 - 使用径向渐变
            corners = [
                (x1, y1, inner_x1, inner_y1),  # 左上
                (inner_x2, y1, x2, inner_y1),   # 右上
                (x1, inner_y2, inner_x1, y2),   # 左下
                (inner_x2, inner_y2, x2, y2)    # 右下
            ]
            
            for cx1, cy1, cx2, cy2 in corners:
                for y in range(max(0, cy1), min(height, cy2)):
                    for x in range(max(0, cx1), min(width, cx2)):
                        # 计算到角落的距离
                        if cx1 == x1:  # 左侧角落
                            dx = x - cx2
                        else:  # 右侧角落
                            dx = cx1 - x
                        
                        if cy1 == y1:  # 上侧角落
                            dy = y - cy2
                        else:  # 下侧角落
                            dy = cy1 - y
                        
                        # 使用欧几里得距离
                        dist = (dx * dx + dy * dy) ** 0.5
                        alpha = min(1.0, dist / feather_size)
                        mask[y, x] = alpha
        
        return mask

    def color_match_region(self, source, target, mask):
        """匹配源图像和目标图像在遮罩区域的颜色"""
        # 计算遮罩区域的平均颜色
        mask_3d = mask.unsqueeze(-1).expand(-1, -1, 3)
        
        # 计算原图在遮罩区域的平均颜色
        if mask.sum() > 0:
            target_mean = (target * mask_3d).sum(dim=[0, 1]) / mask_3d.sum(dim=[0, 1])
            source_mean = (source * mask_3d).sum(dim=[0, 1]) / mask_3d.sum(dim=[0, 1])
            
            # 计算颜色偏移
            color_shift = target_mean - source_mean
            
            # 应用颜色偏移
            adjusted_source = source + color_shift.unsqueeze(0).unsqueeze(0)
            adjusted_source = torch.clamp(adjusted_source, 0, 1)
            
            return adjusted_source
        
        return source

    def gaussian_blur_mask(self, mask, kernel_size):
        """对遮罩应用高斯模糊"""
        if kernel_size <= 1:
            return mask
        
        # 确保kernel_size是奇数
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # 添加批次和通道维度
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        
        # 应用高斯模糊
        blurred = TF.gaussian_blur(mask_4d, kernel_size=kernel_size)
        
        # 移除添加的维度
        return blurred.squeeze(0).squeeze(0)

    def parse_bboxes(self, bboxes):
        """解析各种格式的边界框数据"""
        bboxes_list = []
        
        #print(f"解析边界框，原始类型: {type(bboxes)}")
        
        # 处理嵌套列表的情况
        if isinstance(bboxes, list):
            if len(bboxes) == 1 and isinstance(bboxes[0], list):
                # [[bbox1, bbox2, ...]] 格式
                first_elem = bboxes[0]
                if all(isinstance(item, (list, tuple)) and len(item) == 4 for item in first_elem):
                    bboxes_list = first_elem
                    #print(f"解包边界框列表，得到 {len(bboxes_list)} 个边界框")
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
        
        #print(f"解析后得到 {len(bboxes_list)} 个边界框")
        return bboxes_list

    def paste_single_image(self, base_img, paste_img, bbox, blend_mode, feather_amount, color_match):
        """将单个图像粘贴到指定位置"""
        # 确保输入是3D张量
        if len(base_img.shape) == 4:
            base_img = base_img[0]
        if len(paste_img.shape) == 4:
            paste_img = paste_img[0]
        
        # 获取尺寸
        h, w, _ = base_img.shape
        
        # 解析边界框
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 确保坐标在有效范围内
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            #print(f"无效的边界框: [{x1},{y1},{x2},{y2}]")
            return base_img
        
        target_height = y2 - y1
        target_width = x2 - x1
        
        #print(f"粘贴到区域: [{x1},{y1},{x2},{y2}] (尺寸: {target_width}x{target_height})")
        
        # 调整粘贴图像大小
        paste_tensor = paste_img.permute(2, 0, 1)  # HWC -> CHW
        resized_paste = TF.resize(
            paste_tensor,
            [target_height, target_width],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True
        ).permute(1, 2, 0)  # CHW -> HWC
        
        # 确保设备和数据类型一致
        resized_paste = resized_paste.to(device=base_img.device, dtype=base_img.dtype)
        
        # 颜色匹配
        if color_match and blend_mode != "覆盖":
            # 获取边界区域用于颜色匹配
            border_size = min(20, min(target_width, target_height) // 4)
            if border_size > 0:
                # 创建边界遮罩
                border_mask = torch.ones((target_height, target_width), device=base_img.device)
                if target_height > 2 * border_size and target_width > 2 * border_size:
                    border_mask[border_size:-border_size, border_size:-border_size] = 0
                
                # 在边界区域进行颜色匹配
                original_region = base_img[y1:y2, x1:x2, :]
                resized_paste = self.color_match_region(resized_paste, original_region, border_mask)
        
        # 执行粘贴
        if blend_mode == "覆盖":
            base_img[y1:y2, x1:x2, :] = resized_paste
        elif blend_mode == "普通混合":
            # 简单的alpha混合
            alpha = 0.8
            original_region = base_img[y1:y2, x1:x2, :].clone()
            base_img[y1:y2, x1:x2, :] = original_region * (1 - alpha) + resized_paste * alpha
        else:  # 羽化混合
            # 创建羽化遮罩
            feather_mask = self.create_feather_mask(h, w, [x1, y1, x2, y2], feather_amount)
            
            # 应用额外的高斯模糊使过渡更平滑
            if feather_amount > 0:
                blur_size = max(3, feather_amount // 2)
                feather_mask = self.gaussian_blur_mask(feather_mask, blur_size)
            
            # 提取遮罩区域
            mask_region = feather_mask[y1:y2, x1:x2]
            mask_region_3d = mask_region.unsqueeze(-1).expand(-1, -1, 3)
            
            # 应用羽化混合
            original_region = base_img[y1:y2, x1:x2, :]
            base_img[y1:y2, x1:x2, :] = original_region * (1 - mask_region_3d) + resized_paste * mask_region_3d
        
        return base_img

    def paste_images(self, original_image, paste_images, bboxes, paste_mode, 
                    target_index, blend_mode, feather_amount, color_match):
        """执行图像拼接"""
        
        # 处理参数（可能是列表）
        if isinstance(blend_mode, list):
            blend_mode = blend_mode[0]
        if isinstance(feather_amount, list):
            feather_amount = feather_amount[0]
        if isinstance(paste_mode, list):
            paste_mode = paste_mode[0]
        if isinstance(target_index, list):
            target_index = target_index[0]
        if isinstance(color_match, list):
            color_match = color_match[0]
        
        # 处理原始图像
        if isinstance(original_image, list):
            original_image = original_image[0]
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        
        # 创建输出图像的副本
        output_image = original_image.clone()
        batch_size, height, width, channels = output_image.shape
        
        # 处理粘贴图像列表
        if not isinstance(paste_images, list):
            paste_images = [paste_images]
        
        # 解析边界框
        bboxes_list = self.parse_bboxes(bboxes)
        
        num_paste_images = len(paste_images)
        num_bboxes = len(bboxes_list)
        
        #print(f"\n粘贴参数:")
        #print(f"- 粘贴图像数量: {num_paste_images}")
        #print(f"- 边界框数量: {num_bboxes}")
        #print(f"- 混合模式: {blend_mode}")
        #print(f"- 羽化程度: {feather_amount}")
        #print(f"- 颜色匹配: {color_match}")
        
        # 创建累积遮罩
        cumulative_mask = torch.zeros((height, width), dtype=torch.float32, device=output_image.device)
        
        # 根据模式执行粘贴
        if paste_mode == "指定索引":
            if target_index < num_paste_images and target_index < num_bboxes:
                paste_img = paste_images[target_index]
                bbox = bboxes_list[target_index]
                
                output_image[0] = self.paste_single_image(
                    output_image[0], paste_img, bbox, blend_mode, feather_amount, color_match
                )
                
                # 更新遮罩
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    cumulative_mask[y1:y2, x1:x2] = 1.0
        
        elif paste_mode == "循环使用":
            for i in range(num_bboxes):
                paste_idx = i % num_paste_images
                paste_img = paste_images[paste_idx]
                bbox = bboxes_list[i]
                
                #print(f"\n粘贴第 {i+1}/{num_bboxes} 个区域（使用图像 {paste_idx+1}）")
                
                output_image[0] = self.paste_single_image(
                    output_image[0], paste_img, bbox, blend_mode, feather_amount, color_match
                )
                
                # 更新遮罩
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    cumulative_mask[y1:y2, x1:x2] = 1.0
        
        else:  # 全部粘贴
            max_items = min(num_paste_images, num_bboxes)
            
            for i in range(max_items):
                paste_img = paste_images[i]
                bbox = bboxes_list[i]
                
                #print(f"\n粘贴第 {i+1}/{max_items} 个图像")
                #print(f"边界框: {bbox}")
                
                output_image[0] = self.paste_single_image(
                    output_image[0], paste_img, bbox, blend_mode, feather_amount, color_match
                )
                
                # 更新遮罩
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    cumulative_mask[y1:y2, x1:x2] = 1.0
        
        # 确保输出在0-1范围内
        output_image = torch.clamp(output_image, 0, 1)
        
        # 添加批次维度到遮罩
        output_mask = cumulative_mask.unsqueeze(0)
        
        #print(f"\n粘贴完成")
        #print(f"输出图像尺寸: {output_image.shape}")
        #print(f"输出遮罩尺寸: {output_mask.shape}")
        
        return (output_image, output_mask)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloImagePasteNode": YoloImagePasteNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloImagePasteNode": "🐳YOLO图像拼接"
}