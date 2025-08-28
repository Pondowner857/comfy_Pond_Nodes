import torch
import torch.nn.functional as F

class CropPasteBack:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),      # 背景图像
                "cropped": ("IMAGE",),         # 裁剪图像
                "mask": ("MASK",),            # 遮罩
                "crop_data": ("BOX,CROP_DATA,BBOX,RECT,COORDS,LIST,TUPLE,DICT,INT,FLOAT,STRING",),  # 支持多种类型
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "paste_back"
    CATEGORY = "🐳Pond/bbox"
    
    def paste_back(self, background, cropped, mask, crop_data):
        
        # 确保批次维度存在
        if len(background.shape) == 3:
            background = background.unsqueeze(0)
        if len(cropped.shape) == 3:
            cropped = cropped.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        batch_size = background.shape[0]
        result = background.clone()
        
        # 智能解析裁剪框数据
        x1, y1, x2, y2 = self.parse_crop_data(crop_data)
        
        # 计算裁剪区域尺寸
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # 调整裁剪图像和遮罩的大小以匹配裁剪框
        for b in range(batch_size):
            # 获取当前批次的图像和遮罩
            crop_img = cropped[b:b+1]  # 保持4D形状
            crop_mask = mask[b:b+1] if b < mask.shape[0] else mask[0:1]
            
            # 调整裁剪图像大小
            if crop_img.shape[1] != crop_height or crop_img.shape[2] != crop_width:
                # 转换为 (B, C, H, W) 格式进行插值
                crop_img = crop_img.permute(0, 3, 1, 2)
                crop_img = F.interpolate(crop_img, size=(crop_height, crop_width), 
                                        mode='bilinear', align_corners=False)
                crop_img = crop_img.permute(0, 2, 3, 1)
            
            # 调整遮罩大小
            if crop_mask.shape[1] != crop_height or crop_mask.shape[2] != crop_width:
                crop_mask = crop_mask.unsqueeze(1)  # 添加通道维度
                crop_mask = F.interpolate(crop_mask, size=(crop_height, crop_width), 
                                         mode='bilinear', align_corners=False)
                crop_mask = crop_mask.squeeze(1)
            
            # 确保坐标在有效范围内
            bg_height, bg_width = background.shape[1], background.shape[2]
            
            # 计算实际粘贴区域
            paste_x1 = max(0, x1)
            paste_y1 = max(0, y1)
            paste_x2 = min(bg_width, x2)
            paste_y2 = min(bg_height, y2)
            
            # 计算裁剪图像的对应区域
            crop_x1 = max(0, -x1)
            crop_y1 = max(0, -y1)
            crop_x2 = crop_x1 + (paste_x2 - paste_x1)
            crop_y2 = crop_y1 + (paste_y2 - paste_y1)
            
            # 执行贴回操作
            if paste_x2 > paste_x1 and paste_y2 > paste_y1:
                # 获取要粘贴的区域
                paste_region = crop_img[0, crop_y1:crop_y2, crop_x1:crop_x2]
                mask_region = crop_mask[0, crop_y1:crop_y2, crop_x1:crop_x2]
                
                # 扩展遮罩维度以匹配图像通道
                mask_region = mask_region.unsqueeze(-1)
                
                # 使用遮罩混合图像
                result[b, paste_y1:paste_y2, paste_x1:paste_x2] = \
                    paste_region * mask_region + \
                    result[b, paste_y1:paste_y2, paste_x1:paste_x2] * (1 - mask_region)
        
        return (result,)
    
    def parse_crop_data(self, crop_data):

        # 处理张量类型
        if hasattr(crop_data, 'cpu'):
            crop_data = crop_data.cpu().numpy()
        
        # 处理numpy数组
        if hasattr(crop_data, 'flatten'):
            crop_data = crop_data.flatten().tolist()
        
        # 处理字典类型
        if isinstance(crop_data, dict):
            if 'x1' in crop_data and 'y1' in crop_data and 'x2' in crop_data and 'y2' in crop_data:
                x1 = int(crop_data['x1'])
                y1 = int(crop_data['y1'])
                x2 = int(crop_data['x2'])
                y2 = int(crop_data['y2'])
            elif 'x' in crop_data and 'y' in crop_data:
                x1 = int(crop_data['x'])
                y1 = int(crop_data['y'])
                x2 = int(crop_data.get('x2', x1 + crop_data.get('width', 100)))
                y2 = int(crop_data.get('y2', y1 + crop_data.get('height', 100)))
            elif 'left' in crop_data and 'top' in crop_data:
                x1 = int(crop_data['left'])
                y1 = int(crop_data['top'])
                x2 = int(crop_data.get('right', x1 + crop_data.get('width', 100)))
                y2 = int(crop_data.get('bottom', y1 + crop_data.get('height', 100)))
            else:
                # 默认值
                x1, y1, x2, y2 = 0, 0, 100, 100
        
        # 处理列表或元组类型
        elif isinstance(crop_data, (list, tuple)):
            if len(crop_data) >= 4:
                # 尝试转换为整数
                try:
                    x1 = int(crop_data[0])
                    y1 = int(crop_data[1])
                    x2 = int(crop_data[2])
                    y2 = int(crop_data[3])
                except (ValueError, TypeError):
                    x1, y1, x2, y2 = 0, 0, 100, 100
            elif len(crop_data) == 2:
                # 可能是 [[x1, y1], [x2, y2]] 格式
                try:
                    x1 = int(crop_data[0][0])
                    y1 = int(crop_data[0][1])
                    x2 = int(crop_data[1][0])
                    y2 = int(crop_data[1][1])
                except (ValueError, TypeError, IndexError):
                    x1, y1, x2, y2 = 0, 0, 100, 100
            else:
                x1, y1, x2, y2 = 0, 0, 100, 100
        
        # 处理字符串类型
        elif isinstance(crop_data, str):
            # 尝试解析字符串，如 "673,662,2433,2782"
            try:
                values = [int(v.strip()) for v in crop_data.replace(' ', ',').split(',') if v.strip()]
                if len(values) >= 4:
                    x1, y1, x2, y2 = values[:4]
                else:
                    x1, y1, x2, y2 = 0, 0, 100, 100
            except ValueError:
                x1, y1, x2, y2 = 0, 0, 100, 100
        
        # 处理单个数值（可能是嵌套的）
        elif hasattr(crop_data, '__getitem__'):
            try:
                # 尝试递归获取数据
                temp_data = crop_data
                while hasattr(temp_data, '__getitem__') and not isinstance(temp_data, (list, tuple, dict, str)):
                    if hasattr(temp_data, '__len__') and len(temp_data) > 0:
                        temp_data = temp_data[0]
                    else:
                        break
                # 递归调用解析
                if temp_data != crop_data:
                    return self.parse_crop_data(temp_data)
                else:
                    x1, y1, x2, y2 = 0, 0, 100, 100
            except:
                x1, y1, x2, y2 = 0, 0, 100, 100
        
        # 其他未知类型
        else:
            x1, y1, x2, y2 = 0, 0, 100, 100
            print(f"Warning: Unknown crop_data type: {type(crop_data)}")
        
        # 确保 x2 > x1 和 y2 > y1
        if x2 <= x1:
            x2 = x1 + 100
        if y2 <= y1:
            y2 = y1 + 100
        
        return x1, y1, x2, y2

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "CropPasteBack": CropPasteBack,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropPasteBack": "🐳裁剪贴回",
}