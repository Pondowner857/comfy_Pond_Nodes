import torch
import torch.nn.functional as F

class MaskSizeAlign:
    """
    ComfyUI插件：遮罩尺寸对齐
    输入两个遮罩，将小尺寸的遮罩通过扩展黑色区域对齐到大尺寸
    保持白色区域的位置和大小不变，通过左右/上下平均扩展黑色区域
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("对齐遮罩1", "对齐遮罩2")
    FUNCTION = "align_mask_sizes"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    def expand_mask_to_size(self, mask, target_height, target_width):
        """
        扩展遮罩到目标尺寸，保持白色区域不变，通过添加黑色区域实现
        """
        # 确保mask是2D的
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        current_height, current_width = mask.shape[-2:]
        
        # 如果已经是目标尺寸，直接返回
        if current_height == target_height and current_width == target_width:
            return mask
        
        # 计算需要扩展的像素数
        height_diff = target_height - current_height
        width_diff = target_width - current_width
        
        # 计算上下左右的padding
        # 平均分配，如果是奇数则上边/左边多一个像素
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        
        # 使用F.pad进行扩展，padding值为0（黑色）
        # pad的顺序是 (pad_left, pad_right, pad_top, pad_bottom)
        expanded_mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        
        return expanded_mask
    
    def align_mask_sizes(self, mask1, mask2):
        """
        主要处理函数：对齐两个遮罩的尺寸
        """
        # 确保输入是正确的维度
        if len(mask1.shape) > 2:
            mask1 = mask1.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # 获取两个遮罩的尺寸
        h1, w1 = mask1.shape[-2:]
        h2, w2 = mask2.shape[-2:]
        
        # 确定目标尺寸（取最大值）
        target_height = max(h1, h2)
        target_width = max(w1, w2)
        
        print(f"遮罩1尺寸: {h1}x{w1}")
        print(f"遮罩2尺寸: {h2}x{w2}")
        print(f"目标尺寸: {target_height}x{target_width}")
        
        # 扩展两个遮罩到目标尺寸
        aligned_mask1 = self.expand_mask_to_size(mask1, target_height, target_width)
        aligned_mask2 = self.expand_mask_to_size(mask2, target_height, target_width)
        
        # 确保输出维度正确（ComfyUI期望的格式）
        if len(aligned_mask1.shape) == 2:
            aligned_mask1 = aligned_mask1.unsqueeze(0)
        if len(aligned_mask2.shape) == 2:
            aligned_mask2 = aligned_mask2.unsqueeze(0)
        
        print(f"输出遮罩1尺寸: {aligned_mask1.shape}")
        print(f"输出遮罩2尺寸: {aligned_mask2.shape}")
        
        return (aligned_mask1, aligned_mask2)

class MaskSizeAlignAdvanced:
    """
    ComfyUI插件：高级遮罩尺寸对齐
    可以指定对齐方式：居中、左对齐、右对齐、上对齐、下对齐等
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "horizontal_align": (["居中", "左对齐", "右对齐"], {"default": "居中"}),
                "vertical_align": (["居中", "上对齐", "下对齐"], {"default": "居中"}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("对齐遮罩1", "对齐遮罩2")
    FUNCTION = "align_mask_sizes_advanced"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    def expand_mask_with_alignment(self, mask, target_height, target_width, h_align="居中", v_align="居中"):
        """
        根据对齐方式扩展遮罩到目标尺寸
        """
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        current_height, current_width = mask.shape[-2:]
        
        if current_height == target_height and current_width == target_width:
            return mask
        
        height_diff = target_height - current_height
        width_diff = target_width - current_width
        
        # 根据水平对齐方式计算左右padding
        if h_align == "居中":
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
        elif h_align == "左对齐":
            pad_left = 0
            pad_right = width_diff
        else:  # 右对齐
            pad_left = width_diff
            pad_right = 0
        
        # 根据垂直对齐方式计算上下padding
        if v_align == "居中":
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top
        elif v_align == "上对齐":
            pad_top = 0
            pad_bottom = height_diff
        else:  # 下对齐
            pad_top = height_diff
            pad_bottom = 0
        
        # 应用padding
        expanded_mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        
        return expanded_mask
    
    def align_mask_sizes_advanced(self, mask1, mask2, horizontal_align, vertical_align):
        """
        高级对齐功能
        """
        if len(mask1.shape) > 2:
            mask1 = mask1.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        h1, w1 = mask1.shape[-2:]
        h2, w2 = mask2.shape[-2:]
        
        target_height = max(h1, h2)
        target_width = max(w1, w2)
        
        print(f"遮罩1尺寸: {h1}x{w1}")
        print(f"遮罩2尺寸: {h2}x{w2}")
        print(f"目标尺寸: {target_height}x{target_width}")
        print(f"对齐方式: 水平-{horizontal_align}, 垂直-{vertical_align}")
        
        aligned_mask1 = self.expand_mask_with_alignment(
            mask1, target_height, target_width, horizontal_align, vertical_align
        )
        aligned_mask2 = self.expand_mask_with_alignment(
            mask2, target_height, target_width, horizontal_align, vertical_align
        )
        
        if len(aligned_mask1.shape) == 2:
            aligned_mask1 = aligned_mask1.unsqueeze(0)
        if len(aligned_mask2.shape) == 2:
            aligned_mask2 = aligned_mask2.unsqueeze(0)
        
        print(f"输出遮罩1尺寸: {aligned_mask1.shape}")
        print(f"输出遮罩2尺寸: {aligned_mask2.shape}")
        
        return (aligned_mask1, aligned_mask2)
    

NODE_CLASS_MAPPINGS = {
    "MaskSizeAlign": MaskSizeAlign,
    "MaskSizeAlignAdvanced": MaskSizeAlignAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSizeAlign": "🐳遮罩对齐",
    "MaskSizeAlignAdvanced": "🐳遮罩对齐(V2)"
}

