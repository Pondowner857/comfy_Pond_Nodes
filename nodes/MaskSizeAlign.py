import torch
import torch.nn.functional as F

class MaskSizeAlign:
    """
    ComfyUI插件：遮罩尺寸对齐（基准遮罩版）
    将第二个遮罩调整到基准遮罩的尺寸，并按指定方式对齐
    保持遮罩2的内容区域不变，通过添加黑色边缘实现对齐
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基准遮罩": ("MASK",),
                "遮罩2": ("MASK",),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐", 
                           "左上对齐", "右上对齐", "左下对齐", "右下对齐"], 
                          {"default": "居中对齐"}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("基准遮罩", "对齐后遮罩")
    FUNCTION = "align_mask_to_base"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    def get_mask_bounds(self, mask):
        """获取遮罩中非零区域的边界"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # 找到非零像素
        coords = torch.nonzero(mask > 0.01)
        
        if coords.numel() == 0:
            # 如果没有非零像素，返回整个遮罩作为边界
            return 0, 0, mask.shape[1], mask.shape[0]
        
        min_y, min_x = coords.min(dim=0)[0]
        max_y, max_x = coords.max(dim=0)[0]
        
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)
    
    def align_mask_with_position(self, mask2, base_height, base_width, base_bounds, mask2_bounds, alignment):
        """
        根据对齐方式将mask2对齐到基准遮罩的位置
        """
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # 获取基准遮罩和mask2的内容边界
        base_x, base_y, base_w, base_h = base_bounds
        mask2_x, mask2_y, mask2_w, mask2_h = mask2_bounds
        
        # 创建输出画布（基准遮罩的尺寸）
        output = torch.zeros((base_height, base_width), dtype=mask2.dtype, device=mask2.device)
        
        # 获取mask2的原始尺寸
        mask2_height, mask2_width = mask2.shape
        
        # 根据对齐方式计算mask2在输出画布中的位置
        if alignment == "居中对齐":
            # 将mask2的内容中心对齐到基准遮罩内容的中心
            base_center_x = base_x + base_w // 2
            base_center_y = base_y + base_h // 2
            mask2_center_x = mask2_x + mask2_w // 2
            mask2_center_y = mask2_y + mask2_h // 2
            
            # 计算mask2应该放置的位置
            place_x = base_center_x - mask2_center_x
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "左对齐":
            # 左边缘对齐，垂直居中
            place_x = base_x - mask2_x
            base_center_y = base_y + base_h // 2
            mask2_center_y = mask2_y + mask2_h // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "右对齐":
            # 右边缘对齐，垂直居中
            place_x = (base_x + base_w) - (mask2_x + mask2_w)
            base_center_y = base_y + base_h // 2
            mask2_center_y = mask2_y + mask2_h // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "上对齐":
            # 上边缘对齐，水平居中
            base_center_x = base_x + base_w // 2
            mask2_center_x = mask2_x + mask2_w // 2
            place_x = base_center_x - mask2_center_x
            place_y = base_y - mask2_y
            
        elif alignment == "下对齐":
            # 下边缘对齐，水平居中
            base_center_x = base_x + base_w // 2
            mask2_center_x = mask2_x + mask2_w // 2
            place_x = base_center_x - mask2_center_x
            place_y = (base_y + base_h) - (mask2_y + mask2_h)
            
        elif alignment == "左上对齐":
            # 左上角对齐
            place_x = base_x - mask2_x
            place_y = base_y - mask2_y
            
        elif alignment == "右上对齐":
            # 右上角对齐
            place_x = (base_x + base_w) - (mask2_x + mask2_w)
            place_y = base_y - mask2_y
            
        elif alignment == "左下对齐":
            # 左下角对齐
            place_x = base_x - mask2_x
            place_y = (base_y + base_h) - (mask2_y + mask2_h)
            
        elif alignment == "右下对齐":
            # 右下角对齐
            place_x = (base_x + base_w) - (mask2_x + mask2_w)
            place_y = (base_y + base_h) - (mask2_y + mask2_h)
        
        # 计算有效的复制区域
        src_start_x = max(0, -place_x)
        src_start_y = max(0, -place_y)
        src_end_x = min(mask2_width, base_width - place_x)
        src_end_y = min(mask2_height, base_height - place_y)
        
        dst_start_x = max(0, place_x)
        dst_start_y = max(0, place_y)
        dst_end_x = dst_start_x + (src_end_x - src_start_x)
        dst_end_y = dst_start_y + (src_end_y - src_start_y)
        
        # 复制mask2的内容到输出画布
        if src_end_x > src_start_x and src_end_y > src_start_y:
            output[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = \
                mask2[src_start_y:src_end_y, src_start_x:src_end_x]
        
        return output
    
    def align_mask_to_base(self, 基准遮罩, 遮罩2, 对齐方式):
        """
        主要处理函数：将遮罩2对齐到基准遮罩
        """
        # 确保输入是正确的维度
        base_mask = 基准遮罩.clone()
        mask2 = 遮罩2.clone()
        
        if len(base_mask.shape) > 2:
            base_mask = base_mask.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # 获取尺寸信息
        base_height, base_width = base_mask.shape
        mask2_height, mask2_width = mask2.shape
        
        # 获取内容边界
        base_bounds = self.get_mask_bounds(base_mask)
        mask2_bounds = self.get_mask_bounds(mask2)
        
        
        # 执行对齐
        aligned_mask2 = self.align_mask_with_position(
            mask2, base_height, base_width, base_bounds, mask2_bounds, 对齐方式
        )
        
        # 确保输出维度正确
        if len(基准遮罩.shape) == 3:
            aligned_mask2 = aligned_mask2.unsqueeze(0)
        elif len(基准遮罩.shape) == 2:
            if len(aligned_mask2.shape) == 2:
                基准遮罩 = 基准遮罩.unsqueeze(0)
                aligned_mask2 = aligned_mask2.unsqueeze(0)
        
        
        return (基准遮罩, aligned_mask2)

class MaskSizeAlignAdvanced:
    """
    ComfyUI插件：高级遮罩尺寸对齐（基准遮罩版）
    支持多个遮罩同时对齐到基准遮罩
    """
    
    def __init__(self):
        self.basic_aligner = MaskSizeAlign()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基准遮罩": ("MASK",),
                "遮罩2": ("MASK",),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐", 
                           "左上对齐", "右上对齐", "左下对齐", "右下对齐"], 
                          {"default": "居中对齐"}),
            },
            "optional": {
                "遮罩3": ("MASK",),
                "遮罩4": ("MASK",),
                "遮罩5": ("MASK",),
                "X轴偏移": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "Y轴偏移": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("基准遮罩", "对齐遮罩2", "对齐遮罩3", "对齐遮罩4", "合并遮罩")
    FUNCTION = "align_multiple_masks"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    def apply_offset(self, mask, offset_x, offset_y):
        """应用偏移到遮罩"""
        if offset_x == 0 and offset_y == 0:
            return mask
        
        h, w = mask.shape[-2:]
        output = torch.zeros_like(mask)
        
        # 计算源和目标区域
        src_x_start = max(0, -offset_x)
        src_y_start = max(0, -offset_y)
        src_x_end = min(w, w - offset_x)
        src_y_end = min(h, h - offset_y)
        
        dst_x_start = max(0, offset_x)
        dst_y_start = max(0, offset_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        if src_x_end > src_x_start and src_y_end > src_y_start:
            output[..., dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                mask[..., src_y_start:src_y_end, src_x_start:src_x_end]
        
        return output
    
    def align_multiple_masks(self, 基准遮罩, 遮罩2, 对齐方式, 遮罩3=None, 遮罩4=None, 遮罩5=None, X轴偏移=0, Y轴偏移=0):
        """
        对齐多个遮罩到基准遮罩
        """
        # 收集所有需要对齐的遮罩
        masks_to_align = [遮罩2]
        if 遮罩3 is not None:
            masks_to_align.append(遮罩3)
        if 遮罩4 is not None:
            masks_to_align.append(遮罩4)
        if 遮罩5 is not None:
            masks_to_align.append(遮罩5)
        
        # 对齐所有遮罩
        aligned_masks = []
        for i, mask in enumerate(masks_to_align):
            _, aligned = self.basic_aligner.align_mask_to_base(基准遮罩, mask, 对齐方式)
            
            # 应用偏移
            if X轴偏移 != 0 or Y轴偏移 != 0:
                aligned = self.apply_offset(aligned, X轴偏移, Y轴偏移)
            
            aligned_masks.append(aligned)
        
        # 创建合并遮罩（所有对齐后的遮罩的最大值）
        merged = 基准遮罩.clone()
        for aligned in aligned_masks:
            merged = torch.maximum(merged, aligned)
        
        # 准备输出
        output_masks = [基准遮罩] + aligned_masks
        # 确保有5个输出
        while len(output_masks) < 4:
            output_masks.append(torch.zeros_like(基准遮罩))
        output_masks.append(merged)
        
        return tuple(output_masks[:5])

NODE_CLASS_MAPPINGS = {
    "MaskSizeAlign": MaskSizeAlign,
    "MaskSizeAlignAdvanced": MaskSizeAlignAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSizeAlign": "🐳遮罩对齐扩展",
    "MaskSizeAlignAdvanced": "🐳遮罩对齐扩展(V2)"
}