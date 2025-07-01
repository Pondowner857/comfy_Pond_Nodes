import torch
import numpy as np
from typing import Tuple, List, Optional, Union

class MaskMultiAlignMergeNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基准遮罩": ("MASK",),
                "遮罩2": ("MASK",),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐", 
                             "左上对齐", "右上对齐", "左下对齐", "右下对齐"],
                             {"default": "居中对齐"}),
                "合并模式": (["相加模式", "最大值模式", "最小值模式", "乘法模式", "屏幕模式"], 
                              {"default": "相加模式"}),
            },
            "optional": {
                "遮罩3": ("MASK",),
                "遮罩4": ("MASK",),
                "遮罩5": ("MASK",),
                "遮罩6": ("MASK",),
                "遮罩7": ("MASK",),
                "遮罩8": ("MASK",),
                "X轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "Y轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("合并遮罩",)
    FUNCTION = "multi_merge_masks"
    CATEGORY = "🐳Pond/mask"
    
    def __init__(self):
        self.stats = {
            "total_merges": 0,
            "avg_processing_time": 0.0,
            "last_canvas_size": (0, 0)
        }
    
    def get_mask_bounds_optimized(self, mask: torch.Tensor) -> Tuple[int, int, int, int]:
        """优化的遮罩边界检测"""
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # 使用GPU加速的边界检测
        coords = torch.nonzero(mask > 0.01)  # 使用小阈值避免浮点精度问题
        
        if coords.numel() == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        
        min_y, min_x = coords.min(dim=0)[0]
        max_y, max_x = coords.max(dim=0)[0]
        
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)
    
    def translate_alignment(self, alignment_cn: str) -> str:
        """将中文对齐方式转换为英文"""
        mapping = {
            "居中对齐": "center", "左对齐": "left", "右对齐": "right", 
            "上对齐": "top", "下对齐": "bottom", "左上对齐": "top-left",
            "右上对齐": "top-right", "左下对齐": "bottom-left", "右下对齐": "bottom-right"
        }
        return mapping.get(alignment_cn, "center")
    
    def translate_merge_mode(self, mode_cn: str) -> str:
        """将中文合并模式转换为英文"""
        mapping = {
            "相加模式": "add", "最大值模式": "max", "最小值模式": "min",
            "乘法模式": "multiply", "屏幕模式": "screen"
        }
        return mapping.get(mode_cn, "add")
    
    def calculate_alignment_offset_to_base(self, base_bounds: Tuple[int, int, int, int], 
                                         mask_bounds: Tuple[int, int, int, int], 
                                         alignment: str, offset_x: int = 0, offset_y: int = 0) -> Tuple[int, int]:
        """计算相对于基准遮罩的对齐偏移量"""
        base_x, base_y, base_w, base_h = base_bounds
        mask_x, mask_y, mask_w, mask_h = mask_bounds
        
        # 计算基准遮罩和当前遮罩的中心点
        base_center_x = base_x + base_w // 2
        base_center_y = base_y + base_h // 2
        mask_center_x = mask_x + mask_w // 2
        mask_center_y = mask_y + mask_h // 2
        
        if alignment == "center":
            # 居中对齐：将当前遮罩的中心对齐到基准遮罩的中心
            offset_x_calc = base_center_x - mask_center_x
            offset_y_calc = base_center_y - mask_center_y
            
        elif alignment == "left":
            # 左对齐：对齐到基准遮罩的左边缘，垂直居中
            offset_x_calc = base_x - mask_x
            offset_y_calc = base_center_y - mask_center_y
            
        elif alignment == "right":
            # 右对齐：对齐到基准遮罩的右边缘，垂直居中
            offset_x_calc = (base_x + base_w) - (mask_x + mask_w)
            offset_y_calc = base_center_y - mask_center_y
            
        elif alignment == "top":
            # 上对齐：对齐到基准遮罩的上边缘，水平居中
            offset_x_calc = base_center_x - mask_center_x
            offset_y_calc = base_y - mask_y
            
        elif alignment == "bottom":
            # 下对齐：对齐到基准遮罩的下边缘，水平居中
            offset_x_calc = base_center_x - mask_center_x
            offset_y_calc = (base_y + base_h) - (mask_y + mask_h)
            
        elif alignment == "top-left":
            # 左上对齐：对齐到基准遮罩的左上角
            offset_x_calc = base_x - mask_x
            offset_y_calc = base_y - mask_y
            
        elif alignment == "top-right":
            # 右上对齐：对齐到基准遮罩的右上角
            offset_x_calc = (base_x + base_w) - (mask_x + mask_w)
            offset_y_calc = base_y - mask_y
            
        elif alignment == "bottom-left":
            # 左下对齐：对齐到基准遮罩的左下角
            offset_x_calc = base_x - mask_x
            offset_y_calc = (base_y + base_h) - (mask_y + mask_h)
            
        elif alignment == "bottom-right":
            # 右下对齐：对齐到基准遮罩的右下角
            offset_x_calc = (base_x + base_w) - (mask_x + mask_w)
            offset_y_calc = (base_y + base_h) - (mask_y + mask_h)
        
        # 应用自定义偏移
        offset_x_calc += offset_x
        offset_y_calc += offset_y
        
        return offset_x_calc, offset_y_calc
    
    def apply_merge_mode_optimized(self, base_region: torch.Tensor, overlay_mask: torch.Tensor, 
                                 mode: str) -> torch.Tensor:
        """优化的合并模式应用"""
        if mode == "add":
            return torch.clamp(base_region + overlay_mask, 0, 1)
        elif mode == "max":
            return torch.maximum(base_region, overlay_mask)
        elif mode == "min":
            # 改进的min模式：只在两个遮罩都有值的地方应用min
            both_nonzero = (base_region > 0) & (overlay_mask > 0)
            result = torch.maximum(base_region, overlay_mask)
            result[both_nonzero] = torch.minimum(base_region[both_nonzero], overlay_mask[both_nonzero])
            return result
        elif mode == "multiply":
            return base_region * overlay_mask
        elif mode == "screen":
            return 1 - (1 - base_region) * (1 - overlay_mask)
        else:
            return torch.clamp(base_region + overlay_mask, 0, 1)
    
    def _place_mask_optimized(self, canvas: torch.Tensor, mask: torch.Tensor, 
                            offset_x: int, offset_y: int, mode: str):
        """优化的遮罩放置函数"""
        h, w = mask.shape
        canvas_h, canvas_w = canvas.shape
        
        # 计算有效区域
        start_y = max(offset_y, 0)
        start_x = max(offset_x, 0)
        end_y = min(offset_y + h, canvas_h)
        end_x = min(offset_x + w, canvas_w)
        
        if end_y <= start_y or end_x <= start_x:
            return  # 无重叠区域
        
        # 计算源区域
        src_start_y = start_y - offset_y
        src_start_x = start_x - offset_x
        src_end_y = src_start_y + (end_y - start_y)
        src_end_x = src_start_x + (end_x - start_x)
        
        mask_region = mask[src_start_y:src_end_y, src_start_x:src_end_x]
        
        if mode == "replace":
            canvas[start_y:end_y, start_x:end_x] = mask_region
        else:
            canvas[start_y:end_y, start_x:end_x] = self.apply_merge_mode_optimized(
                canvas[start_y:end_y, start_x:end_x], mask_region, mode
            )

    def multi_merge_masks(self, 基准遮罩, 遮罩2, 对齐方式, 合并模式, 
                         遮罩3=None, 遮罩4=None, 遮罩5=None, 遮罩6=None, 遮罩7=None, 遮罩8=None,
                         X轴偏移=0, Y轴偏移=0):
        """多遮罩合并主函数"""
        import time
        start_time = time.time()
        
        # 收集所有非空遮罩
        all_masks = [基准遮罩, 遮罩2]
        optional_masks = [遮罩3, 遮罩4, 遮罩5, 遮罩6, 遮罩7, 遮罩8]
        
        for mask in optional_masks:
            if mask is not None:
                all_masks.append(mask)
        
        print(f"📦 开始处理 {len(all_masks)} 个遮罩的合并操作（以第一个遮罩为基准）")
        
        # 翻译参数
        alignment = self.translate_alignment(对齐方式)
        merge_mode = self.translate_merge_mode(合并模式)
        
        # 输入验证
        for i, mask in enumerate(all_masks):
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"❌ 错误: 遮罩{i+1}必须是torch.Tensor类型")
            
            if len(mask.shape) < 2 or len(mask.shape) > 3:
                raise ValueError(f"❌ 错误: 遮罩{i+1}维度必须是2D[H,W]或3D[B,H,W]")
        
        # 统一设备
        target_device = all_masks[0].device
        for i in range(1, len(all_masks)):
            if all_masks[i].device != target_device:
                print(f"⚠️ 警告: 遮罩{i+1}在不同设备上，正在移动到设备 {target_device}")
                all_masks[i] = all_masks[i].to(target_device)
        
        # 标准化所有遮罩格式
        original_batch = len(all_masks[0].shape) == 3
        for i in range(len(all_masks)):
            all_masks[i] = torch.clamp(all_masks[i], 0, 1)
            if len(all_masks[i].shape) == 2:
                all_masks[i] = all_masks[i].unsqueeze(0)
        
        # 获取基准遮罩的边界信息
        base_mask = all_masks[0][0]
        base_bounds = self.get_mask_bounds_optimized(base_mask)
        
        # 检查基准遮罩有效性
        if base_bounds[2] == 0 or base_bounds[3] == 0:
            print(f"⚠️ 警告: 基准遮罩没有有效像素，使用整体尺寸")
            base_bounds = (0, 0, base_mask.shape[1], base_mask.shape[0])
        
        # 计算所有遮罩相对于基准遮罩的偏移量
        all_offsets = [(0, 0)]  # 基准遮罩不需要偏移
        max_left = 0
        max_right = base_mask.shape[1]
        max_top = 0
        max_bottom = base_mask.shape[0]
        
        for i in range(1, len(all_masks)):
            current_mask = all_masks[i][0]
            current_bounds = self.get_mask_bounds_optimized(current_mask)
            
            if current_bounds[2] == 0 or current_bounds[3] == 0:
                print(f"⚠️ 警告: 遮罩{i+1}没有有效像素，使用整体尺寸")
                current_bounds = (0, 0, current_mask.shape[1], current_mask.shape[0])
            
            # 计算相对于基准遮罩的偏移
            offset_x, offset_y = self.calculate_alignment_offset_to_base(
                base_bounds, current_bounds, alignment, X轴偏移, Y轴偏移
            )
            all_offsets.append((offset_x, offset_y))
            
            # 更新画布边界
            max_left = min(max_left, offset_x)
            max_right = max(max_right, offset_x + current_mask.shape[1])
            max_top = min(max_top, offset_y)
            max_bottom = max(max_bottom, offset_y + current_mask.shape[0])
        
        # 计算最终画布尺寸
        canvas_w = max_right - max_left
        canvas_h = max_bottom - max_top
        
        # 创建画布
        canvas = torch.zeros((canvas_h, canvas_w), dtype=base_mask.dtype, device=target_device)
        
        # 放置所有遮罩
        for i, (mask, (offset_x, offset_y)) in enumerate(zip(all_masks, all_offsets)):
            mask_2d = mask[0]
            # 调整偏移量以适应画布
            adjusted_offset_x = offset_x - max_left
            adjusted_offset_y = offset_y - max_top
            
            if i == 0:
                # 基准遮罩直接放置
                self._place_mask_optimized(canvas, mask_2d, 
                                         adjusted_offset_x, adjusted_offset_y, "replace")
                print(f"✅ 放置基准遮罩，位置: ({adjusted_offset_x}, {adjusted_offset_y})")
            else:
                # 其他遮罩使用指定的合并模式
                self._place_mask_optimized(canvas, mask_2d, 
                                         adjusted_offset_x, adjusted_offset_y, merge_mode)
                print(f"✅ 合并遮罩{i+1}，位置: ({adjusted_offset_x}, {adjusted_offset_y})，使用{合并模式}")
        
        # 调整输出格式
        result_mask = canvas
        if not original_batch:
            if len(result_mask.shape) == 3:
                result_mask = result_mask.squeeze(0)
        else:
            if len(result_mask.shape) == 2:
                result_mask = result_mask.unsqueeze(0)
        
        # 更新统计信息
        processing_time = time.time() - start_time
        self.stats["total_merges"] += 1
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (self.stats["total_merges"] - 1) + processing_time) 
            / self.stats["total_merges"]
        )
        self.stats["last_canvas_size"] = (canvas_w, canvas_h)
        
        print(f"""🎯 多遮罩合并完成统计:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 最终画布尺寸: {canvas_w} × {canvas_h} 像素
🎯 对齐方式: {对齐方式}（相对于基准遮罩）
🔧 合并模式: {合并模式}
⏱️ 处理时间: {processing_time:.3f} 秒
📦 合并遮罩数量: {len(all_masks)} 个
📍 偏移设置: X({X轴偏移}) Y({Y轴偏移})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 历史统计:
总处理次数: {self.stats["total_merges"]} 次
平均处理时间: {self.stats["avg_processing_time"]:.3f} 秒
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
        
        return (result_mask,)


# 简化版节点 - 保持向后兼容
class MaskAlignMergeSimpleNode:
    """简化版遮罩合并节点（基准遮罩版）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基准遮罩": ("MASK",),
                "遮罩2": ("MASK",),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐"], {"default": "居中对齐"}),
                "合并模式": (["相加模式", "最大值模式", "最小值模式"], {"default": "相加模式"}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("合并遮罩",)
    FUNCTION = "simple_merge_masks"
    CATEGORY = "🐳Pond/mask"
    
    def simple_merge_masks(self, 基准遮罩, 遮罩2, 对齐方式, 合并模式):
        """简化的遮罩合并"""
        # 使用多遮罩节点的核心功能
        multi_node = MaskMultiAlignMergeNode()
        merged, = multi_node.multi_merge_masks(基准遮罩, 遮罩2, 对齐方式, 合并模式)
        return (merged,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "MaskMultiAlignMergeNode": MaskMultiAlignMergeNode,
    "MaskAlignMergeSimpleNode": MaskAlignMergeSimpleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMultiAlignMergeNode": "🐳多遮罩合并",
    "MaskAlignMergeSimpleNode": "🐳遮罩合并"
}