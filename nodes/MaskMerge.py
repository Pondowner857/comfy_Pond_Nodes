import torch
import numpy as np
from typing import Tuple, List, Optional, Union

class MaskMultiAlignMergeNode:
    """
    多遮罩智能对齐合并节点
    
    Features:
    - 📦 支持2-8个遮罩同时输入
    - 🎯 9种对齐方式（包括四角对齐）
    - 🛠️ 自定义偏移微调
    - 📊 保持原始尺寸，无缩放处理
    - 📈 详细的合并统计信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "遮罩1": ("MASK",),
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
    
    def calculate_alignment_offsets(self, mask1_shape, mask2_shape, mask1_bounds, mask2_bounds, 
                                  alignment: str, offset_x: int = 0, offset_y: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """计算对齐偏移量"""
        h1, w1 = mask1_shape
        h2, w2 = mask2_shape
        x1, y1, w1_content, h1_content = mask1_bounds
        x2, y2, w2_content, h2_content = mask2_bounds
        
        # 计算画布尺寸和基础偏移
        if alignment == "center":
            canvas_w = max(w1, w2)
            canvas_h = max(h1, h2)
            offset1_x = (canvas_w - w1) // 2
            offset1_y = (canvas_h - h1) // 2
            offset2_x = (canvas_w - w2) // 2
            offset2_y = (canvas_h - h2) // 2
            
        elif alignment == "left":
            canvas_w = max(x1 + w1, x2 + w2)
            canvas_h = max(h1, h2)
            offset1_x = 0
            offset1_y = (canvas_h - h1) // 2
            offset2_x = x1 - x2
            offset2_y = (canvas_h - h2) // 2
            
        elif alignment == "right":
            canvas_w = max(w1, w2)
            canvas_h = max(h1, h2)
            offset1_x = canvas_w - w1
            offset1_y = (canvas_h - h1) // 2
            offset2_x = canvas_w - w2 - (x2 - x1)
            offset2_y = (canvas_h - h2) // 2
            
        elif alignment == "top":
            canvas_w = max(w1, w2)
            canvas_h = max(y1 + h1, y2 + h2)
            offset1_x = (canvas_w - w1) // 2
            offset1_y = 0
            offset2_x = (canvas_w - w2) // 2
            offset2_y = y1 - y2
            
        elif alignment == "bottom":
            canvas_w = max(w1, w2)
            canvas_h = max(h1, h2)
            offset1_x = (canvas_w - w1) // 2
            offset1_y = canvas_h - h1
            offset2_x = (canvas_w - w2) // 2
            offset2_y = canvas_h - h2 - (y2 - y1)
            
        elif alignment == "top-left":
            canvas_w = max(x1 + w1, x2 + w2)
            canvas_h = max(y1 + h1, y2 + h2)
            offset1_x = 0
            offset1_y = 0
            offset2_x = x1 - x2
            offset2_y = y1 - y2
            
        elif alignment == "top-right":
            canvas_w = max(w1, w2)
            canvas_h = max(y1 + h1, y2 + h2)
            offset1_x = canvas_w - w1
            offset1_y = 0
            offset2_x = canvas_w - w2 - (x2 - x1)
            offset2_y = y1 - y2
            
        elif alignment == "bottom-left":
            canvas_w = max(x1 + w1, x2 + w2)
            canvas_h = max(h1, h2)
            offset1_x = 0
            offset1_y = canvas_h - h1
            offset2_x = x1 - x2
            offset2_y = canvas_h - h2 - (y2 - y1)
            
        elif alignment == "bottom-right":
            canvas_w = max(w1, w2)
            canvas_h = max(h1, h2)
            offset1_x = canvas_w - w1
            offset1_y = canvas_h - h1
            offset2_x = canvas_w - w2 - (x2 - x1)
            offset2_y = canvas_h - h2 - (y2 - y1)
        
        # 应用自定义偏移
        offset2_x += offset_x
        offset2_y += offset_y
        
        return (canvas_w, canvas_h), (offset1_x, offset1_y), (offset2_x, offset2_y)
    
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

    def multi_merge_masks(self, 遮罩1, 遮罩2, 对齐方式, 合并模式, 
                         遮罩3=None, 遮罩4=None, 遮罩5=None, 遮罩6=None, 遮罩7=None, 遮罩8=None,
                         X轴偏移=0, Y轴偏移=0):
        """多遮罩合并主函数"""
        import time
        start_time = time.time()
        
        # 收集所有非空遮罩
        all_masks = [遮罩1, 遮罩2]
        optional_masks = [遮罩3, 遮罩4, 遮罩5, 遮罩6, 遮罩7, 遮罩8]
        
        for mask in optional_masks:
            if mask is not None:
                all_masks.append(mask)
        
        print(f"📦 开始处理 {len(all_masks)} 个遮罩的合并操作")
        
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
        
        # 逐步合并所有遮罩
        result_mask = all_masks[0][0]  # 从第一个遮罩开始
        
        for i in range(1, len(all_masks)):
            current_mask = all_masks[i][0]
            
            # 获取边界信息
            result_bounds = self.get_mask_bounds_optimized(result_mask)
            current_bounds = self.get_mask_bounds_optimized(current_mask)
            
            # 检查有效性
            if result_bounds[2] == 0 or result_bounds[3] == 0:
                print(f"⚠️ 警告: 当前结果遮罩没有有效像素，使用整体尺寸")
                result_bounds = (0, 0, result_mask.shape[1], result_mask.shape[0])
            
            if current_bounds[2] == 0 or current_bounds[3] == 0:
                print(f"⚠️ 警告: 遮罩{i+1}没有有效像素，使用整体尺寸")
                current_bounds = (0, 0, current_mask.shape[1], current_mask.shape[0])
            
            # 计算对齐
            canvas_size, result_offsets, current_offsets = self.calculate_alignment_offsets(
                result_mask.shape, current_mask.shape, result_bounds, current_bounds, 
                alignment, X轴偏移, Y轴偏移
            )
            
            canvas_w, canvas_h = canvas_size
            
            # 创建新画布
            new_canvas = torch.zeros((canvas_h, canvas_w), dtype=result_mask.dtype, device=target_device)
            
            # 放置当前结果
            self._place_mask_optimized(new_canvas, result_mask, 
                                     result_offsets[0], result_offsets[1], "replace")
            
            # 合并新遮罩
            self._place_mask_optimized(new_canvas, current_mask, 
                                     current_offsets[0], current_offsets[1], merge_mode)
            
            result_mask = new_canvas
            print(f"✅ 已合并遮罩{i+1}，当前画布尺寸: {canvas_w}x{canvas_h}")
        
        # 调整输出格式
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
🎯 对齐方式: {对齐方式}
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
    """简化版遮罩合并节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "遮罩1": ("MASK",),
                "遮罩2": ("MASK",),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐"], {"default": "居中对齐"}),
                "合并模式": (["相加模式", "最大值模式", "最小值模式"], {"default": "相加模式"}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("合并遮罩",)
    FUNCTION = "simple_merge_masks"
    CATEGORY = "🐳Pond/mask"
    
    def simple_merge_masks(self, 遮罩1, 遮罩2, 对齐方式, 合并模式):
        """简化的遮罩合并"""
        # 使用多遮罩节点的核心功能
        multi_node = MaskMultiAlignMergeNode()
        merged, = multi_node.multi_merge_masks(遮罩1, 遮罩2, 对齐方式, 合并模式)
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
