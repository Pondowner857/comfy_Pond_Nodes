import torch
import numpy as np
from typing import Tuple, List, Optional, Union
import time

class MaskAlignBooleanNode:
    """
    遮罩对齐布尔运算节点 - 基于Pond合并插件的对齐技术
    
    Features:
    - 🎯 9种对齐方式（包括四角对齐）
    - 🔧 完整的布尔运算支持
    - 📐 智能画布尺寸计算
    - ⚡ GPU加速边界检测
    - 📊 详细运算统计
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "遮罩A": ("MASK", {"tooltip": "基准遮罩"}),
                "遮罩B": ("MASK", {"tooltip": "要对齐的遮罩"}),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐", 
                             "左上对齐", "右上对齐", "左下对齐", "右下对齐"],
                             {"default": "居中对齐", "tooltip": "以遮罩A的白色区域为基准对齐"}),
                "布尔运算": (["交集", "并集", "差集A-B", "差集B-A", "异或", "非A", "非B"], 
                              {"default": "交集", "tooltip": "布尔运算类型"}),
            },
            "optional": {
                "X轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1, "tooltip": "额外的X轴偏移"}),
                "Y轴偏移": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1, "tooltip": "额外的Y轴偏移"}),
                "阈值": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "白色区域检测阈值"}),
                "输出模式": (["运算结果", "对齐预览", "详细输出"], {"default": "运算结果", "tooltip": "输出内容选择"}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("运算结果", "对齐后遮罩A", "对齐后遮罩B", "运算信息")
    FUNCTION = "align_boolean_operation"
    CATEGORY = "🐳Pond/mask"
    DESCRIPTION = "基于白色区域对齐的遮罩布尔运算"
    
    def __init__(self):
        self.stats = {
            "total_operations": 0,
            "avg_processing_time": 0.0,
            "last_canvas_size": (0, 0)
        }
    
    def get_mask_bounds_optimized(self, mask: torch.Tensor, threshold: float = 0.01) -> Tuple[int, int, int, int]:
        """优化的遮罩边界检测 - 基于Pond插件的实现"""
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # 使用GPU加速的边界检测
        coords = torch.nonzero(mask > threshold)
        
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
    
    def calculate_alignment_offsets(self, mask1_shape, mask2_shape, mask1_bounds, mask2_bounds, 
                                  alignment: str, offset_x: int = 0, offset_y: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """计算对齐偏移量 - 基于Pond插件的实现"""
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
    
    def _place_mask_optimized(self, canvas: torch.Tensor, mask: torch.Tensor, 
                            offset_x: int, offset_y: int, mode: str = "replace"):
        """优化的遮罩放置函数 - 基于Pond插件的实现"""
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
            # 这里可以扩展其他放置模式
            canvas[start_y:end_y, start_x:end_x] = mask_region
    
    def apply_boolean_operation(self, mask_a: torch.Tensor, mask_b: torch.Tensor, 
                               operation: str, threshold: float = 0.5) -> torch.Tensor:
        """应用布尔运算"""
        # 二值化
        binary_a = (mask_a > threshold).float()
        binary_b = (mask_b > threshold).float()
        
        if operation == "交集":
            return binary_a * binary_b
        elif operation == "并集":
            return torch.clamp(binary_a + binary_b, 0, 1)
        elif operation == "差集A-B":
            return binary_a * (1.0 - binary_b)
        elif operation == "差集B-A":
            return binary_b * (1.0 - binary_a)
        elif operation == "异或":
            return (binary_a + binary_b) % 2
        elif operation == "非A":
            return 1.0 - binary_a
        elif operation == "非B":
            return 1.0 - binary_b
        else:
            return binary_a * binary_b
    
    def align_boolean_operation(self, 遮罩A, 遮罩B, 对齐方式, 布尔运算, 
                               X轴偏移=0, Y轴偏移=0, 阈值=0.5, 输出模式="运算结果"):
        """主要的对齐布尔运算函数"""
        start_time = time.time()
        
        print(f"🎯 开始遮罩对齐布尔运算: {布尔运算}")
        
        # 翻译参数
        alignment = self.translate_alignment(对齐方式)
        
        # 输入验证
        if not isinstance(遮罩A, torch.Tensor) or not isinstance(遮罩B, torch.Tensor):
            raise ValueError("❌ 错误: 输入必须是torch.Tensor类型")
        
        # 统一设备
        target_device = 遮罩A.device
        if 遮罩B.device != target_device:
            print(f"⚠️ 警告: 遮罩B在不同设备上，正在移动到设备 {target_device}")
            遮罩B = 遮罩B.to(target_device)
        
        # 标准化格式
        original_batch = len(遮罩A.shape) == 3
        if len(遮罩A.shape) == 2:
            遮罩A = 遮罩A.unsqueeze(0)
        if len(遮罩B.shape) == 2:
            遮罩B = 遮罩B.unsqueeze(0)
        
        # 提取单个遮罩进行处理
        mask_a = 遮罩A[0]
        mask_b = 遮罩B[0]
        
        # 应用阈值并获取边界
        mask_a = torch.clamp(mask_a, 0, 1)
        mask_b = torch.clamp(mask_b, 0, 1)
        
        bounds_a = self.get_mask_bounds_optimized(mask_a, 阈值)
        bounds_b = self.get_mask_bounds_optimized(mask_b, 阈值)
        
        print(f"📐 遮罩A边界: {bounds_a}, 遮罩B边界: {bounds_b}")
        
        # 检查有效性
        if bounds_a[2] == 0 or bounds_a[3] == 0:
            print(f"⚠️ 警告: 遮罩A没有有效像素，使用整体尺寸")
            bounds_a = (0, 0, mask_a.shape[1], mask_a.shape[0])
        
        if bounds_b[2] == 0 or bounds_b[3] == 0:
            print(f"⚠️ 警告: 遮罩B没有有效像素，使用整体尺寸")
            bounds_b = (0, 0, mask_b.shape[1], mask_b.shape[0])
        
        # 计算对齐
        canvas_size, offset_a, offset_b = self.calculate_alignment_offsets(
            mask_a.shape, mask_b.shape, bounds_a, bounds_b, 
            alignment, X轴偏移, Y轴偏移
        )
        
        canvas_w, canvas_h = canvas_size
        print(f"📐 画布尺寸: {canvas_w} × {canvas_h}")
        print(f"📍 偏移 - A: {offset_a}, B: {offset_b}")
        
        # 创建对齐后的遮罩
        aligned_mask_a = torch.zeros((canvas_h, canvas_w), dtype=mask_a.dtype, device=target_device)
        aligned_mask_b = torch.zeros((canvas_h, canvas_w), dtype=mask_b.dtype, device=target_device)
        
        # 放置遮罩
        self._place_mask_optimized(aligned_mask_a, mask_a, offset_a[0], offset_a[1])
        self._place_mask_optimized(aligned_mask_b, mask_b, offset_b[0], offset_b[1])
        
        # 执行布尔运算
        result_mask = self.apply_boolean_operation(aligned_mask_a, aligned_mask_b, 布尔运算, 阈值)
        
        # 调整输出格式
        if not original_batch:
            if len(result_mask.shape) == 3:
                result_mask = result_mask.squeeze(0)
                aligned_mask_a = aligned_mask_a.squeeze(0) if len(aligned_mask_a.shape) == 3 else aligned_mask_a
                aligned_mask_b = aligned_mask_b.squeeze(0) if len(aligned_mask_b.shape) == 3 else aligned_mask_b
        else:
            if len(result_mask.shape) == 2:
                result_mask = result_mask.unsqueeze(0)
                aligned_mask_a = aligned_mask_a.unsqueeze(0)
                aligned_mask_b = aligned_mask_b.unsqueeze(0)
        
        # 统计信息
        processing_time = time.time() - start_time
        self.stats["total_operations"] += 1
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (self.stats["total_operations"] - 1) + processing_time) 
            / self.stats["total_operations"]
        )
        self.stats["last_canvas_size"] = (canvas_w, canvas_h)
        
        # 生成详细信息
        info = f"""🎯 遮罩对齐布尔运算完成统计:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 最终画布尺寸: {canvas_w} × {canvas_h} 像素
🎯 对齐方式: {对齐方式} ({alignment})
🔧 布尔运算: {布尔运算}
⏱️ 处理时间: {processing_time:.3f} 秒
📍 偏移设置: X({X轴偏移}) Y({Y轴偏移})
🎚️ 检测阈值: {阈值}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 历史统计:
总运算次数: {self.stats["total_operations"]} 次
平均处理时间: {self.stats["avg_processing_time"]:.3f} 秒
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        print(info)
        
        # 根据输出模式返回不同内容
        if 输出模式 == "对齐预览":
            return (aligned_mask_a, aligned_mask_a, aligned_mask_b, info)
        elif 输出模式 == "详细输出":
            return (result_mask, aligned_mask_a, aligned_mask_b, info)
        else:  # 运算结果
            return (result_mask, aligned_mask_a, aligned_mask_b, info)


class MaskMultiBooleanNode:
    """
    多遮罩布尔运算节点 - 支持多个遮罩的连续布尔运算
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "基准遮罩": ("MASK", {"tooltip": "作为基准的遮罩"}),
                "遮罩2": ("MASK", {"tooltip": "第二个遮罩"}),
                "运算1": (["交集", "并集", "差集", "异或"], {"default": "交集", "tooltip": "基准遮罩与遮罩2的运算"}),
            },
            "optional": {
                "遮罩3": ("MASK", {"tooltip": "第三个遮罩"}),
                "运算2": (["交集", "并集", "差集", "异或"], {"default": "交集", "tooltip": "前面结果与遮罩3的运算"}),
                "遮罩4": ("MASK", {"tooltip": "第四个遮罩"}),
                "运算3": (["交集", "并集", "差集", "异或"], {"default": "交集", "tooltip": "前面结果与遮罩4的运算"}),
                "对齐方式": (["居中对齐", "左对齐", "右对齐", "上对齐", "下对齐"], {"default": "居中对齐"}),
                "阈值": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("最终结果", "运算序列")
    FUNCTION = "multi_boolean_operation"
    CATEGORY = "🐳Pond/mask"
    DESCRIPTION = "多个遮罩的连续布尔运算"
    
    def multi_boolean_operation(self, 基准遮罩, 遮罩2, 运算1, 遮罩3=None, 运算2="交集", 
                               遮罩4=None, 运算3="交集", 对齐方式="居中对齐", 阈值=0.5):
        """多遮罩连续布尔运算"""
        align_node = MaskAlignBooleanNode()
        
        # 第一步运算
        result, _, _, info1 = align_node.align_boolean_operation(
            基准遮罩, 遮罩2, 对齐方式, 运算1, 阈值=阈值, 输出模式="运算结果"
        )
        
        sequence = f"步骤1: 基准遮罩 {运算1} 遮罩2"
        
        # 第二步运算
        if 遮罩3 is not None:
            result, _, _, info2 = align_node.align_boolean_operation(
                result, 遮罩3, 对齐方式, 运算2, 阈值=阈值, 输出模式="运算结果"
            )
            sequence += f"\n步骤2: 结果1 {运算2} 遮罩3"
        
        # 第三步运算
        if 遮罩4 is not None:
            result, _, _, info3 = align_node.align_boolean_operation(
                result, 遮罩4, 对齐方式, 运算3, 阈值=阈值, 输出模式="运算结果"
            )
            sequence += f"\n步骤3: 结果2 {运算3} 遮罩4"
        
        return (result, sequence)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "MaskAlignBooleanNode": MaskAlignBooleanNode,
    "MaskMultiBooleanNode": MaskMultiBooleanNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskAlignBooleanNode": "🐳遮罩运算",
    "MaskMultiBooleanNode": "🐳多遮罩运算",
}