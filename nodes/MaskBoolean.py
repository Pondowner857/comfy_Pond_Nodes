import torch
import numpy as np
from typing import Tuple, List, Optional, Union
import time

class MaskAlignBooleanNode:
    """
    Mask alignment boolean operation node - Based on Pond merge plugin's alignment technology
    
    Features:
    - 🎯 9 alignment methods (including corner alignments)
    - 🔧 Complete boolean operation support
    - 📐 Smart canvas size calculation
    - ⚡ GPU accelerated boundary detection
    - 📊 Detailed operation statistics
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK", {"tooltip": "Base mask"}),
                "mask_b": ("MASK", {"tooltip": "Mask to align"}),
                "alignment": (["center", "left", "right", "top", "bottom", 
                             "top_left", "top_right", "bottom_left", "bottom_right"],
                             {"default": "center", "tooltip": "Align based on white area of mask A"}),
                "boolean_operation": (["intersection", "union", "difference_a_b", "difference_b_a", "xor", "not_a", "not_b"], 
                              {"default": "intersection", "tooltip": "Boolean operation type"}),
            },
            "optional": {
                "x_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1, "tooltip": "Additional X offset"}),
                "y_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1, "tooltip": "Additional Y offset"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "White area detection threshold"}),
                "output_mode": (["operation_result", "alignment_preview", "detailed_output"], {"default": "operation_result", "tooltip": "Output content selection"}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("result", "aligned_mask_a", "aligned_mask_b", "operation_info")
    FUNCTION = "align_boolean_operation"
    CATEGORY = "🐳Pond/mask"
    DESCRIPTION = "Mask boolean operations based on white area alignment"
    
    def __init__(self):
        self.stats = {
            "total_operations": 0,
            "avg_processing_time": 0.0,
            "last_canvas_size": (0, 0)
        }
    
    def get_mask_bounds_optimized(self, mask: torch.Tensor, threshold: float = 0.01) -> Tuple[int, int, int, int]:
        """Optimized mask boundary detection - Based on Pond plugin implementation"""
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # GPU accelerated boundary detection
        coords = torch.nonzero(mask > threshold)
        
        if coords.numel() == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        
        min_y, min_x = coords.min(dim=0)[0]
        max_y, max_x = coords.max(dim=0)[0]
        
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)
    
    def calculate_alignment_offsets(self, mask1_shape, mask2_shape, mask1_bounds, mask2_bounds, 
                                  alignment: str, offset_x: int = 0, offset_y: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Calculate alignment offsets - Based on Pond plugin implementation"""
        h1, w1 = mask1_shape
        h2, w2 = mask2_shape
        x1, y1, w1_content, h1_content = mask1_bounds
        x2, y2, w2_content, h2_content = mask2_bounds
        
        # Calculate canvas size and base offsets
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
            
        elif alignment == "top_left":
            canvas_w = max(x1 + w1, x2 + w2)
            canvas_h = max(y1 + h1, y2 + h2)
            offset1_x = 0
            offset1_y = 0
            offset2_x = x1 - x2
            offset2_y = y1 - y2
            
        elif alignment == "top_right":
            canvas_w = max(w1, w2)
            canvas_h = max(y1 + h1, y2 + h2)
            offset1_x = canvas_w - w1
            offset1_y = 0
            offset2_x = canvas_w - w2 - (x2 - x1)
            offset2_y = y1 - y2
            
        elif alignment == "bottom_left":
            canvas_w = max(x1 + w1, x2 + w2)
            canvas_h = max(h1, h2)
            offset1_x = 0
            offset1_y = canvas_h - h1
            offset2_x = x1 - x2
            offset2_y = canvas_h - h2 - (y2 - y1)
            
        elif alignment == "bottom_right":
            canvas_w = max(w1, w2)
            canvas_h = max(h1, h2)
            offset1_x = canvas_w - w1
            offset1_y = canvas_h - h1
            offset2_x = canvas_w - w2 - (x2 - x1)
            offset2_y = canvas_h - h2 - (y2 - y1)
        
        # Apply custom offset
        offset2_x += offset_x
        offset2_y += offset_y
        
        return (canvas_w, canvas_h), (offset1_x, offset1_y), (offset2_x, offset2_y)
    
    def _place_mask_optimized(self, canvas: torch.Tensor, mask: torch.Tensor, 
                            offset_x: int, offset_y: int, mode: str = "replace"):
        """Optimized mask placement function - Based on Pond plugin implementation"""
        h, w = mask.shape
        canvas_h, canvas_w = canvas.shape
        
        # Calculate valid region
        start_y = max(offset_y, 0)
        start_x = max(offset_x, 0)
        end_y = min(offset_y + h, canvas_h)
        end_x = min(offset_x + w, canvas_w)
        
        if end_y <= start_y or end_x <= start_x:
            return  # No overlap region
        
        # Calculate source region
        src_start_y = start_y - offset_y
        src_start_x = start_x - offset_x
        src_end_y = src_start_y + (end_y - start_y)
        src_end_x = src_start_x + (end_x - start_x)
        
        mask_region = mask[src_start_y:src_end_y, src_start_x:src_end_x]
        
        if mode == "replace":
            canvas[start_y:end_y, start_x:end_x] = mask_region
        else:
            # Other placement modes can be extended here
            canvas[start_y:end_y, start_x:end_x] = mask_region
    
    def apply_boolean_operation(self, mask_a: torch.Tensor, mask_b: torch.Tensor, 
                               operation: str, threshold: float = 0.5) -> torch.Tensor:
        """Apply boolean operations"""
        # Binarize
        binary_a = (mask_a > threshold).float()
        binary_b = (mask_b > threshold).float()
        
        if operation == "intersection":
            return binary_a * binary_b
        elif operation == "union":
            return torch.clamp(binary_a + binary_b, 0, 1)
        elif operation == "difference_a_b":
            return binary_a * (1.0 - binary_b)
        elif operation == "difference_b_a":
            return binary_b * (1.0 - binary_a)
        elif operation == "xor":
            return (binary_a + binary_b) % 2
        elif operation == "not_a":
            return 1.0 - binary_a
        elif operation == "not_b":
            return 1.0 - binary_b
        else:
            return binary_a * binary_b
    
    def align_boolean_operation(self, mask_a, mask_b, alignment, boolean_operation, 
                               x_offset=0, y_offset=0, threshold=0.5, output_mode="operation_result"):
        """Main alignment boolean operation function"""
        start_time = time.time()
        
        print(f"🎯 Starting mask alignment boolean operation: {boolean_operation}")
        
        # Input validation
        if not isinstance(mask_a, torch.Tensor) or not isinstance(mask_b, torch.Tensor):
            raise ValueError("❌ Error: Inputs must be torch.Tensor type")
        
        # Unify device
        target_device = mask_a.device
        if mask_b.device != target_device:
            print(f"⚠️ Warning: mask_b is on different device, moving to device {target_device}")
            mask_b = mask_b.to(target_device)
        
        # Standardize format
        original_batch = len(mask_a.shape) == 3
        if len(mask_a.shape) == 2:
            mask_a = mask_a.unsqueeze(0)
        if len(mask_b.shape) == 2:
            mask_b = mask_b.unsqueeze(0)
        
        # Extract single mask for processing
        mask_a_single = mask_a[0]
        mask_b_single = mask_b[0]
        
        # Apply threshold and get bounds
        mask_a_single = torch.clamp(mask_a_single, 0, 1)
        mask_b_single = torch.clamp(mask_b_single, 0, 1)
        
        bounds_a = self.get_mask_bounds_optimized(mask_a_single, threshold)
        bounds_b = self.get_mask_bounds_optimized(mask_b_single, threshold)
        
        print(f"📐 Mask A bounds: {bounds_a}, Mask B bounds: {bounds_b}")
        
        # Check validity
        if bounds_a[2] == 0 or bounds_a[3] == 0:
            print(f"⚠️ Warning: Mask A has no valid pixels, using full size")
            bounds_a = (0, 0, mask_a_single.shape[1], mask_a_single.shape[0])
        
        if bounds_b[2] == 0 or bounds_b[3] == 0:
            print(f"⚠️ Warning: Mask B has no valid pixels, using full size")
            bounds_b = (0, 0, mask_b_single.shape[1], mask_b_single.shape[0])
        
        # Calculate alignment
        canvas_size, offset_a, offset_b = self.calculate_alignment_offsets(
            mask_a_single.shape, mask_b_single.shape, bounds_a, bounds_b, 
            alignment, x_offset, y_offset
        )
        
        canvas_w, canvas_h = canvas_size
        print(f"📐 Canvas size: {canvas_w} × {canvas_h}")
        print(f"📍 Offsets - A: {offset_a}, B: {offset_b}")
        
        # Create aligned masks
        aligned_mask_a = torch.zeros((canvas_h, canvas_w), dtype=mask_a_single.dtype, device=target_device)
        aligned_mask_b = torch.zeros((canvas_h, canvas_w), dtype=mask_b_single.dtype, device=target_device)
        
        # Place masks
        self._place_mask_optimized(aligned_mask_a, mask_a_single, offset_a[0], offset_a[1])
        self._place_mask_optimized(aligned_mask_b, mask_b_single, offset_b[0], offset_b[1])
        
        # Execute boolean operation
        result_mask = self.apply_boolean_operation(aligned_mask_a, aligned_mask_b, boolean_operation, threshold)
        
        # Adjust output format
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
        
        # Statistics
        processing_time = time.time() - start_time
        self.stats["total_operations"] += 1
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (self.stats["total_operations"] - 1) + processing_time) 
            / self.stats["total_operations"]
        )
        self.stats["last_canvas_size"] = (canvas_w, canvas_h)
        
        # Generate detailed info
        info = f"""🎯 Mask Alignment Boolean Operation Complete:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 Final canvas size: {canvas_w} × {canvas_h} pixels
🎯 Alignment: {alignment}
🔧 Boolean operation: {boolean_operation}
⏱️ Processing time: {processing_time:.3f} seconds
📍 Offset settings: X({x_offset}) Y({y_offset})
🎚️ Detection threshold: {threshold}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Statistics:
Total operations: {self.stats["total_operations"]}
Average processing time: {self.stats["avg_processing_time"]:.3f} seconds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        print(info)
        
        # Return different content based on output mode
        if output_mode == "alignment_preview":
            return (aligned_mask_a, aligned_mask_a, aligned_mask_b, info)
        elif output_mode == "detailed_output":
            return (result_mask, aligned_mask_a, aligned_mask_b, info)
        else:  # operation_result
            return (result_mask, aligned_mask_a, aligned_mask_b, info)


class MaskMultiBooleanNode:
    """
    Multi-mask boolean operation node - Supports continuous boolean operations on multiple masks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {"tooltip": "Base mask"}),
                "mask2": ("MASK", {"tooltip": "Second mask"}),
                "operation1": (["intersection", "union", "difference", "xor"], {"default": "intersection", "tooltip": "Operation between base mask and mask2"}),
            },
            "optional": {
                "mask3": ("MASK", {"tooltip": "Third mask"}),
                "operation2": (["intersection", "union", "difference", "xor"], {"default": "intersection", "tooltip": "Operation between previous result and mask3"}),
                "mask4": ("MASK", {"tooltip": "Fourth mask"}),
                "operation3": (["intersection", "union", "difference", "xor"], {"default": "intersection", "tooltip": "Operation between previous result and mask4"}),
                "alignment": (["center", "left", "right", "top", "bottom"], {"default": "center"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("final_result", "operation_sequence")
    FUNCTION = "multi_boolean_operation"
    CATEGORY = "🐳Pond/mask"
    DESCRIPTION = "Continuous boolean operations on multiple masks"
    
    def multi_boolean_operation(self, base_mask, mask2, operation1, mask3=None, operation2="intersection", 
                               mask4=None, operation3="intersection", alignment="center", threshold=0.5):
        """Multi-mask continuous boolean operations"""
        align_node = MaskAlignBooleanNode()
        
        # First operation
        result, _, _, info1 = align_node.align_boolean_operation(
            base_mask, mask2, alignment, operation1, threshold=threshold, output_mode="operation_result"
        )
        
        sequence = f"Step 1: base_mask {operation1} mask2"
        
        # Second operation
        if mask3 is not None:
            result, _, _, info2 = align_node.align_boolean_operation(
                result, mask3, alignment, operation2, threshold=threshold, output_mode="operation_result"
            )
            sequence += f"\nStep 2: result1 {operation2} mask3"
        
        # Third operation
        if mask4 is not None:
            result, _, _, info3 = align_node.align_boolean_operation(
                result, mask4, alignment, operation3, threshold=threshold, output_mode="operation_result"
            )
            sequence += f"\nStep 3: result2 {operation3} mask4"
        
        return (result, sequence)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "MaskAlignBooleanNode": MaskAlignBooleanNode,
    "MaskMultiBooleanNode": MaskMultiBooleanNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskAlignBooleanNode": "🐳Mask Boolean Operations",
    "MaskMultiBooleanNode": "🐳Multi-Mask Boolean Operations",
}