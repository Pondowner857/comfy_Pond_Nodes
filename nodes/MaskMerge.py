import torch
import numpy as np
from typing import Tuple, List, Optional, Union

class MaskMultiAlignMergeNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK",),
                "mask2": ("MASK",),
                "alignment": (["center", "left", "right", "top", "bottom", 
                             "top-left", "top-right", "bottom-left", "bottom-right"],
                             {"default": "center"}),
                "merge_mode": (["add", "max", "min", "multiply", "screen"], 
                              {"default": "add"}),
            },
            "optional": {
                "mask3": ("MASK",),
                "mask4": ("MASK",),
                "mask5": ("MASK",),
                "mask6": ("MASK",),
                "mask7": ("MASK",),
                "mask8": ("MASK",),
                "x_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)
    FUNCTION = "multi_merge_masks"
    CATEGORY = "🐳Pond/mask"
    
    def __init__(self):
        self.stats = {
            "total_merges": 0,
            "avg_processing_time": 0.0,
            "last_canvas_size": (0, 0)
        }
    
    def get_mask_bounds_optimized(self, mask: torch.Tensor) -> Tuple[int, int, int, int]:
        """Optimized mask boundary detection"""
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # GPU-accelerated boundary detection
        coords = torch.nonzero(mask > 0.01)  # Use small threshold to avoid floating point precision issues
        
        if coords.numel() == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        
        min_y, min_x = coords.min(dim=0)[0]
        max_y, max_x = coords.max(dim=0)[0]
        
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)
    
    def calculate_alignment_offset_to_base(self, base_bounds: Tuple[int, int, int, int], 
                                         mask_bounds: Tuple[int, int, int, int], 
                                         alignment: str, offset_x: int = 0, offset_y: int = 0) -> Tuple[int, int]:
        """Calculate alignment offset relative to base mask"""
        base_x, base_y, base_w, base_h = base_bounds
        mask_x, mask_y, mask_w, mask_h = mask_bounds
        
        # Calculate center points of base mask and current mask
        base_center_x = base_x + base_w // 2
        base_center_y = base_y + base_h // 2
        mask_center_x = mask_x + mask_w // 2
        mask_center_y = mask_y + mask_h // 2
        
        if alignment == "center":
            # Center alignment: align current mask center to base mask center
            offset_x_calc = base_center_x - mask_center_x
            offset_y_calc = base_center_y - mask_center_y
            
        elif alignment == "left":
            # Left alignment: align to base mask left edge, vertically centered
            offset_x_calc = base_x - mask_x
            offset_y_calc = base_center_y - mask_center_y
            
        elif alignment == "right":
            # Right alignment: align to base mask right edge, vertically centered
            offset_x_calc = (base_x + base_w) - (mask_x + mask_w)
            offset_y_calc = base_center_y - mask_center_y
            
        elif alignment == "top":
            # Top alignment: align to base mask top edge, horizontally centered
            offset_x_calc = base_center_x - mask_center_x
            offset_y_calc = base_y - mask_y
            
        elif alignment == "bottom":
            # Bottom alignment: align to base mask bottom edge, horizontally centered
            offset_x_calc = base_center_x - mask_center_x
            offset_y_calc = (base_y + base_h) - (mask_y + mask_h)
            
        elif alignment == "top-left":
            # Top-left alignment: align to base mask top-left corner
            offset_x_calc = base_x - mask_x
            offset_y_calc = base_y - mask_y
            
        elif alignment == "top-right":
            # Top-right alignment: align to base mask top-right corner
            offset_x_calc = (base_x + base_w) - (mask_x + mask_w)
            offset_y_calc = base_y - mask_y
            
        elif alignment == "bottom-left":
            # Bottom-left alignment: align to base mask bottom-left corner
            offset_x_calc = base_x - mask_x
            offset_y_calc = (base_y + base_h) - (mask_y + mask_h)
            
        elif alignment == "bottom-right":
            # Bottom-right alignment: align to base mask bottom-right corner
            offset_x_calc = (base_x + base_w) - (mask_x + mask_w)
            offset_y_calc = (base_y + base_h) - (mask_y + mask_h)
        
        # Apply custom offset
        offset_x_calc += offset_x
        offset_y_calc += offset_y
        
        return offset_x_calc, offset_y_calc
    
    def apply_merge_mode_optimized(self, base_region: torch.Tensor, overlay_mask: torch.Tensor, 
                                 mode: str) -> torch.Tensor:
        """Optimized merge mode application"""
        if mode == "add":
            return torch.clamp(base_region + overlay_mask, 0, 1)
        elif mode == "max":
            return torch.maximum(base_region, overlay_mask)
        elif mode == "min":
            # Improved min mode: only apply min where both masks have values
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
        """Optimized mask placement function"""
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
            canvas[start_y:end_y, start_x:end_x] = self.apply_merge_mode_optimized(
                canvas[start_y:end_y, start_x:end_x], mask_region, mode
            )

    def multi_merge_masks(self, base_mask, mask2, alignment, merge_mode, 
                         mask3=None, mask4=None, mask5=None, mask6=None, mask7=None, mask8=None,
                         x_offset=0, y_offset=0):
        """Multi-mask merge main function"""
        import time
        start_time = time.time()
        
        # Collect all non-empty masks
        all_masks = [base_mask, mask2]
        optional_masks = [mask3, mask4, mask5, mask6, mask7, mask8]
        
        for mask in optional_masks:
            if mask is not None:
                all_masks.append(mask)
        
        print(f"📦 Starting merge operation for {len(all_masks)} masks (using first mask as base)")
        
        # Input validation
        for i, mask in enumerate(all_masks):
            if not isinstance(mask, torch.Tensor):
                raise ValueError(f"❌ Error: Mask{i+1} must be torch.Tensor type")
            
            if len(mask.shape) < 2 or len(mask.shape) > 3:
                raise ValueError(f"❌ Error: Mask{i+1} dimensions must be 2D[H,W] or 3D[B,H,W]")
        
        # Unify device
        target_device = all_masks[0].device
        for i in range(1, len(all_masks)):
            if all_masks[i].device != target_device:
                print(f"⚠️ Warning: Mask{i+1} on different device, moving to device {target_device}")
                all_masks[i] = all_masks[i].to(target_device)
        
        # Standardize all mask formats
        original_batch = len(all_masks[0].shape) == 3
        for i in range(len(all_masks)):
            all_masks[i] = torch.clamp(all_masks[i], 0, 1)
            if len(all_masks[i].shape) == 2:
                all_masks[i] = all_masks[i].unsqueeze(0)
        
        # Get base mask boundary information
        base_mask = all_masks[0][0]
        base_bounds = self.get_mask_bounds_optimized(base_mask)
        
        # Check base mask validity
        if base_bounds[2] == 0 or base_bounds[3] == 0:
            print(f"⚠️ Warning: Base mask has no valid pixels, using overall dimensions")
            base_bounds = (0, 0, base_mask.shape[1], base_mask.shape[0])
        
        # Calculate offsets for all masks relative to base mask
        all_offsets = [(0, 0)]  # Base mask needs no offset
        max_left = 0
        max_right = base_mask.shape[1]
        max_top = 0
        max_bottom = base_mask.shape[0]
        
        for i in range(1, len(all_masks)):
            current_mask = all_masks[i][0]
            current_bounds = self.get_mask_bounds_optimized(current_mask)
            
            if current_bounds[2] == 0 or current_bounds[3] == 0:
                print(f"⚠️ Warning: Mask{i+1} has no valid pixels, using overall dimensions")
                current_bounds = (0, 0, current_mask.shape[1], current_mask.shape[0])
            
            # Calculate offset relative to base mask
            offset_x, offset_y = self.calculate_alignment_offset_to_base(
                base_bounds, current_bounds, alignment, x_offset, y_offset
            )
            all_offsets.append((offset_x, offset_y))
            
            # Update canvas boundaries
            max_left = min(max_left, offset_x)
            max_right = max(max_right, offset_x + current_mask.shape[1])
            max_top = min(max_top, offset_y)
            max_bottom = max(max_bottom, offset_y + current_mask.shape[0])
        
        # Calculate final canvas size
        canvas_w = max_right - max_left
        canvas_h = max_bottom - max_top
        
        # Create canvas
        canvas = torch.zeros((canvas_h, canvas_w), dtype=base_mask.dtype, device=target_device)
        
        # Place all masks
        for i, (mask, (offset_x, offset_y)) in enumerate(zip(all_masks, all_offsets)):
            mask_2d = mask[0]
            # Adjust offset to fit canvas
            adjusted_offset_x = offset_x - max_left
            adjusted_offset_y = offset_y - max_top
            
            if i == 0:
                # Place base mask directly
                self._place_mask_optimized(canvas, mask_2d, 
                                         adjusted_offset_x, adjusted_offset_y, "replace")
                print(f"✅ Placed base mask at position: ({adjusted_offset_x}, {adjusted_offset_y})")
            else:
                # Other masks use specified merge mode
                self._place_mask_optimized(canvas, mask_2d, 
                                         adjusted_offset_x, adjusted_offset_y, merge_mode)
                print(f"✅ Merged mask{i+1} at position: ({adjusted_offset_x}, {adjusted_offset_y}), using {merge_mode} mode")
        
        # Adjust output format
        result_mask = canvas
        if not original_batch:
            if len(result_mask.shape) == 3:
                result_mask = result_mask.squeeze(0)
        else:
            if len(result_mask.shape) == 2:
                result_mask = result_mask.unsqueeze(0)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["total_merges"] += 1
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (self.stats["total_merges"] - 1) + processing_time) 
            / self.stats["total_merges"]
        )
        self.stats["last_canvas_size"] = (canvas_w, canvas_h)
        
        print(f"""🎯 Multi-mask merge complete statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 Final canvas size: {canvas_w} × {canvas_h} pixels
🎯 Alignment: {alignment} (relative to base mask)
🔧 Merge mode: {merge_mode}
⏱️ Processing time: {processing_time:.3f} seconds
📦 Number of merged masks: {len(all_masks)}
📍 Offset settings: X({x_offset}) Y({y_offset})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Historical statistics:
Total processing count: {self.stats["total_merges"]}
Average processing time: {self.stats["avg_processing_time"]:.3f} seconds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
        
        return (result_mask,)


# Simplified node - maintain backward compatibility
class MaskAlignMergeSimpleNode:
    """Simplified mask merge node (base mask version)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK",),
                "mask2": ("MASK",),
                "alignment": (["center", "left", "right", "top", "bottom"], {"default": "center"}),
                "merge_mode": (["add", "max", "min"], {"default": "add"}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)
    FUNCTION = "simple_merge_masks"
    CATEGORY = "🐳Pond/mask"
    
    def simple_merge_masks(self, base_mask, mask2, alignment, merge_mode):
        """Simplified mask merge"""
        # Use multi-mask node's core functionality
        multi_node = MaskMultiAlignMergeNode()
        merged, = multi_node.multi_merge_masks(base_mask, mask2, alignment, merge_mode)
        return (merged,)


# Node mapping
NODE_CLASS_MAPPINGS = {
    "MaskMultiAlignMergeNode": MaskMultiAlignMergeNode,
    "MaskAlignMergeSimpleNode": MaskAlignMergeSimpleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMultiAlignMergeNode": "🐳Multi Mask Merge",
    "MaskAlignMergeSimpleNode": "🐳Mask Merge"
}