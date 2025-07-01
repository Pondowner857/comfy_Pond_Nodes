import torch
import torch.nn.functional as F

class MaskSizeAlign:
    """
    ComfyUI plugin: Mask Size Alignment (Base Mask Version)
    Adjust second mask to base mask size and align as specified
    Keep mask2 content area unchanged, add black edges for alignment
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK",),
                "mask2": ("MASK",),
                "alignment": (["center", "left", "right", "top", "bottom", 
                           "top-left", "top-right", "bottom-left", "bottom-right"], 
                          {"default": "center"}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("base_mask", "aligned_mask")
    FUNCTION = "align_mask_to_base"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    def get_mask_bounds(self, mask):
        """Get boundaries of non-zero areas in mask"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # Find non-zero pixels
        coords = torch.nonzero(mask > 0.01)
        
        if coords.numel() == 0:
            # If no non-zero pixels, return entire mask as boundary
            return 0, 0, mask.shape[1], mask.shape[0]
        
        min_y, min_x = coords.min(dim=0)[0]
        max_y, max_x = coords.max(dim=0)[0]
        
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)
    
    def align_mask_with_position(self, mask2, base_height, base_width, base_bounds, mask2_bounds, alignment):
        """
        Align mask2 to base mask position based on alignment method
        """
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # Get content boundaries of base mask and mask2
        base_x, base_y, base_w, base_h = base_bounds
        mask2_x, mask2_y, mask2_w, mask2_h = mask2_bounds
        
        # Create output canvas (base mask size)
        output = torch.zeros((base_height, base_width), dtype=mask2.dtype, device=mask2.device)
        
        # Get original dimensions of mask2
        mask2_height, mask2_width = mask2.shape
        
        # Calculate mask2 position in output canvas based on alignment
        if alignment == "center":
            # Align mask2 content center to base mask content center
            base_center_x = base_x + base_w // 2
            base_center_y = base_y + base_h // 2
            mask2_center_x = mask2_x + mask2_w // 2
            mask2_center_y = mask2_y + mask2_h // 2
            
            # Calculate where mask2 should be placed
            place_x = base_center_x - mask2_center_x
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "left":
            # Left edge alignment, vertically centered
            place_x = base_x - mask2_x
            base_center_y = base_y + base_h // 2
            mask2_center_y = mask2_y + mask2_h // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "right":
            # Right edge alignment, vertically centered
            place_x = (base_x + base_w) - (mask2_x + mask2_w)
            base_center_y = base_y + base_h // 2
            mask2_center_y = mask2_y + mask2_h // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "top":
            # Top edge alignment, horizontally centered
            base_center_x = base_x + base_w // 2
            mask2_center_x = mask2_x + mask2_w // 2
            place_x = base_center_x - mask2_center_x
            place_y = base_y - mask2_y
            
        elif alignment == "bottom":
            # Bottom edge alignment, horizontally centered
            base_center_x = base_x + base_w // 2
            mask2_center_x = mask2_x + mask2_w // 2
            place_x = base_center_x - mask2_center_x
            place_y = (base_y + base_h) - (mask2_y + mask2_h)
            
        elif alignment == "top-left":
            # Top-left corner alignment
            place_x = base_x - mask2_x
            place_y = base_y - mask2_y
            
        elif alignment == "top-right":
            # Top-right corner alignment
            place_x = (base_x + base_w) - (mask2_x + mask2_w)
            place_y = base_y - mask2_y
            
        elif alignment == "bottom-left":
            # Bottom-left corner alignment
            place_x = base_x - mask2_x
            place_y = (base_y + base_h) - (mask2_y + mask2_h)
            
        elif alignment == "bottom-right":
            # Bottom-right corner alignment
            place_x = (base_x + base_w) - (mask2_x + mask2_w)
            place_y = (base_y + base_h) - (mask2_y + mask2_h)
        
        # Calculate valid copy region
        src_start_x = max(0, -place_x)
        src_start_y = max(0, -place_y)
        src_end_x = min(mask2_width, base_width - place_x)
        src_end_y = min(mask2_height, base_height - place_y)
        
        dst_start_x = max(0, place_x)
        dst_start_y = max(0, place_y)
        dst_end_x = dst_start_x + (src_end_x - src_start_x)
        dst_end_y = dst_start_y + (src_end_y - src_start_y)
        
        # Copy mask2 content to output canvas
        if src_end_x > src_start_x and src_end_y > src_start_y:
            output[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = \
                mask2[src_start_y:src_end_y, src_start_x:src_end_x]
        
        return output
    
    def align_mask_to_base(self, base_mask, mask2, alignment):
        """
        Main processing function: Align mask2 to base mask
        """
        # Ensure inputs have correct dimensions
        base_mask = base_mask.clone()
        mask2 = mask2.clone()
        
        if len(base_mask.shape) > 2:
            base_mask = base_mask.squeeze()
        if len(mask2.shape) > 2:
            mask2 = mask2.squeeze()
        
        # Get size information
        base_height, base_width = base_mask.shape
        mask2_height, mask2_width = mask2.shape
        
        # Get content boundaries
        base_bounds = self.get_mask_bounds(base_mask)
        mask2_bounds = self.get_mask_bounds(mask2)
        
        print(f"Base mask size: {base_height}x{base_width}")
        print(f"Base mask content bounds: x={base_bounds[0]}, y={base_bounds[1]}, w={base_bounds[2]}, h={base_bounds[3]}")
        print(f"Mask2 size: {mask2_height}x{mask2_width}")
        print(f"Mask2 content bounds: x={mask2_bounds[0]}, y={mask2_bounds[1]}, w={mask2_bounds[2]}, h={mask2_bounds[3]}")
        print(f"Alignment: {alignment}")
        
        # Perform alignment
        aligned_mask2 = self.align_mask_with_position(
            mask2, base_height, base_width, base_bounds, mask2_bounds, alignment
        )
        
        # Ensure output dimensions are correct
        if len(base_mask.shape) == 3:
            aligned_mask2 = aligned_mask2.unsqueeze(0)
        elif len(base_mask.shape) == 2:
            if len(aligned_mask2.shape) == 2:
                base_mask = base_mask.unsqueeze(0)
                aligned_mask2 = aligned_mask2.unsqueeze(0)
        
        print(f"Output base mask size: {base_mask.shape}")
        print(f"Output aligned mask size: {aligned_mask2.shape}")
        
        return (base_mask, aligned_mask2)

class MaskSizeAlignAdvanced:
    """
    ComfyUI plugin: Advanced Mask Size Alignment (Base Mask Version)
    Support multiple masks alignment to base mask simultaneously
    """
    
    def __init__(self):
        self.basic_aligner = MaskSizeAlign()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK",),
                "mask2": ("MASK",),
                "alignment": (["center", "left", "right", "top", "bottom", 
                           "top-left", "top-right", "bottom-left", "bottom-right"], 
                          {"default": "center"}),
            },
            "optional": {
                "mask3": ("MASK",),
                "mask4": ("MASK",),
                "mask5": ("MASK",),
                "x_offset": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("base_mask", "aligned_mask2", "aligned_mask3", "aligned_mask4", "merged_mask")
    FUNCTION = "align_multiple_masks"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    def apply_offset(self, mask, offset_x, offset_y):
        """Apply offset to mask"""
        if offset_x == 0 and offset_y == 0:
            return mask
        
        h, w = mask.shape[-2:]
        output = torch.zeros_like(mask)
        
        # Calculate source and target regions
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
    
    def align_multiple_masks(self, base_mask, mask2, alignment, mask3=None, mask4=None, mask5=None, x_offset=0, y_offset=0):
        """
        Align multiple masks to base mask
        """
        # Collect all masks to be aligned
        masks_to_align = [mask2]
        if mask3 is not None:
            masks_to_align.append(mask3)
        if mask4 is not None:
            masks_to_align.append(mask4)
        if mask5 is not None:
            masks_to_align.append(mask5)
        
        # Align all masks
        aligned_masks = []
        for i, mask in enumerate(masks_to_align):
            _, aligned = self.basic_aligner.align_mask_to_base(base_mask, mask, alignment)
            
            # Apply offset
            if x_offset != 0 or y_offset != 0:
                aligned = self.apply_offset(aligned, x_offset, y_offset)
            
            aligned_masks.append(aligned)
            print(f"Aligned mask{i+2}")
        
        # Create merged mask (maximum of all aligned masks)
        merged = base_mask.clone()
        for aligned in aligned_masks:
            merged = torch.maximum(merged, aligned)
        
        # Prepare output
        output_masks = [base_mask] + aligned_masks
        # Ensure 5 outputs
        while len(output_masks) < 4:
            output_masks.append(torch.zeros_like(base_mask))
        output_masks.append(merged)
        
        return tuple(output_masks[:5])

NODE_CLASS_MAPPINGS = {
    "MaskSizeAlign": MaskSizeAlign,
    "MaskSizeAlignAdvanced": MaskSizeAlignAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSizeAlign": "🐳Mask Align Extend",
    "MaskSizeAlignAdvanced": "🐳Mask Align Extend (V2)"
}