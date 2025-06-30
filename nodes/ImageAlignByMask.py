import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

class ImageAlignByMask:
    """
    ComfyUI plugin: Image positioning based on mask alignment
    Synchronously adjust corresponding image position and size according to mask alignment
    Fill expanded areas with white, black or transparent
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK",),
                "mask2": ("MASK",),
                "image2": ("IMAGE",),
                "alignment": (["center", "left", "right", "top", "bottom", 
                           "top_left", "top_right", "bottom_left", "bottom_right"], 
                          {"default": "center"}),
            },
            "optional": {
                "x_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "fill_color": (["white", "black", "transparent"], {"default": "white"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("aligned_image", "base_mask", "aligned_mask")
    FUNCTION = "align_image_by_mask"
    CATEGORY = "🐳Pond/image"
    OUTPUT_NODE = False
    
    def get_mask_bounds(self, mask: torch.Tensor) -> Tuple[int, int, int, int]:
        """Get bounds of non-zero area in mask"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        coords = torch.nonzero(mask > 0.01)
        
        if coords.numel() == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        
        min_y, min_x = coords.min(dim=0)[0]
        max_y, max_x = coords.max(dim=0)[0]
        
        return int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)
    
    def calculate_alignment_transform(self, base_bounds: Tuple[int, int, int, int], 
                                    mask2_bounds: Tuple[int, int, int, int], 
                                    base_size: Tuple[int, int],
                                    mask2_size: Tuple[int, int],
                                    alignment: str, 
                                    offset_x: int = 0, 
                                    offset_y: int = 0) -> Tuple[int, int]:
        """Calculate transformation parameters needed for alignment"""
        base_h, base_w = base_size
        mask2_h, mask2_w = mask2_size
        base_x, base_y, base_w_content, base_h_content = base_bounds
        mask2_x, mask2_y, mask2_w_content, mask2_h_content = mask2_bounds
        
        # Calculate offset based on alignment method
        if alignment == "center":
            base_center_x = base_x + base_w_content // 2
            base_center_y = base_y + base_h_content // 2
            mask2_center_x = mask2_x + mask2_w_content // 2
            mask2_center_y = mask2_y + mask2_h_content // 2
            
            place_x = base_center_x - mask2_center_x
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "left":
            place_x = base_x - mask2_x
            base_center_y = base_y + base_h_content // 2
            mask2_center_y = mask2_y + mask2_h_content // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "right":
            place_x = (base_x + base_w_content) - (mask2_x + mask2_w_content)
            base_center_y = base_y + base_h_content // 2
            mask2_center_y = mask2_y + mask2_h_content // 2
            place_y = base_center_y - mask2_center_y
            
        elif alignment == "top":
            base_center_x = base_x + base_w_content // 2
            mask2_center_x = mask2_x + mask2_w_content // 2
            place_x = base_center_x - mask2_center_x
            place_y = base_y - mask2_y
            
        elif alignment == "bottom":
            base_center_x = base_x + base_w_content // 2
            mask2_center_x = mask2_x + mask2_w_content // 2
            place_x = base_center_x - mask2_center_x
            place_y = (base_y + base_h_content) - (mask2_y + mask2_h_content)
            
        elif alignment == "top_left":
            place_x = base_x - mask2_x
            place_y = base_y - mask2_y
            
        elif alignment == "top_right":
            place_x = (base_x + base_w_content) - (mask2_x + mask2_w_content)
            place_y = base_y - mask2_y
            
        elif alignment == "bottom_left":
            place_x = base_x - mask2_x
            place_y = (base_y + base_h_content) - (mask2_y + mask2_h_content)
            
        elif alignment == "bottom_right":
            place_x = (base_x + base_w_content) - (mask2_x + mask2_w_content)
            place_y = (base_y + base_h_content) - (mask2_y + mask2_h_content)
        
        # Apply user offset
        place_x += offset_x
        place_y += offset_y
        
        return place_x, place_y
    
    def apply_transform_to_image(self, image: torch.Tensor, 
                               place_x: int, place_y: int,
                               target_height: int, target_width: int,
                               fill_color: str) -> torch.Tensor:
        """Apply transformation to image"""
        batch, h, w, c = image.shape
        
        # Set fill value
        if fill_color == "white":
            fill_value = 1.0
        elif fill_color == "black":
            fill_value = 0.0
        else:  # transparent
            fill_value = 0.0
            # If transparent, ensure alpha channel exists
            if c == 3:
                alpha = torch.ones((batch, h, w, 1), dtype=image.dtype, device=image.device)
                image = torch.cat([image, alpha], dim=3)
                c = 4
        
        # Create output image
        output = torch.full((batch, target_height, target_width, c), 
                          fill_value, dtype=image.dtype, device=image.device)
        
        # If transparent mode, set alpha channel
        if fill_color == "transparent" and c == 4:
            output[:, :, :, 3] = 0.0  # Background transparent
        
        # Calculate copy region
        src_x_start = max(0, -place_x)
        src_y_start = max(0, -place_y)
        src_x_end = min(w, target_width - place_x)
        src_y_end = min(h, target_height - place_y)
        
        dst_x_start = max(0, place_x)
        dst_y_start = max(0, place_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Copy image content
        if src_x_end > src_x_start and src_y_end > src_y_start:
            output[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                image[:, src_y_start:src_y_end, src_x_start:src_x_end]
            
            # If transparent mode, set alpha of copied region to 1
            if fill_color == "transparent" and c == 4:
                output[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, 3] = 1.0
        
        return output
    
    def apply_transform_to_mask(self, mask: torch.Tensor,
                              place_x: int, place_y: int,
                              target_height: int, target_width: int) -> torch.Tensor:
        """Apply transformation to mask"""
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # Create output mask
        output = torch.zeros((target_height, target_width), dtype=mask.dtype, device=mask.device)
        
        h, w = mask.shape
        
        # Calculate copy region
        src_x_start = max(0, -place_x)
        src_y_start = max(0, -place_y)
        src_x_end = min(w, target_width - place_x)
        src_y_end = min(h, target_height - place_y)
        
        dst_x_start = max(0, place_x)
        dst_y_start = max(0, place_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Copy mask content
        if src_x_end > src_x_start and src_y_end > src_y_start:
            output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                mask[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return output
    
    def align_image_by_mask(self, base_mask, mask2, image2, alignment, 
                           x_offset=0, y_offset=0, fill_color="white"):
        """
        Main function: Adjust image based on mask alignment
        """
        # Get base mask dimensions
        if len(base_mask.shape) > 2:
            base_h, base_w = base_mask.shape[1:3]
        else:
            base_h, base_w = base_mask.shape
        
        # Get mask2 dimensions
        if len(mask2.shape) > 2:
            mask2_h, mask2_w = mask2.shape[1:3]
        else:
            mask2_h, mask2_w = mask2.shape
        
        # Get content bounds
        base_bounds = self.get_mask_bounds(base_mask)
        mask2_bounds = self.get_mask_bounds(mask2)
        
        print(f"Base mask size: {base_h}x{base_w}")
        print(f"Base mask content bounds: x={base_bounds[0]}, y={base_bounds[1]}, w={base_bounds[2]}, h={base_bounds[3]}")
        print(f"Mask2 size: {mask2_h}x{mask2_w}")
        print(f"Mask2 content bounds: x={mask2_bounds[0]}, y={mask2_bounds[1]}, w={mask2_bounds[2]}, h={mask2_bounds[3]}")
        print(f"Alignment: {alignment}")
        
        # Calculate alignment transformation
        place_x, place_y = self.calculate_alignment_transform(
            base_bounds, mask2_bounds,
            (base_h, base_w), (mask2_h, mask2_w),
            alignment, x_offset, y_offset
        )
        
        print(f"Calculated offset: x={place_x}, y={place_y}")
        
        # Apply transformation to image2
        aligned_image2 = self.apply_transform_to_image(
            image2, place_x, place_y, base_h, base_w, fill_color
        )
        
        # Apply transformation to mask2
        aligned_mask2 = self.apply_transform_to_mask(
            mask2, place_x, place_y, base_h, base_w
        )
        
        # Ensure mask output dimensions are correct
        if len(base_mask.shape) == 3:
            if len(aligned_mask2.shape) == 2:
                aligned_mask2 = aligned_mask2.unsqueeze(0)
        
        print(f"Output image size: {aligned_image2.shape}")
        print(f"Output mask size: {aligned_mask2.shape}")
        
        return (aligned_image2, base_mask, aligned_mask2)


class ImageAlignByMaskBatch:
    """
    Batch version: Support multiple images alignment simultaneously
    """
    
    def __init__(self):
        self.aligner = ImageAlignByMask()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK",),
                "mask2": ("MASK",),
                "image2": ("IMAGE",),
                "alignment": (["center", "left", "right", "top", "bottom", 
                           "top_left", "top_right", "bottom_left", "bottom_right"], 
                          {"default": "center"}),
            },
            "optional": {
                "mask3": ("MASK",),
                "image3": ("IMAGE",),
                "mask4": ("MASK",),
                "image4": ("IMAGE",),
                "x_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "fill_color": (["white", "black", "transparent"], {"default": "white"}),
                "merge_mode": (["separate", "merged"], {"default": "separate"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("aligned_image2", "aligned_image3", "aligned_image4", "merged_mask")
    FUNCTION = "align_multiple_images"
    CATEGORY = "🐳Pond/image"
    
    def merge_images_with_masks(self, images: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """Merge multiple images using masks"""
        if len(images) == 0:
            return None
            
        result = images[0].clone()
        
        for i in range(1, len(images)):
            if i < len(masks):
                mask = masks[i]
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0).unsqueeze(-1)
                elif len(mask.shape) == 3:
                    mask = mask.unsqueeze(-1)
                
                # Blend images using mask
                result = result * (1 - mask) + images[i] * mask
        
        return result
    
    def align_multiple_images(self, base_mask, mask2, image2, alignment,
                            mask3=None, image3=None, 
                            mask4=None, image4=None,
                            x_offset=0, y_offset=0, fill_color="white", merge_mode="separate"):
        """Align multiple images"""
        
        # Collect all mask-image pairs to align
        mask_image_pairs = [(mask2, image2)]
        if mask3 is not None and image3 is not None:
            mask_image_pairs.append((mask3, image3))
        if mask4 is not None and image4 is not None:
            mask_image_pairs.append((mask4, image4))
        
        # Align all images
        aligned_images = []
        aligned_masks = [base_mask]
        
        # Align each image
        for mask, image in mask_image_pairs:
            aligned_img, _, aligned_mask = self.aligner.align_image_by_mask(
                base_mask, mask, image, alignment, x_offset, y_offset, fill_color
            )
            aligned_images.append(aligned_img)
            aligned_masks.append(aligned_mask)
        
        # Create merged mask
        merged_mask = aligned_masks[0].clone()
        for mask in aligned_masks[1:]:
            merged_mask = torch.maximum(merged_mask, mask)
        
        # If in merge output mode, merge all images
        if merge_mode == "merged" and len(aligned_images) > 1:
            # Merge images using corresponding masks
            masks_for_merge = aligned_masks[1:]  # Skip base mask
            merged_image = self.merge_images_with_masks(aligned_images, masks_for_merge)
            # Replace first output with merged image
            aligned_images[0] = merged_image
        
        # Fill empty outputs
        empty_image = torch.zeros_like(aligned_images[0])
        while len(aligned_images) < 3:
            aligned_images.append(empty_image)
        
        return tuple(aligned_images[:3] + [merged_mask])


NODE_CLASS_MAPPINGS = {
    "ImageAlignByMask": ImageAlignByMask,
    "ImageAlignByMaskBatch": ImageAlignByMaskBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAlignByMask": "🐳Image Align by Mask",
    "ImageAlignByMaskBatch": "🐳Image Align by Mask (Batch)"
}