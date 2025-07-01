import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class MaskRegionExpandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "left": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "expand_region": (["black_region", "white_region"], {"default": "black_region"}),
                "edge_smoothing": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "use_gradient": (["no", "yes"], {"default": "no"})
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand_mask_region"
    CATEGORY = "🐳Pond/mask"

    def normalize_mask(self, mask):
        # Process input mask dimensions to ensure (1, H, W)
        if len(mask.shape) == 2:  # (H, W)
            mask = mask.unsqueeze(0)  # (1, H, W)
        elif len(mask.shape) == 3:  # (B, H, W) or (1, H, W)
            if mask.shape[0] > 1:
                mask = mask[0:1]  # Take only the first one
        elif len(mask.shape) == 4:  # (B, C, H, W) or (B, H, W, C)
            if mask.shape[1] == 1:  # (B, 1, H, W)
                mask = mask.squeeze(1)[0:1]  # (1, H, W)
            elif mask.shape[3] == 1:  # (B, H, W, 1)
                mask = mask.squeeze(3)[0:1]  # (1, H, W)
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")
        
        return mask

    def expand_mask_region(self, mask, left, top, right, bottom, expand_region, edge_smoothing=0, use_gradient="no"):
        """Expand specific regions in the mask using dilation operations with edge smoothing options"""
        # Normalize mask to (1, H, W) format
        mask = self.normalize_mask(mask)
        
        # If no expansion needed, return original mask
        if left == 0 and top == 0 and right == 0 and bottom == 0:
            return (mask,)
        
        # Determine whether to expand black or white regions
        if expand_region == "black_region":
            # Invert mask (1 becomes 0, 0 becomes 1)
            work_mask = 1.0 - mask
        else:
            work_mask = mask.clone()
        
        # Convert to numpy for advanced image processing
        cpu_mask = work_mask.cpu().numpy()[0]  # Get as (H, W) format
        height, width = cpu_mask.shape
        
        # Create masks and distance maps for expansion
        expanded_mask = cpu_mask.copy()
        
        # Create distance transform map for gradient effect
        if use_gradient == "yes":
            # Calculate binary mask (threshold 0.5)
            binary_mask = (cpu_mask > 0.5).astype(np.uint8)
            # Generate distance transform
            distance_map = ndimage.distance_transform_edt(1 - binary_mask)
        
        # Process left-right direction (horizontal expansion)
        if left > 0:
            # Scan each row from right to left
            for y in range(height):
                # Find first non-zero pixel in the row
                for x in range(width):
                    if cpu_mask[y, x] > 0.5:
                        # Expand to the left
                        start = max(0, x - left)
                        if use_gradient == "yes":
                            # Use linear gradient fill
                            for i in range(start, x):
                                # Calculate distance percentage
                                distance_percent = (x - i) / left if left > 0 else 0
                                # Apply gradient effect, farther values are smaller
                                expanded_mask[y, i] = max(expanded_mask[y, i], 1.0 - distance_percent)
                        else:
                            # Hard boundary fill
                            expanded_mask[y, start:x] = 1
                        break
        
        if right > 0:
            # Scan each row from left to right
            for y in range(height):
                # Find last non-zero pixel in the row
                for x in range(width-1, -1, -1):
                    if cpu_mask[y, x] > 0.5:
                        # Expand to the right
                        end = min(width, x + right + 1)
                        if use_gradient == "yes":
                            # Use linear gradient fill
                            for i in range(x+1, end):
                                # Calculate distance percentage
                                distance_percent = (i - x) / right if right > 0 else 0
                                # Apply gradient effect, farther values are smaller
                                expanded_mask[y, i] = max(expanded_mask[y, i], 1.0 - distance_percent)
                        else:
                            # Hard boundary fill
                            expanded_mask[y, x+1:end] = 1
                        break
        
        # Process top-bottom direction (vertical expansion)
        if bottom > 0:  # Bottom means expand toward image bottom (increase y value)
            # Scan each column from top to bottom
            for x in range(width):
                # Find last non-zero pixel in the column
                for y in range(height-1, -1, -1):
                    if cpu_mask[y, x] > 0.5:
                        # Expand downward
                        end = min(height, y + bottom + 1)
                        if use_gradient == "yes":
                            # Use linear gradient fill
                            for i in range(y+1, end):
                                # Calculate distance percentage
                                distance_percent = (i - y) / bottom if bottom > 0 else 0
                                # Apply gradient effect
                                expanded_mask[i, x] = max(expanded_mask[i, x], 1.0 - distance_percent)
                        else:
                            # Hard boundary fill
                            expanded_mask[y+1:end, x] = 1
                        break
        
        if top > 0:  # Top means expand toward image top (decrease y value)
            # Scan each column from bottom to top
            for x in range(width):
                # Find first non-zero pixel in the column
                for y in range(height):
                    if cpu_mask[y, x] > 0.5:
                        # Expand upward
                        start = max(0, y - top)
                        if use_gradient == "yes":
                            # Use linear gradient fill
                            for i in range(start, y):
                                # Calculate distance percentage
                                distance_percent = (y - i) / top if top > 0 else 0
                                # Apply gradient effect
                                expanded_mask[i, x] = max(expanded_mask[i, x], 1.0 - distance_percent)
                        else:
                            # Hard boundary fill
                            expanded_mask[start:y, x] = 1
                        break
        
        # Apply edge smoothing (Gaussian blur)
        if edge_smoothing > 0:
            # Apply Gaussian blur to expanded region
            expanded_mask = ndimage.gaussian_filter(expanded_mask, sigma=edge_smoothing/3)
            
            # Ensure original mask areas are not affected
            if use_gradient != "yes":  # Gradient mode already modifies original areas, so skip this step
                expanded_mask = np.maximum(expanded_mask, cpu_mask)
        
        # Convert back to PyTorch format
        result_mask = torch.from_numpy(expanded_mask).float().unsqueeze(0)
        
        # If expanding black regions, invert mask again
        if expand_region == "black_region":
            result_mask = 1.0 - result_mask
        
        return (result_mask,)

NODE_CLASS_MAPPINGS = {"MaskRegionExpandNode": MaskRegionExpandNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskRegionExpandNode": "🐳Mask Region Expand"}