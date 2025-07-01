import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class MaskFeatherPercentageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "feather_percentage": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "preserve_sharp_edges": (["yes", "no"], {"default": "no"})
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "feather_mask_percentage"
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

    def feather_mask_percentage(self, mask, feather_percentage, preserve_sharp_edges):
        """Feather mask edges based on percentage"""
        # Normalize mask to (1, H, W) format
        mask = self.normalize_mask(mask)
        
        # If percentage is 0, return original mask
        if feather_percentage <= 0.1:
            return (mask,)
        
        # Convert to numpy for advanced image processing
        cpu_mask = mask.cpu().numpy()[0]  # Get as (H, W) format
        height, width = cpu_mask.shape
        
        # Calculate percentage-based feather radius
        # Use smaller dimension as reference
        reference_dimension = min(height, width)
        feather_radius = int(reference_dimension * feather_percentage / 100.0)
        
        # Ensure feather radius is at least 1 pixel
        feather_radius = max(1, feather_radius)
        
        # Create binary mask to get edges
        binary_mask = (cpu_mask > 0.5).astype(np.float32)
        
        if preserve_sharp_edges == "yes":
            # Calculate distance transform
            # Calculate separately for foreground and background, then merge
            dist_fg = ndimage.distance_transform_edt(binary_mask)
            dist_bg = ndimage.distance_transform_edt(1.0 - binary_mask)
            
            # Calculate area near boundaries (for feathering)
            edge_region = np.logical_and(dist_fg <= feather_radius, binary_mask > 0.5)
            
            # Convert distance to feather values (linear mapping)
            feathered_mask = binary_mask.copy()
            feathered_mask[edge_region] = dist_fg[edge_region] / feather_radius
            
            # Ensure values are in 0-1 range
            feathered_mask = np.clip(feathered_mask, 0.0, 1.0)
        else:
            # Use Gaussian blur for feathering
            # First apply Gaussian blur to binary mask
            sigma = feather_radius / 2.0  # Standard deviation of Gaussian kernel
            feathered_mask = ndimage.gaussian_filter(binary_mask, sigma=sigma)
            
            # Ensure values in original area remain close to 1
            feathered_mask = np.maximum(feathered_mask, binary_mask * 0.99)
            
            # Ensure values are in 0-1 range
            feathered_mask = np.clip(feathered_mask, 0.0, 1.0)
        
        # Convert back to PyTorch format
        result_mask = torch.from_numpy(feathered_mask).float().unsqueeze(0)
        
        return (result_mask,)

NODE_CLASS_MAPPINGS = {"MaskFeatherPercentageNode": MaskFeatherPercentageNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskFeatherPercentageNode": "🐳Mask Percentage Feather"}