import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class MaskRemoveNode:
    """
    ComfyUI node: Remove image background based on mask
    Keep white areas, remove black areas
    Output original size and cropped images
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "edge_refinement_type": (["none", "gaussian_blur", "morphological_smooth", "edge_feather"],),
                "refinement_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "crop_margin": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("original_size_image", "cropped_image", "used_mask")
    FUNCTION = "remove_background"
    CATEGORY = "🐳Pond/mask"
    
    def remove_background(self, image, mask, edge_refinement_type, refinement_strength, crop_margin):
        """
        Remove image background based on mask with edge refinement
        
        Args:
            image: Input image tensor (B, H, W, C)
            mask: Mask tensor (B, H, W) or (H, W)
            edge_refinement_type: Edge processing method
            refinement_strength: Processing strength
            crop_margin: Extra margin when cropping
            
        Returns:
            tuple: (original size processed image, cropped image, used mask)
        """
        # Ensure inputs are torch tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        
        # Get image dimensions
        if len(image.shape) == 4:  # (B, H, W, C)
            batch_size, height, width, channels = image.shape
        elif len(image.shape) == 3:  # (H, W, C)
            image = image.unsqueeze(0)  # Add batch dimension
            batch_size, height, width, channels = image.shape
        else:
            raise ValueError("Image format incorrect, should be (B, H, W, C) or (H, W, C)")
        
        # Process mask dimensions
        if len(mask.shape) == 2:  # (H, W)
            mask = mask.unsqueeze(0)  # Add batch dimension (B, H, W)
        elif len(mask.shape) == 3:  # (B, H, W)
            pass
        else:
            raise ValueError("Mask format incorrect, should be (H, W) or (B, H, W)")
        
        # Ensure mask and image sizes match
        if mask.shape[-2:] != (height, width):
            # Resize mask
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(), 
                size=(height, width), 
                mode='nearest'
            ).squeeze(1)
        
        # Ensure mask values are in 0-1 range
        mask = torch.clamp(mask, 0, 1)
        
        # Edge refinement processing
        if edge_refinement_type != "none" and refinement_strength > 0:
            mask = self.refine_mask_edges(mask, edge_refinement_type, refinement_strength)
        
        # Create original size result image
        original_size_image = image.clone()
        
        # Apply mask: keep white areas(1), remove black areas(0)
        for b in range(batch_size):
            for c in range(channels):
                if c < 3:  # RGB channels
                    original_size_image[b, :, :, c] = image[b, :, :, c] * mask[b]
                else:  # Alpha channel
                    if channels == 4:
                        original_size_image[b, :, :, c] = mask[b]
        
        # Add alpha channel if original image doesn't have one
        if channels == 3:
            alpha_channel = mask.unsqueeze(-1)  # (B, H, W, 1)
            original_size_image = torch.cat([original_size_image, alpha_channel], dim=-1)
        
        # Create cropped image list
        cropped_image_list = []
        
        for b in range(batch_size):
            # Find boundaries of non-zero areas in mask
            mask_b = mask[b]
            nonzero_positions = torch.where(mask_b > 0)
            
            if len(nonzero_positions[0]) > 0:  # If there are non-zero areas
                # Calculate bounding box
                y_min = nonzero_positions[0].min().item()
                y_max = nonzero_positions[0].max().item()
                x_min = nonzero_positions[1].min().item()
                x_max = nonzero_positions[1].max().item()
                
                # Add margin
                y_min = max(0, y_min - crop_margin)
                y_max = min(height - 1, y_max + crop_margin)
                x_min = max(0, x_min - crop_margin)
                x_max = min(width - 1, x_max + crop_margin)
                
                # Crop image
                cropped_part = original_size_image[b, y_min:y_max+1, x_min:x_max+1, :]
                cropped_image_list.append(cropped_part)
            else:
                # If no non-zero areas, return small transparent image
                small_image = torch.zeros(1, 1, original_size_image.shape[-1], device=image.device)
                cropped_image_list.append(small_image)
        
        # Find maximum crop size to create uniform batch size
        max_h = max(img.shape[0] for img in cropped_image_list)
        max_w = max(img.shape[1] for img in cropped_image_list)
        
        # Create uniform size cropped image batch
        cropped_image_batch = torch.zeros(batch_size, max_h, max_w, original_size_image.shape[-1], device=image.device)
        
        for b, img in enumerate(cropped_image_list):
            h, w = img.shape[:2]
            # Place cropped image in top-left corner
            cropped_image_batch[b, :h, :w, :] = img
        
        # Ensure output format is correct
        original_size_image = torch.clamp(original_size_image, 0, 1)
        cropped_image_batch = torch.clamp(cropped_image_batch, 0, 1)
        used_mask = torch.clamp(mask, 0, 1)
        
        return (original_size_image, cropped_image_batch, used_mask)
    
    def refine_mask_edges(self, mask, refine_type, strength):
        """
        Refine mask edges
        
        Args:
            mask: Mask tensor
            refine_type: Refinement type
            strength: Refinement strength
            
        Returns:
            refined_mask: Refined mask
        """
        if refine_type == "gaussian_blur":
            return self.gaussian_blur_refine(mask, strength)
        elif refine_type == "morphological_smooth":
            return self.morphological_refine(mask, strength)
        elif refine_type == "edge_feather":
            return self.feather_edges(mask, strength)
        else:
            return mask
    
    def gaussian_blur_refine(self, mask, strength):
        """Gaussian blur edge refinement"""
        # Calculate blur kernel size
        kernel_size = int(strength * 6) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Gaussian blur parameters
        sigma = strength * 2.0
        
        # Add channel dimension for blurring
        mask_blur = mask.unsqueeze(1).float()  # (B, 1, H, W)
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Perform horizontal and vertical blur separately
        kernel_1d = g.view(1, 1, -1, 1)
        mask_blur = F.conv2d(mask_blur, kernel_1d, padding=(kernel_size//2, 0))
        
        kernel_1d = g.view(1, 1, 1, -1)
        mask_blur = F.conv2d(mask_blur, kernel_1d, padding=(0, kernel_size//2))
        
        return mask_blur.squeeze(1)
    
    def morphological_refine(self, mask, strength):
        """Morphological smoothing processing"""
        kernel_size = int(strength * 3) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create morphological kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) / (kernel_size * kernel_size)
        mask_morph = mask.unsqueeze(1).float()
        
        # First erode then dilate (opening operation)
        mask_morph = F.conv2d(mask_morph, kernel, padding=kernel_size//2)
        mask_morph = torch.clamp(mask_morph, 0, 1)
        
        # Then dilate then erode (closing operation)
        mask_morph = 1 - F.conv2d(1 - mask_morph, kernel, padding=kernel_size//2)
        mask_morph = torch.clamp(mask_morph, 0, 1)
        
        return mask_morph.squeeze(1)
    
    def feather_edges(self, mask, strength):
        """Edge feathering processing (pure PyTorch implementation)"""
        # Use multiple Gaussian blurs to achieve edge feathering effect
        feather_radius = max(1, int(strength * 5))
        
        # Create multiple blurred versions with different strengths
        mask_float = mask.unsqueeze(1).float()
        blurred_masks = []
        
        for i in range(1, feather_radius + 1):
            blur_strength = i * 0.5
            kernel_size = int(blur_strength * 4) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            sigma = blur_strength
            coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            coords -= kernel_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            
            # Horizontal blur
            kernel_1d = g.view(1, 1, -1, 1)
            blurred = F.conv2d(mask_float, kernel_1d, padding=(kernel_size//2, 0))
            
            # Vertical blur
            kernel_1d = g.view(1, 1, 1, -1)
            blurred = F.conv2d(blurred, kernel_1d, padding=(0, kernel_size//2))
            
            blurred_masks.append(blurred)
        
        # Blend different strength blur results
        if blurred_masks:
            # Use the last (strongest) blur as base
            result = blurred_masks[-1]
            # Keep some sharpness of original edges
            result = mask_float * 0.3 + result * 0.7
        else:
            result = mask_float
        
        return result.squeeze(1)

# Register node
NODE_CLASS_MAPPINGS = {
    "MaskRemoveNode": MaskRemoveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskRemoveNode": "🐳Mask Remove"
}