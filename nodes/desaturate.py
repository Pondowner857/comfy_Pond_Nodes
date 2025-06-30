import torch
import numpy as np

class DesaturateImage:
    """
    Image desaturation node - Simulates Photoshop's desaturate effect
    Converts color images to grayscale (maintaining RGB format)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["luminosity", "average", "maximum", "minimum"], {
                    "default": "luminosity"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "desaturate"
    CATEGORY = "🐳Pond/color"
    
    def desaturate(self, image, method="luminosity", strength=1.0):
        """
        Execute desaturation operation
        
        Args:
            image: Input image tensor (B, H, W, C)
            method: Desaturation method
            strength: Blend factor, controls desaturation strength
        
        Returns:
            Desaturated image
        """
        # Ensure input is in correct format
        batch_size, height, width, channels = image.shape
        
        # Clone image to avoid modifying original data
        result = image.clone()
        
        if method == "luminosity":
            # Use ITU-R BT.709 standard luminance weights (similar to Photoshop)
            # These weights consider human eye sensitivity to different colors
            gray = 0.2126 * image[:, :, :, 0] + \
                   0.7152 * image[:, :, :, 1] + \
                   0.0722 * image[:, :, :, 2]
        
        elif method == "average":
            # Simple average method
            gray = (image[:, :, :, 0] + image[:, :, :, 1] + image[:, :, :, 2]) / 3.0
        
        elif method == "maximum":
            # Use maximum channel value
            gray = torch.max(image[:, :, :, :3], dim=3)[0]
        
        elif method == "minimum":
            # Use minimum channel value
            gray = torch.min(image[:, :, :, :3], dim=3)[0]
        
        # Expand grayscale to all channels
        gray = gray.unsqueeze(3)
        
        # Apply to RGB channels
        for i in range(3):
            result[:, :, :, i] = gray[:, :, :, 0]
        
        # If there's an Alpha channel, keep it unchanged
        if channels == 4:
            result[:, :, :, 3] = image[:, :, :, 3]
        
        # Blend original and desaturated result based on strength
        if strength < 1.0:
            result = image * (1 - strength) + result * strength
        
        return (result,)


class DesaturateImageAdvanced:
    """
    Advanced image desaturation node - Provides more control options
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "R": ("FLOAT", {
                    "default": 0.2126,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "G": ("FLOAT", {
                    "default": 0.7152,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "B": ("FLOAT", {
                    "default": 0.0722,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "normalize": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "desaturate_advanced"
    CATEGORY = "🐳Pond/color"
    
    def desaturate_advanced(self, image, R=0.2126, G=0.7152, 
                          B=0.0722, normalize=True, strength=1.0):
        """
        Execute desaturation with custom weights
        """
        batch_size, height, width, channels = image.shape
        result = image.clone()
        
        # Normalize weights
        if normalize:
            total_weight = R + G + B
            if total_weight > 0:
                R /= total_weight
                G /= total_weight
                B /= total_weight
        
        # Calculate grayscale value
        gray = (R * image[:, :, :, 0] + 
                G * image[:, :, :, 1] + 
                B * image[:, :, :, 2])
        
        # Expand grayscale to all channels
        gray = gray.unsqueeze(3)
        
        # Apply to RGB channels
        for i in range(3):
            result[:, :, :, i] = gray[:, :, :, 0]
        
        # Keep Alpha channel
        if channels == 4:
            result[:, :, :, 3] = image[:, :, :, 3]
        
        # Blend original and result
        if strength < 1.0:
            result = image * (1 - strength) + result * strength
        
        return (result,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "DesaturateImage": DesaturateImage,
    "DesaturateImageAdvanced": DesaturateImageAdvanced,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "DesaturateImage": "🐳Desaturate Image",
    "DesaturateImageAdvanced": "🐳Desaturate Image (Advanced)",
}