import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import comfy.utils

class PixelizeNode:
    """
    Convert normal images to pixel art style
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "scale_mode": (["keep_original_size", "scale_to_pixel_grid"],),
                "anti_aliasing": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pixelize"
    CATEGORY = "🐳Pond/image"
    
    def pixelize(self, image, pixel_size, scale_mode, anti_aliasing):
        batch_size, height, width, channels = image.shape
        processed_images = []
        
        for i in range(batch_size):
            # Convert to PIL image
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            if scale_mode == "keep_original_size":
                # Keep original size, just pixelize
                # Calculate pixelized dimensions
                pixel_width = width // pixel_size
                pixel_height = height // pixel_size
                
                # Downscale to pixel size
                downsample_method = Image.LANCZOS if anti_aliasing else Image.NEAREST
                img_small = img_pil.resize((pixel_width, pixel_height), downsample_method)
                
                # Upscale back to original size
                img_pixelated = img_small.resize((width, height), Image.NEAREST)
            else:
                # Scale to pixel grid (ensure each pixel block is complete)
                pixel_width = width // pixel_size
                pixel_height = height // pixel_size
                new_width = pixel_width * pixel_size
                new_height = pixel_height * pixel_size
                
                # First adjust to grid size
                img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
                
                # Pixelize
                downsample_method = Image.LANCZOS if anti_aliasing else Image.NEAREST
                img_small = img_resized.resize((pixel_width, pixel_height), downsample_method)
                img_pixelated = img_small.resize((new_width, new_height), Image.NEAREST)
            
            # Convert back to tensor
            img_np = np.array(img_pixelated).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output,)

class SquarePixelCorrectionNode:
    """
    Correct non-square pixels in pixel art to 1:1 square pixels
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_mode": (["auto_detect", "manual_set"],),
                "pixel_width": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "pixel_height": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "output_mode": (["stretch_image", "add_padding", "crop_image"],),
                "alignment": (["center", "top_left", "top_right", "bottom_left", "bottom_right"],),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "pixel_width", "pixel_height")
    FUNCTION = "correct_pixels"
    CATEGORY = "🐳Pond/image"
    
    def correct_pixels(self, image, detection_mode, pixel_width, pixel_height, output_mode, alignment):
        batch_size, height, width, channels = image.shape
        processed_images = []
        
        for i in range(batch_size):
            # Convert to PIL image
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            # Auto detect pixel size
            if detection_mode == "auto_detect":
                detected_width, detected_height = self._detect_pixel_size(img_pil)
                if detected_width > 0 and detected_height > 0:
                    pixel_width = detected_width
                    pixel_height = detected_height
            
            # Calculate corrected dimensions
            if output_mode == "stretch_image":
                # Calculate required stretch ratio
                if pixel_width > pixel_height:
                    # Need vertical stretch
                    scale_factor = pixel_width / pixel_height
                    new_width = width
                    new_height = int(height * scale_factor)
                else:
                    # Need horizontal stretch
                    scale_factor = pixel_height / pixel_width
                    new_width = int(width * scale_factor)
                    new_height = height
                
                img_corrected = img_pil.resize((new_width, new_height), Image.NEAREST)
                
            elif output_mode == "add_padding":
                # Calculate padding needed
                target_ratio = 1.0  # Target is 1:1
                current_ratio = pixel_width / pixel_height
                
                if current_ratio > target_ratio:
                    # Pixels too wide, need top/bottom padding
                    new_height = int(height * current_ratio)
                    new_width = width
                    
                    # Create new image
                    img_corrected = Image.new(img_pil.mode, (new_width, new_height), (0, 0, 0))
                    
                    # Place original image based on alignment
                    y_offset = self._calculate_offset(new_height - height, alignment, 'vertical')
                    img_corrected.paste(img_pil, (0, y_offset))
                else:
                    # Pixels too tall, need left/right padding
                    new_width = int(width / current_ratio)
                    new_height = height
                    
                    # Create new image
                    img_corrected = Image.new(img_pil.mode, (new_width, new_height), (0, 0, 0))
                    
                    # Place original image based on alignment
                    x_offset = self._calculate_offset(new_width - width, alignment, 'horizontal')
                    img_corrected.paste(img_pil, (x_offset, 0))
                    
            else:  # crop_image
                # Calculate crop dimensions
                if pixel_width > pixel_height:
                    # Need to crop width
                    crop_ratio = pixel_height / pixel_width
                    new_width = int(width * crop_ratio)
                    new_height = height
                    
                    # Calculate crop position based on alignment
                    x_offset = self._calculate_offset(width - new_width, alignment, 'horizontal')
                    img_corrected = img_pil.crop((x_offset, 0, x_offset + new_width, height))
                else:
                    # Need to crop height
                    crop_ratio = pixel_width / pixel_height
                    new_width = width
                    new_height = int(height * crop_ratio)
                    
                    # Calculate crop position based on alignment
                    y_offset = self._calculate_offset(height - new_height, alignment, 'vertical')
                    img_corrected = img_pil.crop((0, y_offset, width, y_offset + new_height))
            
            # Convert back to tensor
            img_np = np.array(img_corrected).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output, pixel_width, pixel_height)
    
    def _detect_pixel_size(self, img):
        """Auto detect pixel size"""
        # Convert image to numpy array
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Detect horizontal pixel size
        pixel_width = 1
        for w in range(1, min(width // 2, 64)):
            # Check if all pixels are multiple of w width
            is_valid = True
            for x in range(0, width - w, w):
                # Check if pixel block is uniform
                block = img_array[:, x:x+w]
                if not self._is_uniform_block(block):
                    is_valid = False
                    break
            if is_valid:
                pixel_width = w
                break
        
        # Detect vertical pixel size
        pixel_height = 1
        for h in range(1, min(height // 2, 64)):
            # Check if all pixels are multiple of h height
            is_valid = True
            for y in range(0, height - h, h):
                # Check if pixel block is uniform
                block = img_array[y:y+h, :]
                if not self._is_uniform_block(block):
                    is_valid = False
                    break
            if is_valid:
                pixel_height = h
                break
        
        return pixel_width, pixel_height
    
    def _is_uniform_block(self, block):
        """Check if pixel block is uniform"""
        if block.size == 0:
            return False
        
        # Get first pixel color
        first_pixel = block.flat[0:block.shape[-1]]
        
        # Check if all pixels are the same
        return np.all(block == first_pixel)
    
    def _calculate_offset(self, total_offset, alignment, direction):
        """Calculate offset based on alignment"""
        if alignment == "center":
            return total_offset // 2
        elif alignment == "top_left":
            return 0
        elif alignment == "top_right":
            return total_offset if direction == 'horizontal' else 0
        elif alignment == "bottom_left":
            return 0 if direction == 'horizontal' else total_offset
        elif alignment == "bottom_right":
            return total_offset
        return 0

class PartialPixelizeNode:
    """
    Partial pixelization node, control pixelized areas through mask
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "blend_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "feather_radius": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "blend_mode": (["normal", "overlay", "soft_light", "hard_light"],),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "preserve_colors": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "partial_pixelize"
    CATEGORY = "🐳Pond/image"
    
    def partial_pixelize(self, image, mask, pixel_size, blend_strength, feather_radius, blend_mode, invert_mask, preserve_colors):
        batch_size, height, width, channels = image.shape
        processed_images = []
        
        for i in range(batch_size):
            # Convert to PIL image
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            # Process mask
            if i < mask.shape[0]:
                mask_tensor = mask[i]
            else:
                mask_tensor = mask[0]  # Use first mask if not enough masks
            
            mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
            
            # Ensure mask size matches
            if mask_pil.size != (width, height):
                mask_pil = mask_pil.resize((width, height), Image.LANCZOS)
            
            # Invert mask
            if invert_mask:
                mask_np = 255 - np.array(mask_pil)
                mask_pil = Image.fromarray(mask_np, mode='L')
            
            # Apply feathering
            if feather_radius > 0:
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
            
            # Create pixelized version
            pixel_width = max(1, width // pixel_size)
            pixel_height = max(1, height // pixel_size)
            
            # Downscale image
            img_small = img_pil.resize((pixel_width, pixel_height), Image.NEAREST)
            
            # If preserving colors, only pixelize shape
            if preserve_colors:
                # Create luminance map
                img_gray = img_pil.convert('L')
                gray_small = img_gray.resize((pixel_width, pixel_height), Image.NEAREST)
                gray_pixelated = gray_small.resize((width, height), Image.NEAREST)
                
                # Apply pixelized luminance to original colors
                img_hsv = img_pil.convert('HSV')
                h, s, v = img_hsv.split()
                img_hsv = Image.merge('HSV', (h, s, gray_pixelated))
                img_pixelated = img_hsv.convert('RGB')
            else:
                # Standard pixelization
                img_pixelated = img_small.resize((width, height), Image.NEAREST)
            
            # Apply blend mode
            if blend_mode == "normal":
                img_blended = img_pixelated
            elif blend_mode == "overlay":
                img_blended = self._overlay_blend(img_pil, img_pixelated)
            elif blend_mode == "soft_light":
                img_blended = self._soft_light_blend(img_pil, img_pixelated)
            elif blend_mode == "hard_light":
                img_blended = self._hard_light_blend(img_pil, img_pixelated)
            
            # Blend with original based on mask and strength
            if blend_strength < 1.0:
                # Adjust mask strength
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
                mask_np = mask_np * blend_strength
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
            
            # Composite using mask
            img_result = Image.composite(img_blended, img_pil, mask_pil)
            
            # Convert back to tensor
            img_np = np.array(img_result).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output,)
    
    def _overlay_blend(self, base, overlay):
        """Overlay blend mode"""
        base_np = np.array(base).astype(np.float32) / 255.0
        overlay_np = np.array(overlay).astype(np.float32) / 255.0
        
        # Overlay formula
        result = np.where(base_np < 0.5,
                         2 * base_np * overlay_np,
                         1 - 2 * (1 - base_np) * (1 - overlay_np))
        
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result, mode='RGB')
    
    def _soft_light_blend(self, base, overlay):
        """Soft light blend mode"""
        base_np = np.array(base).astype(np.float32) / 255.0
        overlay_np = np.array(overlay).astype(np.float32) / 255.0
        
        # Soft light formula
        result = np.where(overlay_np < 0.5,
                         base_np - (1 - 2 * overlay_np) * base_np * (1 - base_np),
                         base_np + (2 * overlay_np - 1) * (np.sqrt(base_np) - base_np))
        
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result, mode='RGB')
    
    def _hard_light_blend(self, base, overlay):
        """Hard light blend mode"""
        base_np = np.array(base).astype(np.float32) / 255.0
        overlay_np = np.array(overlay).astype(np.float32) / 255.0
        
        # Hard light formula (reverse of overlay)
        result = np.where(overlay_np < 0.5,
                         2 * base_np * overlay_np,
                         1 - 2 * (1 - base_np) * (1 - overlay_np))
        
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result, mode='RGB')

class PixelArtEnhanceNode:
    """
    Pixel art enhancement node, providing more pixel processing options
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "process_mode": (["pixelize", "pixel_correct", "pixel_optimize"],),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "color_quantization": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "dithering": ("BOOLEAN", {"default": False}),
                "keep_sharp": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = "🐳Pond/image"
    
    def enhance(self, image, process_mode, pixel_size, color_quantization, dithering, keep_sharp):
        batch_size, height, width, channels = image.shape
        processed_images = []
        
        for i in range(batch_size):
            # Convert to PIL image
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB' if channels == 3 else 'RGBA')
            
            if process_mode == "pixelize":
                # Standard pixelization process
                pixel_width = width // pixel_size
                pixel_height = height // pixel_size
                
                # Apply color quantization
                if color_quantization > 0:
                    img_pil = img_pil.quantize(colors=color_quantization, dither=Image.FLOYDSTEINBERG if dithering else Image.NONE)
                    img_pil = img_pil.convert('RGB')
                
                # Downscale
                img_small = img_pil.resize((pixel_width, pixel_height), Image.NEAREST if keep_sharp else Image.LANCZOS)
                
                # Upscale
                img_processed = img_small.resize((width, height), Image.NEAREST)
                
            elif process_mode == "pixel_correct":
                # Detect and correct non-square pixels
                # Simplified processing, scale directly by height
                img_processed = img_pil.resize((width, width), Image.NEAREST)
                
            else:  # pixel_optimize
                # Optimize pixel art (remove blur, enhance edges)
                # Enhance sharpness
                if keep_sharp:
                    enhancer = ImageEnhance.Sharpness(img_pil)
                    img_pil = enhancer.enhance(2.0)
                
                # Apply nearest neighbor sampling to ensure pixel clarity
                img_processed = img_pil
                
                # Color quantization
                if color_quantization > 0:
                    img_processed = img_processed.quantize(colors=color_quantization, dither=Image.FLOYDSTEINBERG if dithering else Image.NONE)
                    img_processed = img_processed.convert('RGB')
            
            # Convert back to tensor
            img_np = np.array(img_processed).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            processed_images.append(img_tensor)
        
        output = torch.stack(processed_images)
        return (output,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "Pixelize": PixelizeNode,
    "SquarePixelCorrection": SquarePixelCorrectionNode,
    "PartialPixelize": PartialPixelizeNode,
    "PixelArtEnhance": PixelArtEnhanceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixelize": "🐳Pixelize",
    "SquarePixelCorrection": "🐳Square Pixel Correction",
    "PartialPixelize": "🐳Partial Pixelize",
    "PixelArtEnhance": "🐳Pixel Art Enhance"
}