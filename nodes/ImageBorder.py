import numpy as np
import torch
from PIL import Image

class SimpleBorderRemover:
    """
    A ComfyUI node that removes borders from transparent images and reconstructs them.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_to_content": ("BOOLEAN", {"default": True}),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "fill_background": ("BOOLEAN", {"default": False}),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False
                }),
                "threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "🐳Pond/image"
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def find_content_bounds(self, img_array, threshold):
        """Find the bounding box of non-transparent content."""
        h, w = img_array.shape[:2]
        
        # Determine what constitutes content
        if img_array.shape[2] == 4:
            # Use alpha channel for RGBA images
            content_mask = img_array[:, :, 3] > threshold
        else:
            # For RGB images, find non-black pixels
            gray = np.mean(img_array[:, :, :3], axis=2)
            content_mask = gray > threshold
        
        if not np.any(content_mask):
            # No content found, return original bounds
            return 0, 0, w, h
        
        # Find where the content is
        rows_with_content = np.any(content_mask, axis=1)
        cols_with_content = np.any(content_mask, axis=0)
        
        # Get the indices where content exists
        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            return 0, 0, w, h
        
        # Get bounding box
        y_min = row_indices[0]
        y_max = row_indices[-1] + 1
        x_min = col_indices[0]
        x_max = col_indices[-1] + 1
        
        return x_min, y_min, x_max, y_max
    
    def process_image(self, image, crop_to_content=True, padding=0, 
                     fill_background=False, background_color="#FFFFFF", threshold=0.01):
        """
        Process the image to remove borders and reconstruct them.
        """
        
        # Get the input tensor
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Work with the first image in batch
        img = image[0]
        
        # Convert to numpy array
        img_np = img.cpu().numpy()
        
        # Check if image has alpha channel
        has_alpha = img_np.shape[2] == 4
        
        # Create or extract alpha channel
        if has_alpha:
            # Image already has alpha channel
            rgb = img_np[:, :, :3]
            alpha = img_np[:, :, 3]
        else:
            # No alpha channel, create one
            rgb = img_np
            # Create alpha from non-black pixels
            gray = np.mean(rgb, axis=2)
            alpha = np.where(gray > threshold, 1.0, 0.0)
            # Combine RGB with alpha
            img_np = np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=2)
        
        # Find content bounds and crop if requested
        if crop_to_content:
            x_min, y_min, x_max, y_max = self.find_content_bounds(img_np, threshold)
            
            # Crop the image and alpha
            img_np = img_np[y_min:y_max, x_min:x_max]
        
        # Add padding if requested
        if padding > 0:
            h, w = img_np.shape[:2]
            
            # Create new padded arrays
            new_h = h + 2 * padding
            new_w = w + 2 * padding
            
            # Create padded image with transparent background
            padded_img = np.zeros((new_h, new_w, 4), dtype=np.float32)
            
            # Copy image to center of padded array
            padded_img[padding:padding+h, padding:padding+w] = img_np
            
            img_np = padded_img
        
        # Extract the final alpha channel for mask
        final_alpha = img_np[:, :, 3].copy()
        
        # Process the image based on background fill option
        if fill_background:
            # Get background color
            bg_color = self.hex_to_rgb(background_color)
            bg_color_normalized = np.array(bg_color) / 255.0
            
            # Create background
            h, w = img_np.shape[:2]
            background = np.ones((h, w, 3), dtype=np.float32) * bg_color_normalized
            
            # Extract RGB and alpha
            rgb = img_np[:, :, :3]
            alpha_expanded = img_np[:, :, 3:4]  # Keep dims for broadcasting
            
            # Composite over background
            composited = rgb * alpha_expanded + background * (1 - alpha_expanded)
            
            # Final output is RGB only
            output_img = composited
        else:
            # Keep the RGBA image
            output_img = img_np
        
        # Convert back to tensors
        # Add batch dimension
        output_image = torch.from_numpy(output_img).unsqueeze(0).float()
        
        # Mask should be [batch_size, height, width] for ComfyUI
        output_mask = torch.from_numpy(final_alpha).unsqueeze(0).float()
        
        # Ensure values are in correct range
        output_image = torch.clamp(output_image, 0.0, 1.0)
        output_mask = torch.clamp(output_mask, 0.0, 1.0)
        
        return (output_image, output_mask)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SimpleBorderRemover": SimpleBorderRemover,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleBorderRemover": "🐳图像边框移除",
}