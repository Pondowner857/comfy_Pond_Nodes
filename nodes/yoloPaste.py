import torch
import torchvision.transforms.functional as TF
import numpy as np

class YoloImagePasteNode:
    """
    Companion node for YOLO detection node
    Pastes processed images back to original positions
    Supports list input, outputs single composite image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"display": "Original Image"}),
                "paste_images": ("IMAGE", {"display": "Paste Images List"}),
                "bboxes": ("BBOXES", {"display": "Bounding Boxes"}),
                "paste_mode": (["paste_all", "specific_index", "cycle"], {
                    "default": "paste_all",
                    "display": "Paste Mode"
                }),
                "target_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1,
                    "display": "Target Index"
                }),
                "blend_mode": (["overlay", "blend", "mask_blend"], {
                    "default": "overlay",
                    "display": "Blend Mode"
                }),
                "blend_alpha": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "display": "Blend Alpha"
                }),
                "feather_amount": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "Feather Amount"
                })
            },
            "optional": {
                "mask": ("MASK", {"display": "Mask"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite_image", "composite_mask")
    INPUT_IS_LIST = {"paste_images": True}  # Mark paste_images to receive list
    FUNCTION = "paste_images"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "Paste processed image list back to YOLO detected original positions, outputs single composite image. Supports receiving list output from crop node."

    def create_feathered_mask(self, height, width, bbox, feather_amount):
        """Create feathered mask"""
        mask = np.zeros((height, width), dtype=np.float32)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within valid range
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        if x2 > x1 and y2 > y1:
            # Create base mask
            mask[y1:y2, x1:x2] = 1.0
            
            if feather_amount > 0:
                # Apply Gaussian blur for feathering
                try:
                    import cv2
                    kernel_size = feather_amount * 2 + 1
                    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), feather_amount)
                except ImportError:
                    print("Warning: OpenCV not installed, cannot apply feather effect")
        
        return mask

    def resize_and_paste(self, original_img, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask=None):
        """Resize and paste image to specified position"""
        # Ensure input image dimensions are correct
        if len(original_img.shape) == 4:
            original_img = original_img[0]
        if len(paste_img.shape) == 4:
            paste_img = paste_img[0]
        
        # Get original image dimensions
        height, width = original_img.shape[:2]
        
        # Parse bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        target_width = x2 - x1
        target_height = y2 - y1
        
        if target_width <= 0 or target_height <= 0:
            print(f"Warning: Invalid paste area [{x1},{y1},{x2},{y2}]")
            return original_img, torch.zeros((height, width), dtype=torch.float32)
        
        # Resize paste image
        paste_tensor = paste_img.permute(2, 0, 1)  # HWC -> CHW
        resized_paste = TF.resize(
            paste_tensor, 
            [target_height, target_width],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        ).permute(1, 2, 0)  # CHW -> HWC
        
        # Perform paste
        result_img = original_img.clone()
        
        if blend_mode == "overlay":
            result_img[y1:y2, x1:x2, :] = resized_paste
        elif blend_mode == "blend":
            original_region = result_img[y1:y2, x1:x2, :]
            blended = original_region * (1 - blend_alpha) + resized_paste * blend_alpha
            result_img[y1:y2, x1:x2, :] = blended
        elif blend_mode == "mask_blend":
            # Create feathered mask
            feather_mask = self.create_feathered_mask(height, width, bbox, feather_amount)
            feather_mask_tensor = torch.from_numpy(feather_mask).float()
            
            # Create temporary image for blending
            temp_img = original_img.clone()
            temp_img[y1:y2, x1:x2, :] = resized_paste
            
            # Apply mask blending
            for c in range(3):  # RGB channels
                result_img[:, :, c] = original_img[:, :, c] * (1 - feather_mask_tensor) + \
                                     temp_img[:, :, c] * feather_mask_tensor
        
        # Create output mask
        output_mask = np.zeros((height, width), dtype=np.float32)
        output_mask[y1:y2, x1:x2] = 1.0
        output_mask_tensor = torch.from_numpy(output_mask).float()
        
        return result_img, output_mask_tensor

    def paste_images(self, original_image, paste_images, bboxes, paste_mode, 
                    target_index, blend_mode, blend_alpha, feather_amount, mask=None):
        """Execute image pasting - paste multiple images onto one original image"""
        
        # Handle original image input (might be a list)
        if isinstance(original_image, list):
            # If it's a list, use the first image
            original_image = original_image[0]
        
        # Ensure original image is 4-dimensional tensor
        if len(original_image.shape) == 3:
            original_image = original_image.unsqueeze(0)
        
        # Use first original image as base
        base_image = original_image[0].clone()
        height, width = base_image.shape[:2]
        
        # Initialize cumulative mask
        cumulative_mask = torch.zeros((height, width), dtype=torch.float32)
        
        # Process paste images list
        if not isinstance(paste_images, list):
            paste_images = [paste_images]
        
        # Validate input
        if not paste_images:
            print("Error: No paste images provided")
            empty_mask = torch.zeros((1, original_image.shape[1], original_image.shape[2]), dtype=torch.float32)
            return (original_image.unsqueeze(0) if len(original_image.shape) == 3 else original_image[:1], empty_mask)
        
        # Process bounding box data
        if isinstance(bboxes, torch.Tensor):
            bboxes_list = bboxes.tolist()
        else:
            bboxes_list = bboxes
        
        num_paste_images = len(paste_images)
        num_bboxes = len(bboxes_list)
        
        print(f"Number of paste images: {num_paste_images}, Number of bounding boxes: {num_bboxes}")
        
        try:
            if paste_mode == "specific_index":
                # Specific index mode: only paste image at specified index
                if target_index < num_paste_images and target_index < num_bboxes:
                    paste_img = paste_images[target_index]
                    # Ensure image dimensions are correct
                    if isinstance(paste_img, torch.Tensor) and len(paste_img.shape) == 4:
                        paste_img = paste_img[0]
                    bbox = bboxes_list[target_index]
                    base_image, paste_mask = self.resize_and_paste(
                        base_image, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                    )
                    cumulative_mask = torch.maximum(cumulative_mask, paste_mask)
                    print(f"Pasted using specific index {target_index}")
                else:
                    print(f"Warning: Specified index {target_index} out of range")
            
            elif paste_mode == "cycle":
                # Cycle mode: if fewer images than bboxes, cycle through images
                for i in range(num_bboxes):
                    paste_idx = i % num_paste_images
                    paste_img = paste_images[paste_idx]
                    # Ensure image dimensions are correct
                    if isinstance(paste_img, torch.Tensor) and len(paste_img.shape) == 4:
                        paste_img = paste_img[0]
                    bbox = bboxes_list[i]
                    base_image, paste_mask = self.resize_and_paste(
                        base_image, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                    )
                    cumulative_mask = torch.maximum(cumulative_mask, paste_mask)
                print(f"Cycle mode: pasted {num_bboxes} regions")
            
            else:  # paste_all mode (default)
                # Paste all: paste all available images in order
                max_items = min(num_paste_images, num_bboxes)
                
                for i in range(max_items):
                    paste_img = paste_images[i]
                    # Ensure image dimensions are correct
                    if isinstance(paste_img, torch.Tensor) and len(paste_img.shape) == 4:
                        paste_img = paste_img[0]
                    bbox = bboxes_list[i]
                    base_image, paste_mask = self.resize_and_paste(
                        base_image, paste_img, bbox, blend_mode, blend_alpha, feather_amount, mask
                    )
                    cumulative_mask = torch.maximum(cumulative_mask, paste_mask)
                
                print(f"Paste all mode: pasted {max_items} images")
                
                # Provide hints if there's a mismatch
                if num_bboxes > num_paste_images:
                    print(f"Note: {num_bboxes - num_paste_images} bounding boxes have no corresponding paste images")
                elif num_paste_images > num_bboxes:
                    print(f"Note: {num_paste_images - num_bboxes} paste images were not used")
            
            # Add batch dimension and return single image
            final_image = base_image.unsqueeze(0)
            # MASK format should be (batch, height, width), no channel dimension needed
            final_mask = cumulative_mask.unsqueeze(0)
            
            return (final_image, final_mask)
            
        except Exception as e:
            print(f"Error occurred during image pasting: {e}")
            import traceback
            traceback.print_exc()
            
            # Return original image as fallback
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            # Ensure returning tensor not list
            if isinstance(original_image, list):
                return (original_image[0].unsqueeze(0) if len(original_image[0].shape) == 3 else original_image[0], empty_mask)
            else:
                return (original_image[:1], empty_mask)

# Node registration
NODE_CLASS_MAPPINGS = {
    "YoloImagePasteNode": YoloImagePasteNode
}

# Node display name mapping
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloImagePasteNode": "🐳YOLO Image Paste"
}