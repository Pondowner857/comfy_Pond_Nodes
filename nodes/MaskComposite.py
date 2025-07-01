import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import cv2

class AdvancedMaskImageComposite:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE", {"tooltip": "Background image - base layer"}),  
                "subject_image": ("IMAGE", {"tooltip": "Subject image - image to composite"}),     
                "subject_mask": ("MASK", {"tooltip": "Subject mask - mask for extracting subject"}),
                "position_mask": ("MASK", {"tooltip": "Position mask - white area indicates composite position"}),       
                "scale_mode": (["stretch", "fit", "fill"], {
                    "default": "fit",
                    "tooltip": "Scale mode: stretch=directly stretch to target size, fit=keep ratio to fit target, fill=keep ratio to fill target"
                }),
                "alignment": (["center", "top_left", "top_right", "bottom_left", "bottom_right"], {
                    "default": "center",
                    "tooltip": "Alignment - position within target area"
                }),
                "edge_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Edge blur - higher values create softer edges"
                }),  
                "blend_mode": (["normal", "multiply", "screen", "overlay"], {
                    "default": "normal",
                    "tooltip": "Blend mode: normal=direct overlay, multiply=darkening effect, screen=brightening effect, overlay=contrast enhancement"
                }),
                "feather_edge": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Edge feathering - whether to feather edges for more natural transitions"
                }),
            }
        }
    
    # Add English display names for input parameters
    @classmethod
    def INPUT_NAMES(cls):
        return {
            "background_image": "Background Image",
            "subject_image": "Subject Image", 
            "subject_mask": "Subject Mask",
            "position_mask": "Position Mask",
            "scale_mode": "Scale Mode",
            "alignment": "Alignment",
            "edge_blur": "Edge Blur",
            "blend_mode": "Blend Mode",
            "feather_edge": "Feather Edge"
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite_image", "final_mask")
    FUNCTION = "advanced_composite"
    CATEGORY = "🐳Pond/mask"
    OUTPUT_NODE = False
    
    # Add description
    DESCRIPTION = """
Mask Image Composite Node - Intelligently composites subject image onto background at specified position

Usage:
1. Connect background and subject images
2. Provide subject mask to extract subject
3. Provide position mask to specify composite location (white area)
4. Adjust scale mode and alignment
5. Use edge processing for more natural compositing

Tips:
- White areas in position mask determine composite location
- Edge blur and feathering create more natural composites
- Different blend modes suit different scenarios
"""
    
    def advanced_composite(self, background_image, subject_image, subject_mask, position_mask, 
                          scale_mode, alignment, edge_blur, blend_mode, feather_edge):
        """
        Main function for mask image compositing
        """
        # Convert tensors to numpy arrays
        bg_img = self.tensor_to_numpy(background_image)
        subj_img = self.tensor_to_numpy(subject_image)
        subj_mask_np = self.mask_tensor_to_numpy(subject_mask)
        pos_mask_np = self.mask_tensor_to_numpy(position_mask)
        
        # Validate dimension matching
        if not self.validate_dimensions(bg_img, pos_mask_np, subj_img, subj_mask_np):
            # If dimensions don't match, perform smart resize
            bg_img, pos_mask_np, subj_img, subj_mask_np = self.smart_resize(
                bg_img, pos_mask_np, subj_img, subj_mask_np
            )
        
        # Step 1: Extract subject using subject mask
        extracted_subject = self.extract_subject(subj_img, subj_mask_np)
        
        # Step 2: Analyze position mask to get target area
        target_bbox = self.get_position_bbox(pos_mask_np)
        if target_bbox is None:
            # If no white area detected, return original background
            print("⚠️ Warning: No valid white area detected in position mask, returning original background")
            return (self.numpy_to_tensor(bg_img), self.numpy_to_tensor(pos_mask_np))
        
        # Step 3: Scale extracted subject to target size
        scaled_subject, scaled_mask = self.scale_subject_to_target(
            extracted_subject, subj_mask_np, target_bbox, scale_mode, alignment, bg_img.shape
        )
        
        # Step 4: Apply edge processing
        if feather_edge or edge_blur > 0:
            scaled_mask = self.apply_edge_processing(scaled_mask, edge_blur, feather_edge)
        
        # Step 5: Execute final composite
        result_img = self.blend_images(bg_img, scaled_subject, scaled_mask, blend_mode)
        
        # Convert back to tensor format
        result_tensor = self.numpy_to_tensor(result_img)
        final_mask_tensor = self.numpy_to_tensor(scaled_mask)
        
        return (result_tensor, final_mask_tensor)
    
    def validate_dimensions(self, bg_img, pos_mask, subj_img, subj_mask):
        """Validate input dimensions"""
        bg_h, bg_w = bg_img.shape[:2]
        pos_h, pos_w = pos_mask.shape[:2]
        subj_h, subj_w = subj_img.shape[:2]
        mask_h, mask_w = subj_mask.shape[:2]
        
        bg_match = (bg_h == pos_h and bg_w == pos_w)
        subj_match = (subj_h == mask_h and subj_w == mask_w)
        
        if not bg_match:
            print(f"📐 Background image size ({bg_w}x{bg_h}) doesn't match position mask size ({pos_w}x{pos_h})")
        if not subj_match:
            print(f"📐 Subject image size ({subj_w}x{subj_h}) doesn't match subject mask size ({mask_w}x{mask_h})")
        
        return bg_match and subj_match
    
    def smart_resize(self, bg_img, pos_mask, subj_img, subj_mask):
        """Smart resize to match requirements"""
        print("🔧 Auto-adjusting dimensions...")
        
        # Resize position mask to match background image
        bg_h, bg_w = bg_img.shape[:2]
        pos_mask_resized = cv2.resize(pos_mask, (bg_w, bg_h))
        
        # Resize subject mask to match subject image
        subj_h, subj_w = subj_img.shape[:2]
        subj_mask_resized = cv2.resize(subj_mask, (subj_w, subj_h))
        
        print("✅ Dimension adjustment complete")
        return bg_img, pos_mask_resized, subj_img, subj_mask_resized
    
    def extract_subject(self, subject_img, subject_mask):
        """Extract subject from image using mask"""
        # Ensure mask has correct dimensions
        if len(subject_mask.shape) == 2:
            mask_3d = np.expand_dims(subject_mask, axis=2)
            mask_3d = np.repeat(mask_3d, 3, axis=2)
        else:
            mask_3d = subject_mask
        
        # Extract subject with transparent background
        extracted = subject_img * mask_3d
        
        return extracted
    
    def get_position_bbox(self, position_mask):
        """Get bounding box of white area from position mask"""
        # Binarize mask
        binary_mask = (position_mask > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        print(f"📍 Detected composite position: x={x}, y={y}, width={w}, height={h}")
        
        return {
            'x': x, 'y': y, 'width': w, 'height': h,
            'x2': x + w, 'y2': y + h
        }
    
    def scale_subject_to_target(self, extracted_subject, subject_mask, target_bbox, scale_mode, alignment, bg_shape):
        """Scale extracted subject to target area"""
        target_w, target_h = target_bbox['width'], target_bbox['height']
        bg_h, bg_w = bg_shape[:2]
        
        # Get subject's valid area
        subject_bbox = self.get_subject_bbox(subject_mask)
        if subject_bbox is None:
            # If no valid subject area, return empty image with background size
            print("⚠️ Warning: No valid subject area detected")
            empty_img = np.zeros((bg_h, bg_w, 3), dtype=np.float32)
            empty_mask = np.zeros((bg_h, bg_w), dtype=np.float32)
            return empty_img, empty_mask
        
        # Crop subject to valid area
        cropped_subject = extracted_subject[
            subject_bbox['y']:subject_bbox['y2'],
            subject_bbox['x']:subject_bbox['x2']
        ]
        cropped_mask = subject_mask[
            subject_bbox['y']:subject_bbox['y2'],
            subject_bbox['x']:subject_bbox['x2']
        ]
        
        print(f"🎯 Scale mode: {scale_mode}")
        
        # Process based on scale mode
        if scale_mode == "stretch":
            # Directly stretch to target size
            scaled_subject = cv2.resize(cropped_subject, (target_w, target_h))
            scaled_mask = cv2.resize(cropped_mask, (target_w, target_h))
        elif scale_mode == "fit":
            # Keep aspect ratio, fit target size
            scaled_subject, scaled_mask = self.scale_with_aspect_ratio(
                cropped_subject, cropped_mask, target_w, target_h, "fit"
            )
        else:  # fill
            # Keep aspect ratio, fill target size
            scaled_subject, scaled_mask = self.scale_with_aspect_ratio(
                cropped_subject, cropped_mask, target_w, target_h, "fill"
            )
        
        # Create final image and mask with same size as background
        final_subject = np.zeros((bg_h, bg_w, 3), dtype=np.float32)
        final_mask = np.zeros((bg_h, bg_w), dtype=np.float32)
        
        # Calculate placement position based on alignment
        placement_x, placement_y = self.calculate_placement(
            target_bbox, scaled_subject.shape, alignment
        )
        
        print(f"📐 Alignment: {alignment}, placement position: ({placement_x}, {placement_y})")
        
        # Add boundary checks and safe cropping
        scaled_h, scaled_w = scaled_subject.shape[:2]
        
        # Ensure placement position doesn't exceed background boundaries
        placement_x = max(0, min(placement_x, bg_w - 1))
        placement_y = max(0, min(placement_y, bg_h - 1))
        
        # Calculate actual available area
        available_w = bg_w - placement_x
        available_h = bg_h - placement_y
        
        # If scaled image exceeds available space, need to crop
        actual_w = min(scaled_w, available_w)
        actual_h = min(scaled_h, available_h)
        
        # If cropping needed, crop from center of scaled image
        if actual_w < scaled_w or actual_h < scaled_h:
            crop_start_x = max(0, (scaled_w - actual_w) // 2)
            crop_start_y = max(0, (scaled_h - actual_h) // 2)
            
            scaled_subject_cropped = scaled_subject[
                crop_start_y:crop_start_y + actual_h,
                crop_start_x:crop_start_x + actual_w
            ]
            scaled_mask_cropped = scaled_mask[
                crop_start_y:crop_start_y + actual_h,
                crop_start_x:crop_start_x + actual_w
            ]
        else:
            scaled_subject_cropped = scaled_subject
            scaled_mask_cropped = scaled_mask
        
        # Safely place subject
        end_x = placement_x + actual_w
        end_y = placement_y + actual_h
        
        try:
            final_subject[placement_y:end_y, placement_x:end_x] = scaled_subject_cropped
            final_mask[placement_y:end_y, placement_x:end_x] = scaled_mask_cropped
        except ValueError as e:
            print(f"🚨 Composite warning: {e}")
            print(f"Target area: [{placement_y}:{end_y}, {placement_x}:{end_x}] = ({end_y-placement_y}, {end_x-placement_x})")
            print(f"Source image size: {scaled_subject_cropped.shape}")
            # If still error, use more conservative method
            min_h = min(end_y - placement_y, scaled_subject_cropped.shape[0])
            min_w = min(end_x - placement_x, scaled_subject_cropped.shape[1])
            final_subject[placement_y:placement_y+min_h, placement_x:placement_x+min_w] = scaled_subject_cropped[:min_h, :min_w]
            final_mask[placement_y:placement_y+min_h, placement_x:placement_x+min_w] = scaled_mask_cropped[:min_h, :min_w]
        
        return final_subject, final_mask
    
    def get_subject_bbox(self, mask):
        """Get subject bounding box"""
        binary_mask = (mask > 0.1).astype(np.uint8)
        coords = np.column_stack(np.where(binary_mask > 0))
        
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return {
            'x': x_min, 'y': y_min,
            'width': x_max - x_min + 1,
            'height': y_max - y_min + 1,
            'x2': x_max + 1, 'y2': y_max + 1
        }
    
    def scale_with_aspect_ratio(self, img, mask, target_w, target_h, mode):
        """Scale keeping aspect ratio"""
        img_h, img_w = img.shape[:2]
        
        # Calculate scale factor
        scale_w = target_w / img_w
        scale_h = target_h / img_h
        
        if mode == "fit":
            scale = min(scale_w, scale_h)
        else:  # fill
            scale = max(scale_w, scale_h)
        
        # Calculate new size
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Scale
        scaled_img = cv2.resize(img, (new_w, new_h))
        scaled_mask = cv2.resize(mask, (new_w, new_h))
        
        # If fit mode and size smaller than target, center placement needed
        if mode == "fit" and (new_w < target_w or new_h < target_h):
            final_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            final_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
            
            start_x = (target_w - new_w) // 2
            start_y = (target_h - new_h) // 2
            
            final_img[start_y:start_y+new_h, start_x:start_x+new_w] = scaled_img
            final_mask[start_y:start_y+new_h, start_x:start_x+new_w] = scaled_mask
            
            return final_img, final_mask
        
        # If fill mode and size larger than target, cropping needed
        elif mode == "fill" and (new_w > target_w or new_h > target_h):
            start_x = max(0, (new_w - target_w) // 2)
            start_y = max(0, (new_h - target_h) // 2)
            
            # Ensure crop area doesn't exceed image boundaries
            end_x = min(start_x + target_w, new_w)
            end_y = min(start_y + target_h, new_h)
            actual_w = end_x - start_x
            actual_h = end_y - start_y
            
            cropped_img = scaled_img[start_y:end_y, start_x:end_x]
            cropped_mask = scaled_mask[start_y:end_y, start_x:end_x]
            
            # If cropped size insufficient, pad with zeros
            if actual_w < target_w or actual_h < target_h:
                final_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
                final_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
                final_img[:actual_h, :actual_w] = cropped_img
                final_mask[:actual_h, :actual_w] = cropped_mask
                return final_img, final_mask
            
            return cropped_img, cropped_mask
        
        return scaled_img, scaled_mask
    
    def calculate_placement(self, target_bbox, subject_shape, alignment):
        """Calculate subject placement position in target area"""
        target_x, target_y = target_bbox['x'], target_bbox['y']
        target_w, target_h = target_bbox['width'], target_bbox['height']
        subj_h, subj_w = subject_shape[:2]
        
        if alignment == "center":
            x = target_x + max(0, (target_w - subj_w) // 2)
            y = target_y + max(0, (target_h - subj_h) // 2)
        elif alignment == "top_left":
            x, y = target_x, target_y
        elif alignment == "top_right":
            x = target_x + max(0, target_w - subj_w)
            y = target_y
        elif alignment == "bottom_left":
            x = target_x
            y = target_y + max(0, target_h - subj_h)
        else:  # bottom_right
            x = target_x + max(0, target_w - subj_w)
            y = target_y + max(0, target_h - subj_h)
        
        return max(0, x), max(0, y)
    
    def apply_edge_processing(self, mask, blur_radius, feather_edge):
        """Apply edge processing effects"""
        processed_mask = mask.copy()
        
        if feather_edge:
            # Feather edges
            print("🎨 Applying edge feathering...")
            processed_mask = self.feather_mask_edges(processed_mask)
        
        if blur_radius > 0:
            # Edge blur
            print(f"🎨 Applying edge blur (radius: {blur_radius})...")
            processed_mask = self.apply_edge_blur(processed_mask, blur_radius)
        
        return processed_mask
    
    def feather_mask_edges(self, mask):
        """Feather mask edges"""
        # Use morphological operations to create feathering effect
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Create distance transform
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        
        # Normalize distance transform
        if dist_transform.max() > 0:
            feathered = dist_transform / dist_transform.max()
            # Apply smooth curve
            feathered = np.power(feathered, 0.5)
        else:
            feathered = mask
        
        return feathered.astype(np.float32)
    
    def tensor_to_numpy(self, tensor):
        """Convert ComfyUI image tensor to numpy array"""
        if len(tensor.shape) == 4:  # batch dimension
            tensor = tensor[0]
        
        # Convert from CHW or HWC format to HWC
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # CHW format
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and ensure data type
        img = tensor.cpu().numpy()
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # Ensure values are in 0-1 range
        img = np.clip(img, 0, 1)
        
        return img
    
    def mask_tensor_to_numpy(self, mask_tensor):
        """Convert mask tensor to numpy array"""
        if len(mask_tensor.shape) == 3:  # Remove batch dimension
            mask_tensor = mask_tensor[0]
        
        mask = mask_tensor.cpu().numpy()
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        # Ensure values are in 0-1 range
        mask = np.clip(mask, 0, 1)
        
        return mask
    
    def numpy_to_tensor(self, img):
        """Convert numpy array to ComfyUI tensor format"""
        # Ensure HWC format
        if len(img.shape) == 2:  # Grayscale
            img = np.expand_dims(img, axis=2)
        
        # Convert to tensor
        tensor = torch.from_numpy(img).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def apply_edge_blur(self, mask, blur_radius):
        """Apply blur effect to mask edges"""
        # Convert mask to 0-255 range for OpenCV processing
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Gaussian blur
        kernel_size = max(3, int(blur_radius * 2) | 1)  # Ensure odd number
        blurred_mask = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), blur_radius)
        
        # Convert back to 0-1 range
        blurred_mask = blurred_mask.astype(np.float32) / 255.0
        
        return blurred_mask
    
    def blend_images(self, background, subject, mask, blend_mode):
        """Blend images based on mask and blend mode"""
        # Ensure all images have consistent dimensions
        bg_h, bg_w = background.shape[:2]
        
        # If subject or mask dimensions don't match, resize to background size
        if subject.shape[:2] != (bg_h, bg_w):
            print(f"🔧 Adjusting subject image size: {subject.shape[:2]} -> ({bg_h}, {bg_w})")
            subject = cv2.resize(subject, (bg_w, bg_h))
        
        if mask.shape[:2] != (bg_h, bg_w):
            print(f"🔧 Adjusting mask size: {mask.shape[:2]} -> ({bg_h}, {bg_w})")
            mask = cv2.resize(mask, (bg_w, bg_h))
        
        # Ensure mask has correct dimensions
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        
        print(f"🎨 Applying blend mode: {blend_mode}")
        
        # Process based on blend mode
        if blend_mode == "normal":
            blended = subject
        elif blend_mode == "multiply":
            blended = background * subject
        elif blend_mode == "screen":
            blended = 1 - (1 - background) * (1 - subject)
        elif blend_mode == "overlay":
            # Overlay blend mode
            blended = np.where(background < 0.5,
                             2 * background * subject,
                             1 - 2 * (1 - background) * (1 - subject))
        else:
            blended = subject
        
        # Use mask to blend background and processed subject
        result = background * (1 - mask) + blended * mask
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 1)
        
        print("✅ Image composite complete!")
        
        return result


class MaskBasedImageComposite:
    """
    ComfyUI custom node: Mask-based image composite
    Composite subject image onto background based on mask, with edge blur control
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE", {"tooltip": "Background Image"}),  
                "subject_image": ("IMAGE", {"tooltip": "Subject Image"}),     
                "range_mask": ("MASK", {"tooltip": "Composite Range Mask"}),         
                "subject_mask": ("MASK", {"tooltip": "Subject Shape Mask"}),       
                "edge_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Edge Blur Amount"
                }),  
                "blend_mode": (["normal", "multiply", "screen", "overlay"], {
                    "default": "normal",
                    "tooltip": "Blend Mode"
                }),  
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite_image",)
    FUNCTION = "composite_images"
    CATEGORY = "🐳Pond/mask"
    
    def composite_images(self, background_image, subject_image, range_mask, subject_mask, edge_blur, blend_mode):
        """
        Main function for image compositing
        """
        # Convert tensors to numpy arrays
        bg_img = self.tensor_to_numpy(background_image)
        subj_img = self.tensor_to_numpy(subject_image)
        range_mask_np = self.mask_tensor_to_numpy(range_mask)
        subject_mask_np = self.mask_tensor_to_numpy(subject_mask)
        
        # Get all input dimensions, find maximum size
        bg_h, bg_w = bg_img.shape[:2]
        subj_h, subj_w = subj_img.shape[:2]
        range_h, range_w = range_mask_np.shape[:2]
        subject_h, subject_w = subject_mask_np.shape[:2]
        
        # Calculate maximum canvas size
        max_h = max(bg_h, subj_h, range_h, subject_h)
        max_w = max(bg_w, subj_w, range_w, subject_w)
        
        # Center align all images and masks to maximum canvas
        bg_img_aligned = self.center_align_image(bg_img, max_h, max_w)
        subj_img_aligned = self.center_align_image(subj_img, max_h, max_w)
        range_mask_aligned = self.center_align_mask(range_mask_np, max_h, max_w)
        subject_mask_aligned = self.center_align_mask(subject_mask_np, max_h, max_w)
        
        # Process mask combination
        # range_mask defines overall composite range
        # subject_mask defines specific shape within that range
        combined_mask = range_mask_aligned * subject_mask_aligned
        
        # Edge blur processing
        if edge_blur > 0:
            combined_mask = self.apply_edge_blur(combined_mask, edge_blur)
        
        # Execute image blending
        result_img = self.blend_images(bg_img_aligned, subj_img_aligned, combined_mask, blend_mode)
        
        # Convert back to tensor format
        result_tensor = self.numpy_to_tensor(result_img)
        
        return (result_tensor,)
    
    def center_align_image(self, img, target_h, target_w):
        """Center align image to target size"""
        current_h, current_w = img.shape[:2]
        
        # If already target size, return directly
        if current_h == target_h and current_w == target_w:
            return img
        
        # Create target size canvas, fill with black
        if len(img.shape) == 3:  # Color image
            canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        else:  # Grayscale image
            canvas = np.zeros((target_h, target_w), dtype=img.dtype)
        
        # Calculate center position
        start_y = (target_h - current_h) // 2
        start_x = (target_w - current_w) // 2
        end_y = start_y + current_h
        end_x = start_x + current_w
        
        # Place original image at canvas center
        canvas[start_y:end_y, start_x:end_x] = img
        
        return canvas
    
    def center_align_mask(self, mask, target_h, target_w):
        """Center align mask to target size"""
        current_h, current_w = mask.shape[:2]
        
        # If already target size, return directly
        if current_h == target_h and current_w == target_w:
            return mask
        
        # Create target size canvas, fill with 0 (black mask)
        canvas = np.zeros((target_h, target_w), dtype=mask.dtype)
        
        # Calculate center position
        start_y = (target_h - current_h) // 2
        start_x = (target_w - current_w) // 2
        end_y = start_y + current_h
        end_x = start_x + current_w
        
        # Place original mask at canvas center
        canvas[start_y:end_y, start_x:end_x] = mask
        
        return canvas

    def tensor_to_numpy(self, tensor):
        """Convert ComfyUI image tensor to numpy array"""
        if len(tensor.shape) == 4:  # batch dimension
            tensor = tensor[0]
        
        # Convert from CHW or HWC format to HWC
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # CHW format
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and ensure data type
        img = tensor.cpu().numpy()
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # Ensure values are in 0-1 range
        img = np.clip(img, 0, 1)
        
        return img
    
    def mask_tensor_to_numpy(self, mask_tensor):
        """Convert mask tensor to numpy array"""
        if len(mask_tensor.shape) == 3:  # Remove batch dimension
            mask_tensor = mask_tensor[0]
        
        mask = mask_tensor.cpu().numpy()
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        # Ensure values are in 0-1 range
        mask = np.clip(mask, 0, 1)
        
        return mask
    
    def numpy_to_tensor(self, img):
        """Convert numpy array to ComfyUI tensor format"""
        # Ensure HWC format
        if len(img.shape) == 2:  # Grayscale
            img = np.expand_dims(img, axis=2)
        
        # Convert to tensor
        tensor = torch.from_numpy(img).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def apply_edge_blur(self, mask, blur_radius):
        """Apply blur effect to mask edges"""
        # Convert mask to 0-255 range for OpenCV processing
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Gaussian blur
        kernel_size = max(3, int(blur_radius * 2) | 1)  # Ensure odd number
        blurred_mask = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), blur_radius)
        
        # Convert back to 0-1 range
        blurred_mask = blurred_mask.astype(np.float32) / 255.0
        
        return blurred_mask
    
    def blend_images(self, background, subject, mask, blend_mode):
        """Blend images based on mask and blend mode"""
        # Ensure mask has correct dimensions
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        
        # Process based on blend mode
        if blend_mode == "normal":
            blended = subject
        elif blend_mode == "multiply":
            blended = background * subject
        elif blend_mode == "screen":
            blended = 1 - (1 - background) * (1 - subject)
        elif blend_mode == "overlay":
            # Overlay blend mode
            blended = np.where(background < 0.5,
                             2 * background * subject,
                             1 - 2 * (1 - background) * (1 - subject))
        else:
            blended = subject
        
        # Use mask to blend background and processed subject
        result = background * (1 - mask) + blended * mask
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 1)
        
        return result



# ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "AdvancedMaskImageComposite": AdvancedMaskImageComposite,
    "MaskBasedImageComposite": MaskBasedImageComposite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedMaskImageComposite": "🐳Mask Image Composite",
    "MaskBasedImageComposite": "🎭 Mask Image Composite"
}