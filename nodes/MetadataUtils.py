import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import json

class RemoveMetadata:
    """
    Remove all metadata information from images
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_clean"}),
                "remove_all_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "save_workflow": ("BOOLEAN", {"default": False}),
                "custom_metadata": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_metadata"
    OUTPUT_NODE = True
    CATEGORY = "🐳Pond/metadata"

    def remove_metadata(self, image, filename_prefix="ComfyUI_clean", 
                       remove_all_metadata=True, save_workflow=False, 
                       custom_metadata=""):
        
        # Get output directory
        output_dir = folder_paths.get_output_directory()
        
        # Process batch images
        batch_size = image.shape[0]
        results = []
        
        for batch_idx in range(batch_size):
            # Convert tensor to PIL image
            img_tensor = image[batch_idx]
            img_array = 255. * img_tensor.cpu().numpy()
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # If RGB image
            if img_array.shape[-1] == 3:
                img = Image.fromarray(img_array, mode='RGB')
            # If RGBA image
            elif img_array.shape[-1] == 4:
                img = Image.fromarray(img_array, mode='RGBA')
            else:
                # Grayscale image
                img = Image.fromarray(img_array.squeeze(), mode='L')
            
            # Prepare filename
            file_name = f"{filename_prefix}_{batch_idx:05d}.png"
            file_path = os.path.join(output_dir, file_name)
            
            # Prepare metadata
            metadata = PngInfo()
            
            if not remove_all_metadata:
                # If not removing all metadata, can add custom metadata
                if save_workflow:
                    # Can add workflow information here if needed
                    metadata.add_text("workflow", "cleaned")
                
                if custom_metadata:
                    # Add custom metadata
                    try:
                        custom_dict = json.loads(custom_metadata)
                        for key, value in custom_dict.items():
                            metadata.add_text(str(key), str(value))
                    except:
                        # If not JSON format, add directly as text
                        metadata.add_text("custom", custom_metadata)
                
                # Save image with selective metadata
                img.save(file_path, pnginfo=metadata, compress_level=4)
            else:
                # Completely remove all metadata
                img.save(file_path, compress_level=4)
            
            results.append({
                "filename": file_name,
                "subfolder": "",
                "type": "output"
            })
        
        # Return original image (unmodified)
        return (image,)


class LoadImageWithoutMetadata:
    """
    Load image and automatically remove metadata
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "clear_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "🐳Pond/metadata"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata_info")
    FUNCTION = "load_image"

    def load_image(self, image, clear_metadata=True):
        image_path = folder_paths.get_annotated_filepath(image)
        
        # Load image using PIL
        img = Image.open(image_path)
        
        # Extract metadata information
        metadata_info = ""
        if hasattr(img, 'info'):
            metadata_info = json.dumps(img.info, indent=2, ensure_ascii=False)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_array)[None,]
        
        return (img_tensor, metadata_info)


class MetadataInspector:
    """
    Inspect image metadata information
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "upload_image": (sorted(files), {"image_upload": True}),
            },
        }

    CATEGORY = "🐳Pond/metadata"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("all_metadata", "prompt", "workflow", "image")
    FUNCTION = "inspect_metadata"

    def inspect_metadata(self, upload_image):
        # Use uploaded image
        image_path = folder_paths.get_annotated_filepath(upload_image)
        
        # Load image using PIL
        img = Image.open(image_path)
        
        all_metadata = ""
        prompt = ""
        workflow = ""
        
        if hasattr(img, 'info'):
            # Get all metadata
            all_metadata = json.dumps(img.info, indent=2, ensure_ascii=False)
            
            # Try to extract prompt
            if 'prompt' in img.info:
                prompt = img.info.get('prompt', '')
            
            # Try to extract workflow
            if 'workflow' in img.info:
                workflow_data = img.info.get('workflow', '')
                if workflow_data:
                    try:
                        # Try to parse and format workflow JSON
                        workflow_json = json.loads(workflow_data)
                        workflow = json.dumps(workflow_json, indent=2, ensure_ascii=False)
                    except:
                        workflow = workflow_data
        
        # Convert image to tensor for output
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        
        return (all_metadata if all_metadata else "No metadata", 
                prompt if prompt else "No prompt", 
                workflow if workflow else "No workflow",
                img_tensor)


class BatchMetadataRemover:
    """
    Batch process images in a folder to remove metadata
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_folder": ("STRING", {"default": ""}),
                "output_folder": ("STRING", {"default": "cleaned_images"}),
                "file_pattern": ("STRING", {"default": "*.png"}),
                "keep_original": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("process_status",)
    FUNCTION = "batch_remove"
    OUTPUT_NODE = True
    CATEGORY = "🐳Pond/metadata"

    def batch_remove(self, input_folder, output_folder, file_pattern="*.png", keep_original=True):
        import glob
        
        if not input_folder:
            return ("Error: No input folder specified",)
        
        if not os.path.exists(input_folder):
            return (f"Error: Input folder '{input_folder}' does not exist",)
        
        # Create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Get matching files
        pattern_path = os.path.join(input_folder, file_pattern)
        files = glob.glob(pattern_path)
        
        if not files:
            return (f"No matching files found: {pattern_path}",)
        
        processed = 0
        errors = 0
        
        for file_path in files:
            try:
                # Open image
                img = Image.open(file_path)
                
                # Get filename
                filename = os.path.basename(file_path)
                
                # Prepare output path
                if keep_original:
                    output_path = os.path.join(output_folder, filename)
                else:
                    output_path = file_path
                
                # Save based on original format
                if file_path.lower().endswith('.png'):
                    img.save(output_path, 'PNG', compress_level=4)
                elif file_path.lower().endswith(('.jpg', '.jpeg')):
                    img.save(output_path, 'JPEG', quality=95)
                else:
                    img.save(output_path)
                
                processed += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                errors += 1
        
        status = f"Process complete: {processed} files succeeded, {errors} failed"
        return (status,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "RemoveMetadata": RemoveMetadata,
    "LoadImageWithoutMetadata": LoadImageWithoutMetadata,
    "MetadataInspector": MetadataInspector,
    "BatchMetadataRemover": BatchMetadataRemover,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveMetadata": "🐳Remove Metadata",
    "LoadImageWithoutMetadata": "🐳Load Image (Clear Metadata)",
    "MetadataInspector": "🐳Inspect Metadata",
    "BatchMetadataRemover": "🐳Batch Remove Metadata",
}

# Plugin info
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']