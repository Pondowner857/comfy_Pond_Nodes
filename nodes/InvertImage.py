from PIL import ImageOps, Image
import torch
import numpy as np

class InvertImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert_enabled": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"
    CATEGORY = "🐳Pond/color"

    def invert(self, image, invert_enabled):
        # If switch is off, return original image directly
        if not invert_enabled:
            return (image,)
        
        # Convert image from torch tensor to numpy array
        image_np = image.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy()
        
        # Perform invert operation for each image
        inverted_images = []
        for img in image_np:
            # Use PIL's ImageOps.invert to invert
            pil_img = Image.fromarray(img)
            inverted_pil_img = ImageOps.invert(pil_img)
            
            # Convert back to torch tensor
            inverted_np = np.array(inverted_pil_img)
            inverted_tensor = torch.from_numpy(inverted_np).float() / 255.0
            inverted_images.append(inverted_tensor)
        
        # Stack tensors and return
        return (torch.stack(inverted_images).unsqueeze(0),)

NODE_CLASS_MAPPINGS = {
    "InvertImage": InvertImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InvertImage": "🐳Invert Image"
}