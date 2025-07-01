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
    CATEGORY = "🐳Pond/颜色"

    def invert(self, image, invert_enabled):
        # 如果开关关闭，直接返回原图
        if not invert_enabled:
            return (image,)
        
        # 将图像从torch张量转换为numpy数组
        image_np = image.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy()
        
        # 对每个图像执行反相操作
        inverted_images = []
        for img in image_np:
            # 使用PIL的ImageOps.invert反相
            pil_img = Image.fromarray(img)
            inverted_pil_img = ImageOps.invert(pil_img)
            
            # 转换回torch张量
            inverted_np = np.array(inverted_pil_img)
            inverted_tensor = torch.from_numpy(inverted_np).float() / 255.0
            inverted_images.append(inverted_tensor)
        
        # 堆叠张量并返回
        return (torch.stack(inverted_images).unsqueeze(0),)

NODE_CLASS_MAPPINGS = {
    "InvertImage": InvertImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InvertImage": "🐳图像反相 (Invert Image)"
}