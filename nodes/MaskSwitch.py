import torch
import comfy.model_management

class 遮罩切换:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "select_mask": [("mask1", "mask2")],
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "select_mask"
    CATEGORY = "🐳Pond/mask"

    def select_mask(self, mask1, mask2, select_mask):
        if select_mask == "mask1":
            return (mask1,)
        else:
            return (mask2,)

# Make sure this mapping exists for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "遮罩切换": 遮罩切换
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "遮罩切换": "🐳遮罩切换"
}