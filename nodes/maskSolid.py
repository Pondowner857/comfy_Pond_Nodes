import torch

class AutoMaskSolidifier:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "solidify"
    CATEGORY = "🐳Pond/mask"
    
    def solidify(self, mask):
        
        # 确保mask是正确的维度
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        # 将所有大于0的值变为1（只要有一点点白色就变成完全白色）
        result = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
        
        return (result,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AutoMaskSolidifier": AutoMaskSolidifier,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoMaskSolidifier": "🐳遮罩虚实",
}