import comfy.sd

class Pond_c_e:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "脸": ("BOOLEAN", {"default": False}),
                "手": ("BOOLEAN", {"default": False}),
                "头发": ("BOOLEAN", {"default": False}),
                "眼睛": ("BOOLEAN", {"default": False}),
                "嘴巴": ("BOOLEAN", {"default": False}),
                "鼻子": ("BOOLEAN", {"default": False}),
                "耳朵": ("BOOLEAN", {"default": False}),
                "身体": ("BOOLEAN", {"default": False}),
                "左手臂": ("BOOLEAN", {"default": False}),
                "右手臂": ("BOOLEAN", {"default": False}),
                "左腿": ("BOOLEAN", {"default": False}),
                "右腿": ("BOOLEAN", {"default": False}),
                "上衣": ("BOOLEAN", {"default": False}),
                "外套": ("BOOLEAN", {"default": False}),
                "T恤": ("BOOLEAN", {"default": False}),
                "衬衫": ("BOOLEAN", {"default": False}),
                "裤子": ("BOOLEAN", {"default": False}),
                "鞋子": ("BOOLEAN", {"default": False}),
                "裙子": ("BOOLEAN", {"default": False}),
                "连衣裙": ("BOOLEAN", {"default": False}),
                "领带": ("BOOLEAN", {"default": False}),
                "腰带": ("BOOLEAN", {"default": False}),
                "背包": ("BOOLEAN", {"default": False}),
                "围巾": ("BOOLEAN", {"default": False}),
                "眼镜": ("BOOLEAN", {"default": False}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "🐳Pond/选择器"

    def process(self, **kwargs):
        # 获取所有被选中的部位的英文名称
        active_parts = []
        body_parts_map = {
            "脸": "face",
            "手": "hand",
            "头发": "hair",
            "眼睛": "eyes",
            "嘴巴": "mouth",
            "鼻子": "nose",
            "耳朵": "ears",
            "身体": "body",
            "左手臂": "left arm",
            "右手臂": "right arm",
            "左腿": "left leg",
            "右腿": "right leg",
            "上衣": "top",
            "外套": "jacket",
            "T恤": "T-shirt",
            "衬衫": "shirt",
            "裤子": "pants",
            "鞋子": "shoes",
            "裙子": "skirt",
            "连衣裙": "dress",
            "领带": "tie",
            "腰带": "belt",
            "背包": "backpack",
            "围巾": "scarf",
            "眼镜": "glasses"
        }

        for part, value in kwargs.items():
            if value:
                active_parts.append(body_parts_map[part])

        # 返回逗号分隔的英文部位名称
        return (", ".join(active_parts),)

NODE_CLASS_MAPPINGS = {
    "Pond_c_e": Pond_c_e
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pond_c_e": "🐳身体部位选择器"
}