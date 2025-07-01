import comfy.sd

class Pond_c_f:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "头发": ("BOOLEAN", {"default": False}),
                "左眼": ("BOOLEAN", {"default": False}),
                "右眼": ("BOOLEAN", {"default": False}),
                "左眉": ("BOOLEAN", {"default": False}),
                "右眉": ("BOOLEAN", {"default": False}),
                "鼻子": ("BOOLEAN", {"default": False}),
                "嘴巴": ("BOOLEAN", {"default": False}),
                "牙齿": ("BOOLEAN", {"default": False}),
                "左耳": ("BOOLEAN", {"default": False}),
                "右耳": ("BOOLEAN", {"default": False}),
                "左胳膊": ("BOOLEAN", {"default": False}),
                "右胳膊": ("BOOLEAN", {"default": False}),
                "左手": ("BOOLEAN", {"default": False}),
                "右手": ("BOOLEAN", {"default": False}),
                "左腿": ("BOOLEAN", {"default": False}),
                "右腿": ("BOOLEAN", {"default": False}),
                "左脚": ("BOOLEAN", {"default": False}),
                "右脚": ("BOOLEAN", {"default": False}),
                "脸部": ("BOOLEAN", {"default": False}),
                "身体": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "🐳Pond/选择器"

    def process(self, **kwargs):
        # 获取所有被选中的部位的英文名称
        active_parts = []
        body_parts_map = {
            "头发": "hair",
            "左眼": "left eye",
            "右眼": "right eye",
            "左眉": "left eyebrow",
            "右眉": "right eyebrow",
            "鼻子": "nose",
            "嘴巴": "mouth",
            "牙齿": "teeth",
            "左耳": "left ear",
            "右耳": "right ear",
            "左胳膊": "left arm",
            "右胳膊": "right arm",
            "左手": "left hand",
            "右手": "right hand",
            "左腿": "left leg",
            "右腿": "right leg",
            "左脚": "left foot",
            "右脚": "right foot",
            "脸部": "face",
            "身体": "body"
        }

        for part, value in kwargs.items():
            if value:
                active_parts.append(body_parts_map[part])

        # 返回逗号分隔的英文部位名称
        return (", ".join(active_parts),)

NODE_CLASS_MAPPINGS = {
    "Pond_c_f": Pond_c_f
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pond_c_f": "🐳肢体选择器"
}