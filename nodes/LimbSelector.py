import comfy.sd

class Pond_c_f:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hair": ("BOOLEAN", {"default": False}),
                "left_eye": ("BOOLEAN", {"default": False}),
                "right_eye": ("BOOLEAN", {"default": False}),
                "left_eyebrow": ("BOOLEAN", {"default": False}),
                "right_eyebrow": ("BOOLEAN", {"default": False}),
                "nose": ("BOOLEAN", {"default": False}),
                "mouth": ("BOOLEAN", {"default": False}),
                "teeth": ("BOOLEAN", {"default": False}),
                "left_ear": ("BOOLEAN", {"default": False}),
                "right_ear": ("BOOLEAN", {"default": False}),
                "left_arm": ("BOOLEAN", {"default": False}),
                "right_arm": ("BOOLEAN", {"default": False}),
                "left_hand": ("BOOLEAN", {"default": False}),
                "right_hand": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "left_foot": ("BOOLEAN", {"default": False}),
                "right_foot": ("BOOLEAN", {"default": False}),
                "face": ("BOOLEAN", {"default": False}),
                "body": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "🐳Pond/selector"

    def process(self, **kwargs):
        # Get English names of all selected body parts
        active_parts = []
        body_parts_map = {
            "hair": "hair",
            "left_eye": "left eye",
            "right_eye": "right eye",
            "left_eyebrow": "left eyebrow",
            "right_eyebrow": "right eyebrow",
            "nose": "nose",
            "mouth": "mouth",
            "teeth": "teeth",
            "left_ear": "left ear",
            "right_ear": "right ear",
            "left_arm": "left arm",
            "right_arm": "right arm",
            "left_hand": "left hand",
            "right_hand": "right hand",
            "left_leg": "left leg",
            "right_leg": "right leg",
            "left_foot": "left foot",
            "right_foot": "right foot",
            "face": "face",
            "body": "body"
        }

        for part, value in kwargs.items():
            if value and part in body_parts_map:
                active_parts.append(body_parts_map[part])

        # Return comma-separated English body part names
        return (", ".join(active_parts),)

NODE_CLASS_MAPPINGS = {
    "Pond_c_f": Pond_c_f
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pond_c_f": "🐳Limb Selector"
}