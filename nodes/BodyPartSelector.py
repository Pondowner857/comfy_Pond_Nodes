import comfy.sd

class Pond_c_e:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face": ("BOOLEAN", {"default": False}),
                "hand": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "eyes": ("BOOLEAN", {"default": False}),
                "mouth": ("BOOLEAN", {"default": False}),
                "nose": ("BOOLEAN", {"default": False}),
                "ears": ("BOOLEAN", {"default": False}),
                "body": ("BOOLEAN", {"default": False}),
                "left_arm": ("BOOLEAN", {"default": False}),
                "right_arm": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "top": ("BOOLEAN", {"default": False}),
                "jacket": ("BOOLEAN", {"default": False}),
                "t_shirt": ("BOOLEAN", {"default": False}),
                "shirt": ("BOOLEAN", {"default": False}),
                "pants": ("BOOLEAN", {"default": False}),
                "shoes": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}),
                "tie": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}),
                "backpack": ("BOOLEAN", {"default": False}),
                "scarf": ("BOOLEAN", {"default": False}),
                "glasses": ("BOOLEAN", {"default": False}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "🐳Pond/selector"

    def process(self, **kwargs):
        # Get English names of all selected parts
        active_parts = []
        body_parts_map = {
            "face": "face",
            "hand": "hand",
            "hair": "hair",
            "eyes": "eyes",
            "mouth": "mouth",
            "nose": "nose",
            "ears": "ears",
            "body": "body",
            "left_arm": "left arm",
            "right_arm": "right arm",
            "left_leg": "left leg",
            "right_leg": "right leg",
            "top": "top",
            "jacket": "jacket",
            "t_shirt": "T-shirt",
            "shirt": "shirt",
            "pants": "pants",
            "shoes": "shoes",
            "skirt": "skirt",
            "dress": "dress",
            "tie": "tie",
            "belt": "belt",
            "backpack": "backpack",
            "scarf": "scarf",
            "glasses": "glasses"
        }

        for part, value in kwargs.items():
            if value and part in body_parts_map:
                active_parts.append(body_parts_map[part])

        # Return comma-separated English part names
        return (", ".join(active_parts),)

NODE_CLASS_MAPPINGS = {
    "Pond_c_e": Pond_c_e
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pond_c_e": "🐳Body Part Selector"
}