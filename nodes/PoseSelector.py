import json
from typing import Dict, List, Tuple

class PoseSelectorNode:
    """
    ComfyUI node for selecting multiple pose tags
    """
    
    # Pose data - keeping Chinese names for reference but categories in English
    POSE_DATA = {
        "general": {
            "站立": "standing",
            "弯腰": "bent over",
            "弓背": "arched back",
            "拉伸": "stretching",
            "躺着": "lying on back",
            "趴着": "on stomach",
            "侧躺": "on side",
            "坐着": "sitting",
            "w坐割座": "wariza",
            "跨坐": "straddling",
            "四肢着地": "all fours",
            "jack-o": "jack-o' challenge",
            "双腿过头": "legs over head",
            "胎儿姿势": "fetal position",
            "自拍": "selfie",
            "通过腿看": "looking through legs",
            "二郎腿": "crossed_legs",
            "跪姿": "kneel",
            "萝莉坐": "kneeling&setting on floot",
            "裸露的肩膀": "bare shoulders",
            "坐在地上": "sitting on the ground",
            "提裙": "Skirt lift",
            "一字马": "standing split",
            "手臂在背后": "arms behind back",
            "狗趴式": "doggystyle",
            "鸭子坐（女子座）": "wariza",
            "泡温泉": "half body under water",
            "张开腿": "spread legs",
            "趴着翘臀": "top-down_bottom-up",
            "開腳": "open your legs wide",
            "漏腋": "armpits",
            "坐在地上(XWX)": "w-sitting on the ground",
            "战斗姿态": "fighting_stance",
            "坐在椅子上": "sitting on chair",
            "瑜伽": "yoga",
            "绝对空域（大腿三角）": "thigh gap",
            "骑马": "horse riding",
            "掀裙子": "skirt_lift",
            "行走": "walk",
            "鸭子坐": "wariza",
            "正骑乘": "girl on top",
            "祈祷": "pray",
            "蹲着": "squatting",
            "坐在床上": "sitting on bed",
            "翘PP": "top-down bottom-up",
            "抱膝": "huddle, clasp knees",
            "公主抱": "princess carry",
            "侧躺着": "Lie on your side,",
            "**": "groping",
            "撩起衣服": "clothes_lift",
            "盘腿坐": "indian style,",
            "动态姿势": "dynamic pose",
            "敬礼": "salute"
        },
        "posture": {
            "侧身坐": "yokozuwari",
            "鸭子坐": "ahirusuwari",
            "盘腿": "indian style",
            "跪着": "kneeling",
            "躬躯": "arched back",
            "膝枕": "lap pillow",
            "学猫叫": "paw pose",
            "单膝跪地": "one knee",
            "蜷起身子侧躺": "fetal position",
            "仰卧": "on back",
            "俯卧": "on stomach",
            "坐着": "sitting",
            "屈膝抱腿坐": "hugging own legs",
            "立式跨骑": "upright straddle",
            "站着": "standing",
            "蹲着": "squatting",
            "绑在十字架上": "crucifixion",
            "双腿缠绕": "leg lock",
            "四肢着地": "all fours",
            "戴耳机": "hand on headphones",
            "鬼姿势": "ghost pose",
            "回头": "turning around",
            "歪头": "head tilt",
            "前倾": "leaning forward"
        },
        "gesture": {
            "嘘手势": "shushing",
            "翘大拇指": "thumbs up",
            "手放脑后": "arms behind head",
            "手放身后": "arms behind back",
            "手插口袋": "hand in pocket",
            "双手插口袋": "hands in pocket",
            "十指相扣": "interlocked fingers",
            "V字手势": "victory pose",
            "手在地板上": "hand on floor",
            "手在额头上": "hand on forehead",
            "手在肚子上": "hand on own stomach",
            "手在肩膀上": "arm over shoulder",
            "手搭别人的腿": "hand on another's leg",
            "手搭别人的腰": "hand on another's waist",
            "双手合十": "own hands clasped",
            "翼展双臂": "wide open arms",
            "手放嘴边": "hand to mouth",
            "手枪手势": "finger gun",
            "猫爪手势": "cat pose"
        },
        "gaze": {
            "远眺": "looking afar",
            "照镜子": "looking at mirror",
            "看手机": "looking at phone",
            "看向别处": "looking away",
            "透过刘海看": "visible through hair",
            "透过眼镜看": "looking over glasses",
            "面向观者": "look at viewer",
            "靠近观者": "close to viewer",
            "动态角度": "dynamic angle",
            "舞台角度": "dramatic angle",
            "凝视": "stare",
            "向上看": "looking up",
            "向下看": "looking down",
            "看向旁边": "looking to the side",
            "移开目光": "looking away"
        },
        "overall": {
            "嗅闻": "smelling",
            "公主抱": "princess carry",
            "拥抱": "hug",
            "背对背": "back-to-back",
            "耶": "peace symbol",
            "调整过膝袜": "adjusting_thighhigh",
            "抓住": "grabbing",
            "战斗姿态": "fighting_stance",
            "走": "walking",
            "跑": "running",
            "跨坐": "straddling",
            "跳": "jump",
            "飞": "fly",
            "靠墙": "against wall",
            "躺": "lie",
            "从背后抱": "hug from behind",
            "遛狗": "walk a dog",
            "提裙": "skirt lift",
            "泡温泉": "half body under water",
            "骑马": "horse riding",
            "自拍": "selfie",
            "一字马": "standing split",
            "敬礼": "salute",
            "祈祷": "pray",
            "冥想": "doing a meditation"
        },
        "upper_body": {
            "伸懒腰": "stretch",
            "托腮": "gill support",
            "牵手": "holding hands",
            "单手叉腰": "hand_on_hip",
            "双手叉腰": "hands_on_hips",
            "招手": "waving",
            "撮头发": "hair scrunchie",
            "拉头发": "hair_pull",
            "抓别人的头发": "grabbing another's hair",
            "竖中指": "middle_finger",
            "弯腰": "bent over",
            "亲吻脸颊": "kissing cheek",
            "亲吻额头": "kissing forehead",
            "踮起脚尖吻": "tiptoe kiss",
            "头顶水果": "fruit on head",
            "咬手套": "glove biting",
            "脸贴脸": "cheek-to-cheek",
            "手牵手": "hand on another's hand",
            "双手交叉": "crossed arms",
            "双手张开伸直": "spread arms",
            "挥动手臂": "waving arms",
            "伸出手臂": "outstretched arm",
            "用手臂支撑": "carrying",
            "搂着手臂": "arm hug",
            "拿着": "holding",
            "拿着餐刀": "holding knife",
            "拿着枪": "holding gun",
            "拿着杯子": "holding cup",
            "拿着食物": "holding food",
            "拿着书": "holding book",
            "拿着魔杖": "holding wand",
            "打着伞": "holding umbrella",
            "捧着花": "holding flower",
            "拿着麦克风": "holding microphone",
            "抱着物品": "object hug",
            "抱着心": "holding heart"
        }
    }
    
    def __init__(self):
        self.selected_poses = {}
        
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types"""
        inputs = {
            "required": {
                "separator": (["comma", "space", "newline"], {"default": "comma"}),
            },
            "optional": {}
        }
        
        # Create multiple selection inputs for each category (3 selection boxes per category)
        for category, poses in cls.POSE_DATA.items():
            # Create options list, format: "Chinese (english)"
            options = ["none"]
            for cn, en in poses.items():
                options.append(f"{cn} ({en})")
            
            # Create 3 selection boxes for each category
            for i in range(1, 4):  # Create 3 selection boxes
                inputs["optional"][f"select_{category}_{i}"] = (options, {
                    "default": "none",
                    "tooltip": f"Select {category} related pose tag #{i}"
                })
        
        # Add custom tags input
        inputs["optional"]["custom_tags"] = ("STRING", {
            "default": "",
            "multiline": True,
            "placeholder": "Enter custom tags, comma separated"
        })
        
        return inputs
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("english_tags", "chinese_tags", "combined_tags")
    FUNCTION = "process_poses"
    CATEGORY = "🐳Pond/text"
    
    def process_poses(self, separator="comma", custom_tags="", **kwargs):
        """Process selected poses and return tags"""
        english_tags = []
        chinese_tags = []
        
        # Process selections for each category
        for key, value in kwargs.items():
            if key.startswith("select_") and value and value != "none":
                # Extract category name from key (remove trailing _number)
                key_parts = key.replace("select_", "").rsplit("_", 1)
                category = key_parts[0]
                
                if category in self.POSE_DATA:
                    # value is a string
                    selected = value
                    # Extract from "Chinese (english)" format
                    if " (" in selected and selected.endswith(")"):
                        cn_part = selected.split(" (")[0]
                        # Find corresponding English in original data
                        if cn_part in self.POSE_DATA[category]:
                            en_tag = self.POSE_DATA[category][cn_part]
                            # Avoid duplicates
                            if en_tag not in english_tags:
                                english_tags.append(en_tag)
                                chinese_tags.append(cn_part)
        
        # Process custom tags
        if custom_tags.strip():
            custom_list = [tag.strip() for tag in custom_tags.split(",") if tag.strip()]
            english_tags.extend(custom_list)
            chinese_tags.extend(custom_list)
        
        # Combine tags based on separator
        if separator == "comma":
            sep = ", "
        elif separator == "space":
            sep = " "
        else:  # newline
            sep = "\n"
        
        english_result = sep.join(english_tags)
        chinese_result = sep.join(chinese_tags)
        combined_result = sep.join([f"{cn} ({en})" for cn, en in zip(chinese_tags, english_tags)])
        
        return (english_result, chinese_result, combined_result)

# Simplified version - using category selector
class PoseSelectorSimple:
    """
    ComfyUI node for selecting pose tags in a more convenient way
    """
    
    POSE_DATA = PoseSelectorNode.POSE_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types"""
        
        # Generate numbered lists for each category
        pose_lists = {}
        for category, poses in cls.POSE_DATA.items():
            pose_list = []
            for i, (cn, en) in enumerate(poses.items()):
                pose_list.append(f"{i+1}. {cn} ({en})")
            pose_lists[category] = "\n".join(pose_list)
        
        inputs = {
            "required": {
                "separator": (["comma", "space", "newline"], {"default": "comma"}),
                "output_format": (["english", "chinese", "both"], {"default": "english"}),
            },
            "optional": {}
        }
        
        # Create selection inputs for each category
        for category in cls.POSE_DATA.keys():
            # Display available poses list
            inputs["optional"][f"{category}_list"] = ("STRING", {
                "default": pose_lists[category],
                "multiline": True,
                "dynamicPrompts": False,
                "tooltip": f"All available poses in {category} category"
            })
            
            # Input for selected numbers
            inputs["optional"][f"{category}_selection"] = ("STRING", {
                "default": "",
                "placeholder": "Enter numbers, e.g.: 1,3,5 or 1-5,8,10",
                "tooltip": f"Enter numbers of {category} poses to select"
            })
        
        # Quick presets
        inputs["optional"]["quick_preset"] = (["none", "basic_standing", "basic_sitting", "action_poses", "cute_poses", "daily_actions"], {
            "default": "none"
        })
        
        # Add custom tags input
        inputs["optional"]["custom_tags"] = ("STRING", {
            "default": "",
            "multiline": True,
            "placeholder": "Enter custom tags, comma separated"
        })
        
        return inputs
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "process_poses"
    CATEGORY = "🐳Pond/text"
    
    # Preset definitions
    PRESETS = {
        "basic_standing": {"general": [1, 8], "gaze": [7]},  # standing, sitting, look at viewer
        "basic_sitting": {"general": [8, 22, 34], "gaze": [7]},  # sitting related
        "action_poses": {"general": [55, 38], "overall": [8, 9, 10]},  # dynamic pose, fighting_stance etc
        "cute_poses": {"posture": [23], "gesture": [8], "gaze": [7]},  # head tilt, V pose etc
        "daily_actions": {"overall": [9, 10], "upper_body": [27, 29, 30]}  # walking, holding related
    }
    
    def parse_selection(self, selection_str):
        """Parse selection string, supports 1,3,5 or 1-5,8,10 format"""
        selected = []
        if not selection_str.strip():
            return selected
            
        parts = selection_str.replace(" ", "").split(",")
        for part in parts:
            if "-" in part:
                # Range selection
                try:
                    start, end = part.split("-")
                    start, end = int(start), int(end)
                    selected.extend(range(start, end + 1))
                except:
                    pass
            else:
                # Single selection
                try:
                    selected.append(int(part))
                except:
                    pass
        
        return selected
    
    def process_poses(self, separator="comma", output_format="english", quick_preset="none", custom_tags="", **kwargs):
        """Process selected poses and return tags"""
        selected_tags = []
        
        # Process presets
        if quick_preset != "none" and quick_preset in self.PRESETS:
            preset = self.PRESETS[quick_preset]
            for category, indices in preset.items():
                if category in self.POSE_DATA:
                    poses_list = list(self.POSE_DATA[category].items())
                    for idx in indices:
                        if 0 < idx <= len(poses_list):
                            cn, en = poses_list[idx - 1]
                            if output_format == "english":
                                selected_tags.append(en)
                            elif output_format == "chinese":
                                selected_tags.append(cn)
                            else:  # both
                                selected_tags.append(f"{cn} ({en})")
        
        # Process selections for each category
        for category in self.POSE_DATA.keys():
            selection_key = f"{category}_selection"
            if selection_key in kwargs and kwargs[selection_key]:
                selected_indices = self.parse_selection(kwargs[selection_key])
                poses_list = list(self.POSE_DATA[category].items())
                
                for idx in selected_indices:
                    if 0 < idx <= len(poses_list):
                        cn, en = poses_list[idx - 1]
                        if output_format == "english":
                            selected_tags.append(en)
                        elif output_format == "chinese":
                            selected_tags.append(cn)
                        else:  # both
                            selected_tags.append(f"{cn} ({en})")
        
        # Process custom tags
        if custom_tags.strip():
            custom_list = [tag.strip() for tag in custom_tags.split(",") if tag.strip()]
            selected_tags.extend(custom_list)
        
        # Remove duplicates
        selected_tags = list(dict.fromkeys(selected_tags))
        
        # Combine tags based on separator
        if separator == "comma":
            sep = ", "
        elif separator == "space":
            sep = " "
        else:  # newline
            sep = "\n"
        
        result = sep.join(selected_tags)
        return (result,)


class PoseSelectorBatch:
    """
    Batch pose generator - generates multiple pose combinations
    """
    
    POSE_DATA = PoseSelectorNode.POSE_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 1, "max": 10}),
                "tags_per_batch": ("INT", {"default": 3, "min": 1, "max": 10}),
                "category_weights": ("STRING", {
                    "default": "general:0.3, posture:0.2, gesture:0.2, gaze:0.1, overall:0.1, upper_body:0.1",
                    "placeholder": "category:weight, e.g. general:0.3 (weights should sum to 1)"
                }),
                "ensure_tags": ("STRING", {
                    "default": "look at viewer",
                    "placeholder": "Tags to include in every batch, comma separated"
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("STRING",) * 10  # Return up to 10
    RETURN_NAMES = tuple(f"batch_{i+1}" for i in range(10))
    FUNCTION = "generate_batches"
    CATEGORY = "🐳Pond/text"
    OUTPUT_IS_LIST = (False,) * 10
    
    def generate_batches(self, batch_count, tags_per_batch, category_weights, ensure_tags, seed):
        """Generate multiple random pose combinations"""
        import random
        
        if seed != -1:
            random.seed(seed)
        
        # Parse weights
        weights = {}
        for item in category_weights.split(','):
            if ':' in item:
                cat, weight = item.split(':')
                weights[cat.strip()] = float(weight.strip())
        
        # Parse must-have tags
        must_have = [t.strip() for t in ensure_tags.split(',') if t.strip()]
        
        # Generate batches
        batches = []
        for i in range(batch_count):
            selected = must_have.copy()
            remaining = tags_per_batch - len(selected)
            
            # Select based on weights
            for _ in range(remaining):
                # Select category
                categories = list(weights.keys())
                cat_weights = [weights.get(c, 1) for c in categories]
                category = random.choices(categories, weights=cat_weights)[0]
                
                if category in self.POSE_DATA:
                    poses = list(self.POSE_DATA[category].values())
                    if poses:
                        tag = random.choice(poses)
                        if tag not in selected:
                            selected.append(tag)
            
            batches.append(", ".join(selected))
        
        # Pad to 10 outputs
        while len(batches) < 10:
            batches.append("")
        
        return tuple(batches)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PoseSelector": PoseSelectorNode,
    "PoseSelectorSimple": PoseSelectorSimple,
    "PoseSelectorBatch": PoseSelectorBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseSelector": "🐳Pose Selector (Multi-Select)",
    "PoseSelectorSimple": "🐳Pose Selector (Number Selection)",
    "PoseSelectorBatch": "🐳Pose Selector (Batch Random)"
}