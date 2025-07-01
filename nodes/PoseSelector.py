import json
from typing import Dict, List, Tuple

class PoseSelectorNode:
    """
    ComfyUI节点，用于选择多个姿势标签
    """
    
    # 姿势数据
    POSE_DATA = {
        "综合": {
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
        "姿态": {
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
        "手势": {
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
        "视线": {
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
        "整体": {
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
        "上半身": {
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
        """定义输入类型"""
        inputs = {
            "required": {
                "separator": (["comma", "space", "newline"], {"default": "comma"}),
            },
            "optional": {}
        }
        
        # 为每个分类创建多个单选输入（每个分类提供3个选择框）
        for category, poses in cls.POSE_DATA.items():
            # 创建选项列表，格式为 "中文 (english)"
            options = ["无"]
            for cn, en in poses.items():
                options.append(f"{cn} ({en})")
            
            # 为每个分类创建3个选择框
            for i in range(1, 4):  # 创建3个选择框
                inputs["optional"][f"select_{category}_{i}"] = (options, {
                    "default": "无",
                    "tooltip": f"选择{category}相关的姿势标签 #{i}"
                })
        
        # 添加自定义标签输入
        inputs["optional"]["custom_tags"] = ("STRING", {
            "default": "",
            "multiline": True,
            "placeholder": "输入自定义标签，用逗号分隔"
        })
        
        return inputs
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("english_tags", "chinese_tags", "combined_tags")
    FUNCTION = "process_poses"
    CATEGORY = "🐳Pond/text"
    
    def process_poses(self, separator="comma", custom_tags="", **kwargs):
        """处理选择的姿势并返回标签"""
        english_tags = []
        chinese_tags = []
        
        # 处理每个分类的选择
        for key, value in kwargs.items():
            if key.startswith("select_") and value and value != "无":
                # 从键名中提取分类名（去掉末尾的_数字）
                key_parts = key.replace("select_", "").rsplit("_", 1)
                category = key_parts[0]
                
                if category in self.POSE_DATA:
                    # value 是一个字符串
                    selected = value
                    # 从 "中文 (english)" 格式中提取
                    if " (" in selected and selected.endswith(")"):
                        cn_part = selected.split(" (")[0]
                        # 在原始数据中查找对应的英文
                        if cn_part in self.POSE_DATA[category]:
                            en_tag = self.POSE_DATA[category][cn_part]
                            # 避免重复添加
                            if en_tag not in english_tags:
                                english_tags.append(en_tag)
                                chinese_tags.append(cn_part)
        
        # 处理自定义标签
        if custom_tags.strip():
            custom_list = [tag.strip() for tag in custom_tags.split(",") if tag.strip()]
            english_tags.extend(custom_list)
            chinese_tags.extend(custom_list)
        
        # 根据分隔符组合标签
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

# 简化版本 - 使用分类选择器
class PoseSelectorSimple:
    """
    ComfyUI节点，使用更便捷的方式选择姿势标签
    """
    
    POSE_DATA = PoseSelectorNode.POSE_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入类型"""
        
        # 为每个分类生成编号列表
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
        
        # 为每个分类创建选择输入
        for category in cls.POSE_DATA.keys():
            # 显示可选姿势列表
            inputs["optional"][f"{category}_列表"] = ("STRING", {
                "default": pose_lists[category],
                "multiline": True,
                "dynamicPrompts": False,
                "tooltip": f"{category}分类的所有可选姿势"
            })
            
            # 输入选择的编号
            inputs["optional"][f"{category}_选择"] = ("STRING", {
                "default": "",
                "placeholder": "输入编号，如: 1,3,5 或 1-5,8,10",
                "tooltip": f"输入要选择的{category}姿势编号"
            })
        
        # 快速预设
        inputs["optional"]["快速预设"] = (["无", "基础站姿", "基础坐姿", "动作姿势", "可爱姿势", "日常动作"], {
            "default": "无"
        })
        
        # 添加自定义标签输入
        inputs["optional"]["custom_tags"] = ("STRING", {
            "default": "",
            "multiline": True,
            "placeholder": "输入自定义标签，用逗号分隔"
        })
        
        return inputs
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "process_poses"
    CATEGORY = "🐳Pond/text"
    
    # 预设定义
    PRESETS = {
        "基础站姿": {"综合": [1, 8], "视线": [7]},  # standing, sitting, look at viewer
        "基础坐姿": {"综合": [8, 22, 34], "视线": [7]},  # sitting相关
        "动作姿势": {"综合": [55, 38], "整体": [8, 9, 10]},  # dynamic pose, fighting_stance等
        "可爱姿势": {"姿态": [23], "手势": [8], "视线": [7]},  # head tilt, V pose等
        "日常动作": {"整体": [9, 10], "上半身": [27, 29, 30]}  # walking, holding相关
    }
    
    def parse_selection(self, selection_str):
        """解析选择字符串，支持 1,3,5 或 1-5,8,10 格式"""
        selected = []
        if not selection_str.strip():
            return selected
            
        parts = selection_str.replace(" ", "").split(",")
        for part in parts:
            if "-" in part:
                # 范围选择
                try:
                    start, end = part.split("-")
                    start, end = int(start), int(end)
                    selected.extend(range(start, end + 1))
                except:
                    pass
            else:
                # 单个选择
                try:
                    selected.append(int(part))
                except:
                    pass
        
        return selected
    
    def process_poses(self, separator="comma", output_format="english", 快速预设="无", custom_tags="", **kwargs):
        """处理选择的姿势并返回标签"""
        selected_tags = []
        
        # 处理预设
        if 快速预设 != "无" and 快速预设 in self.PRESETS:
            preset = self.PRESETS[快速预设]
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
        
        # 处理每个分类的选择
        for category in self.POSE_DATA.keys():
            selection_key = f"{category}_选择"
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
        
        # 处理自定义标签
        if custom_tags.strip():
            custom_list = [tag.strip() for tag in custom_tags.split(",") if tag.strip()]
            selected_tags.extend(custom_list)
        
        # 去重
        selected_tags = list(dict.fromkeys(selected_tags))
        
        # 根据分隔符组合标签
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
    批量姿势生成器 - 生成多组姿势组合
    """
    
    POSE_DATA = PoseSelectorNode.POSE_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 1, "max": 10}),
                "tags_per_batch": ("INT", {"default": 3, "min": 1, "max": 10}),
                "category_weights": ("STRING", {
                    "default": "综合:0.3, 姿态:0.2, 手势:0.2, 视线:0.1, 整体:0.1, 上半身:0.1",
                    "placeholder": "分类:权重, 例如 综合:0.3 (权重总和应为1)"
                }),
                "ensure_tags": ("STRING", {
                    "default": "look at viewer",
                    "placeholder": "每组都包含的标签，逗号分隔"
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("STRING",) * 10  # 最多返回10个
    RETURN_NAMES = tuple(f"batch_{i+1}" for i in range(10))
    FUNCTION = "generate_batches"
    CATEGORY = "🐳Pond/text"
    OUTPUT_IS_LIST = (False,) * 10
    
    def generate_batches(self, batch_count, tags_per_batch, category_weights, ensure_tags, seed):
        """生成多组随机姿势组合"""
        import random
        
        if seed != -1:
            random.seed(seed)
        
        # 解析权重
        weights = {}
        for item in category_weights.split(','):
            if ':' in item:
                cat, weight = item.split(':')
                weights[cat.strip()] = float(weight.strip())
        
        # 解析必须包含的标签
        must_have = [t.strip() for t in ensure_tags.split(',') if t.strip()]
        
        # 生成批次
        batches = []
        for i in range(batch_count):
            selected = must_have.copy()
            remaining = tags_per_batch - len(selected)
            
            # 根据权重随机选择
            for _ in range(remaining):
                # 选择分类
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
        
        # 填充到10个输出
        while len(batches) < 10:
            batches.append("")
        
        return tuple(batches)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "PoseSelector": PoseSelectorNode,
    "PoseSelectorSimple": PoseSelectorSimple,
    "PoseSelectorBatch": PoseSelectorBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseSelector": "🐳Pose Selector (Multi-Select)",
    "PoseSelectorSimple": "🐳Pose Selector (Number Selection)",
    "PoseSelectorBatch": "🐳随机批次姿势"
}