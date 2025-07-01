import json
from typing import Dict, List, Tuple

class ClothingSelectorNode:
    """
    ComfyUI节点，用于选择多个服装标签
    """
    
    # 服装数据
    CLOTHING_DATA = {
        "连衣裙": {
            "包裹式连衣裙": "wrap dress",
            "百褶长裙": "pleated maxi dress",
            "A字小黑裙": "A-line little black dress",
            "露肩鸡尾酒裙": "off-the-shoulder cocktail dress",
            "波西米亚刺绣裙": "boho-inspired embroidered dress",
            "蕾丝喇叭裙": "fit and flare lace dress",
            "条纹背心裙": "striped sundress",
            "紧身荷叶边裙": "sheath dress with ruffle detailing",
            "挂脖长裙": "halter neck maxi dress",
            "丝绒吊带裙": "velvet slip dress",
            "层叠荷叶边裙": "tiered ruffle dress",
            "波点茶会裙": "polka dot tea-length dress",
            "高低裙摆裙": "high-low hemline dress",
            "亮片派对裙": "sequin party dress",
            "花卉中长裙": "floral midi dress",
            "单肩晚礼服": "one-shoulder evening gown",
            "格子裙": "gingham checkered dress",
            "装饰性直筒裙": "embellished shift dress",
            "不对称裹身裙": "asymmetrical wrap dress",
            "蕾丝绑带紧身裙": "lace-up corset dress",
            "钟形袖摆裙": "bell sleeve swing dress",
            "荷叶边裙": "peplum dress",
            "缎面长袍": "satin slip gown",
            "牛仔衬衫裙": "denim shirt dress",
            "波西米亚露肩裙": "bohemian off-the-shoulder dress"
        },
        "上衣": {
            "花卉印花裹身衬衫": "floral-print wrap blouse",
            "荷叶边露肩上衣": "off-the-shoulder ruffled top",
            "短款上衣": "crop top",
            "背心": "tank top",
            "吊带衫": "camisole",
            "T恤": "t-shirt",
            "雪纺衬衫": "chiffon blouse",
            "蕾丝上衣": "lace top",
            "针织衫": "knit sweater",
            "连帽衫": "hoodie",
            "运动文胸": "sports bra",
            "马球衫": "polo shirt",
            "高领毛衣": "turtleneck sweater",
            "V领上衣": "v-neck top",
            "圆领衫": "crew neck shirt",
            "露脐装": "midriff top",
            "绑带上衣": "tie-front top",
            "网眼上衣": "mesh top",
            "亮片上衣": "sequin top",
            "一字肩上衣": "bardot top",
            "斜肩上衣": "one-shoulder top",
            "泡泡袖上衣": "puff sleeve top",
            "喇叭袖上衣": "bell sleeve top",
            "露背上衣": "backless top",
            "挂脖上衣": "halter top"
        },
        "下装": {
            "迷你裙": "mini skirt",
            "中长裙": "midi skirt",
            "长裙": "maxi skirt",
            "百褶裙": "pleated skirt",
            "A字裙": "a-line skirt",
            "铅笔裙": "pencil skirt",
            "牛仔短裤": "denim shorts",
            "高腰裤": "high-waisted pants",
            "阔腿裤": "wide-leg pants",
            "紧身牛仔裤": "skinny jeans",
            "短裤": "shorts",
            "七分裤": "capri pants",
            "工装裤": "cargo pants",
            "运动裤": "joggers",
            "皮革裙": "leather skirt",
            "薄纱裙": "tulle skirt",
            "裙裤": "skort",
            "喇叭裤": "flare pants",
            "直筒裤": "straight leg pants",
            "锥形裤": "tapered pants",
            "纸袋裤": "paperbag pants",
            "灯笼裤": "harem pants",
            "百慕大短裤": "bermuda shorts",
            "自行车短裤": "bike shorts",
            "瑜伽裤": "yoga pants",
            "打底裤": "leggings"
        },
        "泳装": {
            "三角比基尼": "triangle bikini",
            "高腰比基尼": "high-waisted bikini",
            "抹胸比基尼": "bandeau bikini",
            "挂脖比基尼": "halter bikini",
            "一件式泳衣": "one-piece swimsuit",
            "分体泳衣": "tankini",
            "泳裙": "swim dress",
            "运动型泳衣": "athletic swimsuit",
            "深V泳衣": "plunge swimsuit",
            "镂空泳衣": "cut-out swimsuit",
            "荷叶边泳衣": "ruffled swimsuit",
            "巴西比基尼": "brazilian bikini",
            "防晒泳衣": "rash guard",
            "潜水服": "wetsuit",
            "复古高腰泳衣": "retro high-waisted swimsuit",
            "绑带比基尼": "string bikini",
            "运动泳裤": "swim shorts",
            "比基尼罩衫": "bikini cover-up",
            "泳帽": "swim cap",
            "花卉印花泳衣": "floral print swimsuit"
        },
        "运动装": {
            "运动文胸": "sports bra",
            "瑜伽裤": "yoga pants",
            "运动短裤": "athletic shorts",
            "紧身衣": "compression top",
            "运动背心": "tank top",
            "运动T恤": "athletic t-shirt",
            "运动外套": "track jacket",
            "运动裙": "tennis skirt",
            "自行车短裤": "cycling shorts",
            "跑步紧身裤": "running tights",
            "健身背心": "fitness tank",
            "速干T恤": "moisture-wicking shirt",
            "运动连体衣": "athletic romper",
            "瑜伽上衣": "yoga top",
            "运动内衣": "sports underwear",
            "压缩裤": "compression leggings",
            "网球裙": "tennis dress",
            "高尔夫裙": "golf skirt",
            "马拉松背心": "marathon singlet",
            "健身短裤": "gym shorts"
        },
        "内衣": {
            "蕾丝文胸": "lace bra",
            "无痕内裤": "seamless panties",
            "运动内衣": "sports bra",
            "塑身衣": "shapewear",
            "吊带背心": "camisole",
            "三角内裤": "bikini panties",
            "平角内裤": "boyshorts",
            "丁字裤": "thong",
            "高腰内裤": "high-waisted panties",
            "无钢圈文胸": "wireless bra",
            "聚拢文胸": "push-up bra",
            "美背文胸": "racerback bra",
            "硅胶文胸": "silicone bra",
            "睡衣": "pajamas",
            "睡袍": "robe",
            "吊带睡裙": "nightgown",
            "情趣内衣": "lingerie",
            "连体衣": "bodysuit",
            "束腰": "corset",
            "吊袜带": "garter belt"
        },
        "外套": {
            "牛仔夹克": "denim jacket",
            "皮夹克": "leather jacket",
            "风衣": "trench coat",
            "西装外套": "blazer",
            "开衫": "cardigan",
            "飞行员夹克": "bomber jacket",
            "派克大衣": "parka",
            "羽绒服": "puffer jacket",
            "毛呢大衣": "wool coat",
            "斗篷": "cape",
            "针织外套": "knit cardigan",
            "运动外套": "track jacket",
            "机车夹克": "moto jacket",
            "军装外套": "military jacket",
            "雨衣": "raincoat",
            "马甲": "vest",
            "披肩": "shawl",
            "短款外套": "cropped jacket",
            "长款大衣": "long coat",
            "毛皮大衣": "fur coat"
        },
        "特殊服装": {
            "旗袍": "cheongsam",
            "韩服": "hanbok",
            "和服": "kimono",
            "晚礼服": "evening gown",
            "婚纱": "wedding dress",
            "舞会礼服": "ball gown",
            "鸡尾酒裙": "cocktail dress",
            "空姐制服": "flight attendant uniform",
            "护士服": "nurse uniform",
            "女仆装": "maid outfit",
            "学生制服": "school uniform",
            "啦啦队服": "cheerleader outfit",
            "舞蹈服": "dance costume",
            "花样滑冰服": "figure skating dress",
            "体操服": "leotard",
            "芭蕾舞裙": "tutu",
            "肚皮舞服": "belly dance costume",
            "拉丁舞裙": "latin dance dress",
            "戏服": "costume",
            "角色扮演服": "cosplay outfit"
        }
    }
    
    def __init__(self):
        self.selected_clothes = {}
        
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
        for category, clothes in cls.CLOTHING_DATA.items():
            # 创建选项列表，格式为 "中文 (english)"
            options = ["无"]
            for cn, en in clothes.items():
                options.append(f"{cn} ({en})")
            
            # 为每个分类创建3个选择框
            for i in range(1, 4):  # 创建3个选择框
                inputs["optional"][f"select_{category}_{i}"] = (options, {
                    "default": "无",
                    "tooltip": f"选择{category}相关的服装标签 #{i}"
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
    FUNCTION = "process_clothes"
    CATEGORY = "🐳Pond/text"
    
    def process_clothes(self, separator="comma", custom_tags="", **kwargs):
        """处理选择的服装并返回标签"""
        english_tags = []
        chinese_tags = []
        
        # 处理每个分类的选择
        for key, value in kwargs.items():
            if key.startswith("select_") and value and value != "无":
                # 从键名中提取分类名（去掉末尾的_数字）
                key_parts = key.replace("select_", "").rsplit("_", 1)
                category = key_parts[0]
                
                if category in self.CLOTHING_DATA:
                    # value 是一个字符串
                    selected = value
                    # 从 "中文 (english)" 格式中提取
                    if " (" in selected and selected.endswith(")"):
                        cn_part = selected.split(" (")[0]
                        # 在原始数据中查找对应的英文
                        if cn_part in self.CLOTHING_DATA[category]:
                            en_tag = self.CLOTHING_DATA[category][cn_part]
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
class ClothingSelectorSimple:
    """
    ComfyUI节点，使用更便捷的方式选择服装标签
    """
    
    CLOTHING_DATA = ClothingSelectorNode.CLOTHING_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入类型"""
        
        # 为每个分类生成编号列表
        clothing_lists = {}
        for category, clothes in cls.CLOTHING_DATA.items():
            clothes_list = []
            for i, (cn, en) in enumerate(clothes.items()):
                clothes_list.append(f"{i+1}. {cn} ({en})")
            clothing_lists[category] = "\n".join(clothes_list)
        
        inputs = {
            "required": {
                "separator": (["comma", "space", "newline"], {"default": "comma"}),
                "output_format": (["english", "chinese", "both"], {"default": "english"}),
            },
            "optional": {}
        }
        
        # 为每个分类创建选择输入
        for category in cls.CLOTHING_DATA.keys():
            # 显示可选服装列表
            inputs["optional"][f"{category}_列表"] = ("STRING", {
                "default": clothing_lists[category],
                "multiline": True,
                "dynamicPrompts": False,
                "tooltip": f"{category}分类的所有可选服装"
            })
            
            # 输入选择的编号
            inputs["optional"][f"{category}_选择"] = ("STRING", {
                "default": "",
                "placeholder": "输入编号，如: 1,3,5 或 1-5,8,10",
                "tooltip": f"输入要选择的{category}服装编号"
            })
        
        # 快速预设
        inputs["optional"]["快速预设"] = (["无", "休闲装", "正装", "运动装", "泳装搭配", "夏日装扮", "派对装"], {
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
    FUNCTION = "process_clothes"
    CATEGORY = "🐳Pond/text"
    
    # 预设定义
    PRESETS = {
        "休闲装": {"上衣": [6, 10], "下装": [7, 11], "外套": [1]},  # T恤、连帽衫、牛仔短裤、短裤、牛仔夹克
        "正装": {"连衣裙": [3, 16], "上衣": [7], "外套": [4]},  # 小黑裙、晚礼服、雪纺衬衫、西装外套
        "运动装": {"运动装": [1, 2, 3], "上衣": [11]},  # 运动文胸、瑜伽裤、运动短裤、运动文胸
        "泳装搭配": {"泳装": [1, 4, 13], "外套": [16]},  # 三角比基尼、挂脖比基尼、防晒泳衣、披肩
        "夏日装扮": {"连衣裙": [7, 15], "上衣": [3, 5], "下装": [1]},  # 条纹背心裙、花卉中长裙、短上衣、吊带衫、迷你裙
        "派对装": {"连衣裙": [14, 16], "上衣": [19], "下装": [15]}  # 亮片派对裙、单肩晚礼服、亮片上衣、薄纱裙
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
    
    def process_clothes(self, separator="comma", output_format="english", 快速预设="无", custom_tags="", **kwargs):
        """处理选择的服装并返回标签"""
        selected_tags = []
        
        # 处理预设
        if 快速预设 != "无" and 快速预设 in self.PRESETS:
            preset = self.PRESETS[快速预设]
            for category, indices in preset.items():
                if category in self.CLOTHING_DATA:
                    clothes_list = list(self.CLOTHING_DATA[category].items())
                    for idx in indices:
                        if 0 < idx <= len(clothes_list):
                            cn, en = clothes_list[idx - 1]
                            if output_format == "english":
                                selected_tags.append(en)
                            elif output_format == "chinese":
                                selected_tags.append(cn)
                            else:  # both
                                selected_tags.append(f"{cn} ({en})")
        
        # 处理每个分类的选择
        for category in self.CLOTHING_DATA.keys():
            selection_key = f"{category}_选择"
            if selection_key in kwargs and kwargs[selection_key]:
                selected_indices = self.parse_selection(kwargs[selection_key])
                clothes_list = list(self.CLOTHING_DATA[category].items())
                
                for idx in selected_indices:
                    if 0 < idx <= len(clothes_list):
                        cn, en = clothes_list[idx - 1]
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


class ClothingSelectorBatch:
    """
    批量服装生成器 - 生成多组服装组合
    """
    
    CLOTHING_DATA = ClothingSelectorNode.CLOTHING_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 1, "max": 10}),
                "tags_per_batch": ("INT", {"default": 3, "min": 1, "max": 10}),
                "category_weights": ("STRING", {
                    "default": "连衣裙:0.2, 上衣:0.2, 下装:0.2, 泳装:0.1, 运动装:0.1, 内衣:0.1, 外套:0.05, 特殊服装:0.05",
                    "placeholder": "分类:权重, 例如 连衣裙:0.3 (权重总和应为1)"
                }),
                "style_preset": (["随机", "休闲", "正装", "运动", "性感", "日常"], {"default": "随机"}),
                "ensure_tags": ("STRING", {
                    "default": "",
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
    
    # 风格权重预设
    STYLE_WEIGHTS = {
        "休闲": {"连衣裙": 0.1, "上衣": 0.3, "下装": 0.3, "泳装": 0.05, "运动装": 0.1, "内衣": 0.05, "外套": 0.1, "特殊服装": 0.0},
        "正装": {"连衣裙": 0.4, "上衣": 0.2, "下装": 0.2, "泳装": 0.0, "运动装": 0.0, "内衣": 0.0, "外套": 0.15, "特殊服装": 0.05},
        "运动": {"连衣裙": 0.0, "上衣": 0.1, "下装": 0.1, "泳装": 0.1, "运动装": 0.6, "内衣": 0.05, "外套": 0.05, "特殊服装": 0.0},
        "性感": {"连衣裙": 0.2, "上衣": 0.2, "下装": 0.15, "泳装": 0.2, "运动装": 0.0, "内衣": 0.2, "外套": 0.0, "特殊服装": 0.05},
        "日常": {"连衣裙": 0.15, "上衣": 0.25, "下装": 0.25, "泳装": 0.0, "运动装": 0.1, "内衣": 0.05, "外套": 0.15, "特殊服装": 0.05}
    }
    
    def generate_batches(self, batch_count, tags_per_batch, category_weights, style_preset, ensure_tags, seed):
        """生成多组随机服装组合"""
        import random
        
        if seed != -1:
            random.seed(seed)
        
        # 解析权重或使用预设
        if style_preset != "随机" and style_preset in self.STYLE_WEIGHTS:
            weights = self.STYLE_WEIGHTS[style_preset]
        else:
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
                
                if category in self.CLOTHING_DATA:
                    clothes = list(self.CLOTHING_DATA[category].values())
                    if clothes:
                        tag = random.choice(clothes)
                        if tag not in selected:
                            selected.append(tag)
            
            batches.append(", ".join(selected))
        
        # 填充到10个输出
        while len(batches) < 10:
            batches.append("")
        
        return tuple(batches)


# 服装搭配建议节点
class ClothingOutfitSuggestion:
    """
    根据选择的主要服装推荐搭配
    """
    
    CLOTHING_DATA = ClothingSelectorNode.CLOTHING_DATA
    
    # 扩展的搭配规则
    OUTFIT_RULES = {
        # 连衣裙类
        "wrap dress": ["denim jacket", "ankle boots", "crossbody bag", "belt", "cardigan"],
        "midi dress": ["blazer", "heels", "clutch", "statement necklace", "belt"],
        "maxi dress": ["sandals", "sun hat", "tote bag", "denim jacket", "wedges"],
        "cocktail dress": ["high heels", "clutch", "statement jewelry", "wrap", "evening bag"],
        "sundress": ["sandals", "straw hat", "canvas bag", "cardigan", "espadrilles"],
        "bodycon dress": ["stiletto heels", "clutch", "statement earrings", "choker", "ankle strap heels"],
        "slip dress": ["strappy heels", "delicate jewelry", "clutch", "shawl", "thigh-high boots"],
        
        # 上衣类
        "t-shirt": ["jeans", "sneakers", "baseball cap", "backpack", "bomber jacket"],
        "blouse": ["pencil skirt", "heels", "blazer", "tote bag", "pearl necklace"],
        "crop top": ["high-waisted pants", "sneakers", "choker", "denim jacket", "mini backpack"],
        "tank top": ["shorts", "sandals", "sunglasses", "crossbody bag", "kimono"],
        "hoodie": ["joggers", "sneakers", "beanie", "backpack", "windbreaker"],
        "halter top": ["high-waisted skirt", "heels", "statement earrings", "clutch", "body chain"],
        "lace top": ["leather pants", "stilettos", "clutch", "red lipstick", "statement necklace"],
        
        # 下装类
        "mini skirt": ["crop top", "ankle boots", "bomber jacket", "choker", "crossbody bag"],
        "high-waisted pants": ["tucked-in blouse", "belt", "heels", "blazer", "structured bag"],
        "jeans": ["t-shirt", "sneakers", "denim jacket", "belt", "casual bag"],
        "shorts": ["tank top", "sandals", "sun hat", "beach bag", "kimono"],
        "leather skirt": ["silk blouse", "heels", "clutch", "statement jewelry", "leather jacket"],
        "pencil skirt": ["fitted blouse", "pumps", "structured bag", "belt", "blazer"],
        
        # 运动装类
        "sports bra": ["yoga pants", "athletic shoes", "gym bag", "water bottle", "headband"],
        "yoga pants": ["sports bra", "tank top", "yoga mat", "sneakers", "hoodie"],
        
        # 泳装类
        "bikini": ["beach cover-up", "sun hat", "sandals", "beach bag", "sunglasses"],
        "one-piece swimsuit": ["sarong", "flip-flops", "sun hat", "beach tote", "kimono"],
        
        # 正装类
        "evening gown": ["clutch", "heels", "statement jewelry", "wrap", "evening gloves"],
        "blazer": ["pencil skirt", "blouse", "pumps", "structured bag", "watch"]
    }
    
    # 扩展的风格配饰
    STYLE_ACCESSORIES = {
        "休闲": ["sneakers", "backpack", "baseball cap", "crossbody bag", "sunglasses", "canvas tote"],
        "正式": ["heels", "clutch", "blazer", "pearl necklace", "structured bag", "silk scarf"],
        "运动": ["athletic shoes", "gym bag", "headband", "sports watch", "water bottle", "windbreaker"],
        "派对": ["high heels", "statement jewelry", "evening bag", "bold lipstick", "cocktail ring", "wrap"],
        "日常": ["comfortable shoes", "tote bag", "sunglasses", "watch", "crossbody bag", "cardigan"],
        "街头": ["sneakers", "bucket hat", "chain necklace", "mini backpack", "oversized jacket", "socks"],
        "波西米亚": ["sandals", "fringe bag", "headband", "layered necklaces", "kimono", "anklet"],
        "复古": ["vintage bag", "cat-eye sunglasses", "headscarf", "brooch", "mary jane shoes", "gloves"],
        "极简": ["minimalist bag", "simple jewelry", "loafers", "structured coat", "monochrome scarf", "watch"],
        "浪漫": ["ballet flats", "pearl accessories", "hair bow", "lace gloves", "clutch", "shawl"],
        "朋克": ["combat boots", "leather jacket", "studded bag", "choker", "chain belt", "fingerless gloves"],
        "优雅": ["kitten heels", "silk scarf", "pearl earrings", "structured handbag", "gloves", "brooch"],
        "学院": ["loafers", "messenger bag", "preppy blazer", "knee socks", "headband", "plaid scarf"],
        "度假": ["espadrilles", "straw bag", "sun hat", "oversized sunglasses", "beach cover-up", "anklet"],
        "商务": ["pumps", "laptop bag", "blazer", "silk blouse", "watch", "structured tote"],
        "性感": ["stiletto heels", "body chain", "choker necklace", "thigh-high boots", "statement earrings", "red lipstick"]
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # 创建主要服装选项
        all_clothes = []
        for category, clothes in cls.CLOTHING_DATA.items():
            for cn, en in clothes.items():
                all_clothes.append(f"{cn} ({en}) - {category}")
        
        return {
            "required": {
                "main_clothing": (all_clothes, {
                    "default": all_clothes[0] if all_clothes else "无"
                }),
                "style": ([
                    "休闲", "正式", "运动", "派对", "日常",
                    "街头", "波西米亚", "复古", "极简", "浪漫",
                    "朋克", "优雅", "学院", "度假", "商务", "性感"
                ], {"default": "日常"}),
                "tag_count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "display": "slider"
                }),
                "include_accessories": (["是", "否"], {"default": "是"}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "随机种子，-1为随机"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("outfit_tags", "outfit_description")
    FUNCTION = "suggest_outfit"
    CATEGORY = "🐳Pond/text"
    
    def suggest_outfit(self, main_clothing, style, tag_count, include_accessories, seed):
        """根据主要服装推荐搭配"""
        import random
        
        # 设置随机种子
        if seed != -1:
            random.seed(seed)
        
        # 提取英文标签
        if " (" in main_clothing and ") - " in main_clothing:
            en_tag = main_clothing.split(" (")[1].split(")")[0]
            cn_tag = main_clothing.split(" (")[0]
            category = main_clothing.split(" - ")[1]
        else:
            en_tag = ""
            cn_tag = ""
            category = ""
        
        suggestions = []
        
        # 基于规则的搭配
        if en_tag in self.OUTFIT_RULES:
            rule_suggestions = self.OUTFIT_RULES[en_tag].copy()
            random.shuffle(rule_suggestions)
            suggestions.extend(rule_suggestions)
        
        # 基于风格的额外推荐
        if style in self.STYLE_ACCESSORIES and include_accessories == "是":
            style_items = self.STYLE_ACCESSORIES[style].copy()
            random.shuffle(style_items)
            for item in style_items:
                if item not in suggestions:
                    suggestions.append(item)
        
        # 根据服装类别添加通用搭配
        category_suggestions = {
            "连衣裙": ["heels", "sandals", "clutch", "cardigan", "belt"],
            "上衣": ["pants", "skirt", "jeans", "shorts", "blazer"],
            "下装": ["blouse", "t-shirt", "tank top", "crop top", "sweater"],
            "泳装": ["beach bag", "sun hat", "cover-up", "sandals", "sunglasses"],
            "运动装": ["sneakers", "gym bag", "water bottle", "headband", "sports watch"],
            "外套": ["jeans", "dress", "boots", "scarf", "gloves"],
            "内衣": ["robe", "slippers", "pajamas", "silk scarf", "perfume"],
            "特殊服装": ["accessories", "shoes", "bag", "jewelry", "hair accessories"]
        }
        
        if category in category_suggestions:
            cat_items = category_suggestions[category].copy()
            random.shuffle(cat_items)
            for item in cat_items[:3]:  # 只添加前3个
                if item not in suggestions:
                    suggestions.append(item)
        
        # 确保不重复，并限制数量
        unique_suggestions = []
        for item in suggestions:
            if item not in unique_suggestions and item != en_tag:
                unique_suggestions.append(item)
        
        # 构建最终标签列表
        outfit_tags = [en_tag]
        outfit_tags.extend(unique_suggestions[:tag_count-1])  # 减1因为已经包含主要服装
        
        # 生成描述
        style_adjectives = {
            "休闲": "casual and comfortable",
            "正式": "formal and elegant",
            "运动": "sporty and active",
            "派对": "glamorous party",
            "日常": "everyday chic",
            "街头": "urban streetwear",
            "波西米亚": "bohemian free-spirited",
            "复古": "vintage-inspired",
            "极简": "minimalist modern",
            "浪漫": "romantic feminine",
            "朋克": "edgy punk",
            "优雅": "sophisticated elegant",
            "学院": "preppy collegiate",
            "度假": "vacation resort",
            "商务": "professional business",
            "性感": "alluring and sensual"
        }
        
        style_desc = style_adjectives.get(style, style)
        outfit_description = f"A {style_desc} outfit featuring {', '.join(outfit_tags)}"
        
        return (", ".join(outfit_tags), outfit_description)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "ClothingSelector": ClothingSelectorNode,
    "ClothingSelectorSimple": ClothingSelectorSimple,
    "ClothingSelectorBatch": ClothingSelectorBatch,
    "ClothingOutfitSuggestion": ClothingOutfitSuggestion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClothingSelector": "🐳Clothing Selector (Multi-Select)",
    "ClothingSelectorSimple": "🐳Clothing Selector (Number Selection)",
    "ClothingSelectorBatch": "🐳Clothing Random Batch",
    "ClothingOutfitSuggestion": "🐳Clothing Outfit Suggestion"
}