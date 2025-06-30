import json
from typing import Dict, List, Tuple

class ClothingSelectorNode:
    """
    ComfyUI node for selecting multiple clothing tags
    """
    
    # Clothing data - keeping Chinese names for reference
    CLOTHING_DATA = {
        "dresses": {
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
        "tops": {
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
        "bottoms": {
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
        "swimwear": {
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
        "sportswear": {
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
        "underwear": {
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
        "outerwear": {
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
        "special": {
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
        """Define input types"""
        inputs = {
            "required": {
                "separator": (["comma", "space", "newline"], {"default": "comma"}),
            },
            "optional": {}
        }
        
        # Create multiple selection inputs for each category (3 selection boxes per category)
        for category, clothes in cls.CLOTHING_DATA.items():
            # Create options list, format: "Chinese (english)"
            options = ["none"]
            for cn, en in clothes.items():
                options.append(f"{cn} ({en})")
            
            # Create 3 selection boxes for each category
            for i in range(1, 4):  # Create 3 selection boxes
                inputs["optional"][f"select_{category}_{i}"] = (options, {
                    "default": "none",
                    "tooltip": f"Select {category} related clothing tag #{i}"
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
    FUNCTION = "process_clothes"
    CATEGORY = "🐳Pond/text"
    
    def process_clothes(self, separator="comma", custom_tags="", **kwargs):
        """Process selected clothing and return tags"""
        english_tags = []
        chinese_tags = []
        
        # Process selections for each category
        for key, value in kwargs.items():
            if key.startswith("select_") and value and value != "none":
                # Extract category name from key (remove trailing _number)
                key_parts = key.replace("select_", "").rsplit("_", 1)
                category = key_parts[0]
                
                if category in self.CLOTHING_DATA:
                    # value is a string
                    selected = value
                    # Extract from "Chinese (english)" format
                    if " (" in selected and selected.endswith(")"):
                        cn_part = selected.split(" (")[0]
                        # Find corresponding English in original data
                        if cn_part in self.CLOTHING_DATA[category]:
                            en_tag = self.CLOTHING_DATA[category][cn_part]
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
class ClothingSelectorSimple:
    """
    ComfyUI node for selecting clothing tags in a more convenient way
    """
    
    CLOTHING_DATA = ClothingSelectorNode.CLOTHING_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types"""
        
        # Generate numbered lists for each category
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
        
        # Create selection inputs for each category
        for category in cls.CLOTHING_DATA.keys():
            # Display available clothing list
            inputs["optional"][f"{category}_list"] = ("STRING", {
                "default": clothing_lists[category],
                "multiline": True,
                "dynamicPrompts": False,
                "tooltip": f"All available clothing in {category} category"
            })
            
            # Input for selected numbers
            inputs["optional"][f"{category}_selection"] = ("STRING", {
                "default": "",
                "placeholder": "Enter numbers, e.g.: 1,3,5 or 1-5,8,10",
                "tooltip": f"Enter numbers of {category} clothing to select"
            })
        
        # Quick presets
        inputs["optional"]["quick_preset"] = (["none", "casual", "formal", "sports", "swimwear", "summer", "party"], {
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
    FUNCTION = "process_clothes"
    CATEGORY = "🐳Pond/text"
    
    # Preset definitions
    PRESETS = {
        "casual": {"tops": [6, 10], "bottoms": [7, 11], "outerwear": [1]},  # T-shirt, hoodie, denim shorts, shorts, denim jacket
        "formal": {"dresses": [3, 16], "tops": [7], "outerwear": [4]},  # LBD, evening gown, chiffon blouse, blazer
        "sports": {"sportswear": [1, 2, 3], "tops": [11]},  # Sports bra, yoga pants, athletic shorts, sports bra
        "swimwear": {"swimwear": [1, 4, 13], "outerwear": [16]},  # Triangle bikini, halter bikini, rash guard, shawl
        "summer": {"dresses": [7, 15], "tops": [3, 5], "bottoms": [1]},  # Sundress, floral midi, crop top, camisole, mini skirt
        "party": {"dresses": [14, 16], "tops": [19], "bottoms": [15]}  # Sequin dress, evening gown, sequin top, tulle skirt
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
    
    def process_clothes(self, separator="comma", output_format="english", quick_preset="none", custom_tags="", **kwargs):
        """Process selected clothing and return tags"""
        selected_tags = []
        
        # Process presets
        if quick_preset != "none" and quick_preset in self.PRESETS:
            preset = self.PRESETS[quick_preset]
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
        
        # Process selections for each category
        for category in self.CLOTHING_DATA.keys():
            selection_key = f"{category}_selection"
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


class ClothingSelectorBatch:
    """
    Batch clothing generator - generates multiple clothing combinations
    """
    
    CLOTHING_DATA = ClothingSelectorNode.CLOTHING_DATA
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 3, "min": 1, "max": 10}),
                "tags_per_batch": ("INT", {"default": 3, "min": 1, "max": 10}),
                "category_weights": ("STRING", {
                    "default": "dresses:0.2, tops:0.2, bottoms:0.2, swimwear:0.1, sportswear:0.1, underwear:0.1, outerwear:0.05, special:0.05",
                    "placeholder": "category:weight, e.g. dresses:0.3 (weights should sum to 1)"
                }),
                "style_preset": (["random", "casual", "formal", "sports", "sexy", "daily"], {"default": "random"}),
                "ensure_tags": ("STRING", {
                    "default": "",
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
    
    # Style weight presets
    STYLE_WEIGHTS = {
        "casual": {"dresses": 0.1, "tops": 0.3, "bottoms": 0.3, "swimwear": 0.05, "sportswear": 0.1, "underwear": 0.05, "outerwear": 0.1, "special": 0.0},
        "formal": {"dresses": 0.4, "tops": 0.2, "bottoms": 0.2, "swimwear": 0.0, "sportswear": 0.0, "underwear": 0.0, "outerwear": 0.15, "special": 0.05},
        "sports": {"dresses": 0.0, "tops": 0.1, "bottoms": 0.1, "swimwear": 0.1, "sportswear": 0.6, "underwear": 0.05, "outerwear": 0.05, "special": 0.0},
        "sexy": {"dresses": 0.2, "tops": 0.2, "bottoms": 0.15, "swimwear": 0.2, "sportswear": 0.0, "underwear": 0.2, "outerwear": 0.0, "special": 0.05},
        "daily": {"dresses": 0.15, "tops": 0.25, "bottoms": 0.25, "swimwear": 0.0, "sportswear": 0.1, "underwear": 0.05, "outerwear": 0.15, "special": 0.05}
    }
    
    def generate_batches(self, batch_count, tags_per_batch, category_weights, style_preset, ensure_tags, seed):
        """Generate multiple random clothing combinations"""
        import random
        
        if seed != -1:
            random.seed(seed)
        
        # Parse weights or use preset
        if style_preset != "random" and style_preset in self.STYLE_WEIGHTS:
            weights = self.STYLE_WEIGHTS[style_preset]
        else:
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
                
                if category in self.CLOTHING_DATA:
                    clothes = list(self.CLOTHING_DATA[category].values())
                    if clothes:
                        tag = random.choice(clothes)
                        if tag not in selected:
                            selected.append(tag)
            
            batches.append(", ".join(selected))
        
        # Pad to 10 outputs
        while len(batches) < 10:
            batches.append("")
        
        return tuple(batches)


# Clothing outfit suggestion node
class ClothingOutfitSuggestion:
    """
    Recommend outfits based on selected main clothing
    """
    
    CLOTHING_DATA = ClothingSelectorNode.CLOTHING_DATA
    
    # Extended outfit rules
    OUTFIT_RULES = {
        # Dress category
        "wrap dress": ["denim jacket", "ankle boots", "crossbody bag", "belt", "cardigan"],
        "midi dress": ["blazer", "heels", "clutch", "statement necklace", "belt"],
        "maxi dress": ["sandals", "sun hat", "tote bag", "denim jacket", "wedges"],
        "cocktail dress": ["high heels", "clutch", "statement jewelry", "wrap", "evening bag"],
        "sundress": ["sandals", "straw hat", "canvas bag", "cardigan", "espadrilles"],
        "bodycon dress": ["stiletto heels", "clutch", "statement earrings", "choker", "ankle strap heels"],
        "slip dress": ["strappy heels", "delicate jewelry", "clutch", "shawl", "thigh-high boots"],
        
        # Top category
        "t-shirt": ["jeans", "sneakers", "baseball cap", "backpack", "bomber jacket"],
        "blouse": ["pencil skirt", "heels", "blazer", "tote bag", "pearl necklace"],
        "crop top": ["high-waisted pants", "sneakers", "choker", "denim jacket", "mini backpack"],
        "tank top": ["shorts", "sandals", "sunglasses", "crossbody bag", "kimono"],
        "hoodie": ["joggers", "sneakers", "beanie", "backpack", "windbreaker"],
        "halter top": ["high-waisted skirt", "heels", "statement earrings", "clutch", "body chain"],
        "lace top": ["leather pants", "stilettos", "clutch", "red lipstick", "statement necklace"],
        
        # Bottom category
        "mini skirt": ["crop top", "ankle boots", "bomber jacket", "choker", "crossbody bag"],
        "high-waisted pants": ["tucked-in blouse", "belt", "heels", "blazer", "structured bag"],
        "jeans": ["t-shirt", "sneakers", "denim jacket", "belt", "casual bag"],
        "shorts": ["tank top", "sandals", "sun hat", "beach bag", "kimono"],
        "leather skirt": ["silk blouse", "heels", "clutch", "statement jewelry", "leather jacket"],
        "pencil skirt": ["fitted blouse", "pumps", "structured bag", "belt", "blazer"],
        
        # Sportswear category
        "sports bra": ["yoga pants", "athletic shoes", "gym bag", "water bottle", "headband"],
        "yoga pants": ["sports bra", "tank top", "yoga mat", "sneakers", "hoodie"],
        
        # Swimwear category
        "bikini": ["beach cover-up", "sun hat", "sandals", "beach bag", "sunglasses"],
        "one-piece swimsuit": ["sarong", "flip-flops", "sun hat", "beach tote", "kimono"],
        
        # Formal category
        "evening gown": ["clutch", "heels", "statement jewelry", "wrap", "evening gloves"],
        "blazer": ["pencil skirt", "blouse", "pumps", "structured bag", "watch"]
    }
    
    # Extended style accessories
    STYLE_ACCESSORIES = {
        "casual": ["sneakers", "backpack", "baseball cap", "crossbody bag", "sunglasses", "canvas tote"],
        "formal": ["heels", "clutch", "blazer", "pearl necklace", "structured bag", "silk scarf"],
        "sports": ["athletic shoes", "gym bag", "headband", "sports watch", "water bottle", "windbreaker"],
        "party": ["high heels", "statement jewelry", "evening bag", "bold lipstick", "cocktail ring", "wrap"],
        "daily": ["comfortable shoes", "tote bag", "sunglasses", "watch", "crossbody bag", "cardigan"],
        "street": ["sneakers", "bucket hat", "chain necklace", "mini backpack", "oversized jacket", "socks"],
        "boho": ["sandals", "fringe bag", "headband", "layered necklaces", "kimono", "anklet"],
        "vintage": ["vintage bag", "cat-eye sunglasses", "headscarf", "brooch", "mary jane shoes", "gloves"],
        "minimalist": ["minimalist bag", "simple jewelry", "loafers", "structured coat", "monochrome scarf", "watch"],
        "romantic": ["ballet flats", "pearl accessories", "hair bow", "lace gloves", "clutch", "shawl"],
        "punk": ["combat boots", "leather jacket", "studded bag", "choker", "chain belt", "fingerless gloves"],
        "elegant": ["kitten heels", "silk scarf", "pearl earrings", "structured handbag", "gloves", "brooch"],
        "preppy": ["loafers", "messenger bag", "preppy blazer", "knee socks", "headband", "plaid scarf"],
        "vacation": ["espadrilles", "straw bag", "sun hat", "oversized sunglasses", "beach cover-up", "anklet"],
        "business": ["pumps", "laptop bag", "blazer", "silk blouse", "watch", "structured tote"],
        "sexy": ["stiletto heels", "body chain", "choker necklace", "thigh-high boots", "statement earrings", "red lipstick"]
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Create main clothing options
        all_clothes = []
        for category, clothes in cls.CLOTHING_DATA.items():
            for cn, en in clothes.items():
                all_clothes.append(f"{cn} ({en}) - {category}")
        
        return {
            "required": {
                "main_clothing": (all_clothes, {
                    "default": all_clothes[0] if all_clothes else "none"
                }),
                "style": ([
                    "casual", "formal", "sports", "party", "daily",
                    "street", "boho", "vintage", "minimalist", "romantic",
                    "punk", "elegant", "preppy", "vacation", "business", "sexy"
                ], {"default": "daily"}),
                "tag_count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "display": "slider"
                }),
                "include_accessories": (["yes", "no"], {"default": "yes"}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "Random seed, -1 for random"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("outfit_tags", "outfit_description")
    FUNCTION = "suggest_outfit"
    CATEGORY = "🐳Pond/text"
    
    def suggest_outfit(self, main_clothing, style, tag_count, include_accessories, seed):
        """Suggest outfit based on main clothing"""
        import random
        
        # Set random seed
        if seed != -1:
            random.seed(seed)
        
        # Extract English tag
        if " (" in main_clothing and ") - " in main_clothing:
            en_tag = main_clothing.split(" (")[1].split(")")[0]
            cn_tag = main_clothing.split(" (")[0]
            category = main_clothing.split(" - ")[1]
        else:
            en_tag = ""
            cn_tag = ""
            category = ""
        
        suggestions = []
        
        # Rule-based matching
        if en_tag in self.OUTFIT_RULES:
            rule_suggestions = self.OUTFIT_RULES[en_tag].copy()
            random.shuffle(rule_suggestions)
            suggestions.extend(rule_suggestions)
        
        # Style-based additional recommendations
        if style in self.STYLE_ACCESSORIES and include_accessories == "yes":
            style_items = self.STYLE_ACCESSORIES[style].copy()
            random.shuffle(style_items)
            for item in style_items:
                if item not in suggestions:
                    suggestions.append(item)
        
        # Add general matching based on clothing category
        category_suggestions = {
            "dresses": ["heels", "sandals", "clutch", "cardigan", "belt"],
            "tops": ["pants", "skirt", "jeans", "shorts", "blazer"],
            "bottoms": ["blouse", "t-shirt", "tank top", "crop top", "sweater"],
            "swimwear": ["beach bag", "sun hat", "cover-up", "sandals", "sunglasses"],
            "sportswear": ["sneakers", "gym bag", "water bottle", "headband", "sports watch"],
            "outerwear": ["jeans", "dress", "boots", "scarf", "gloves"],
            "underwear": ["robe", "slippers", "pajamas", "silk scarf", "perfume"],
            "special": ["accessories", "shoes", "bag", "jewelry", "hair accessories"]
        }
        
        if category in category_suggestions:
            cat_items = category_suggestions[category].copy()
            random.shuffle(cat_items)
            for item in cat_items[:3]:  # Only add first 3
                if item not in suggestions:
                    suggestions.append(item)
        
        # Ensure no duplicates and limit quantity
        unique_suggestions = []
        for item in suggestions:
            if item not in unique_suggestions and item != en_tag:
                unique_suggestions.append(item)
        
        # Build final tag list
        outfit_tags = [en_tag]
        outfit_tags.extend(unique_suggestions[:tag_count-1])  # Minus 1 because main clothing is already included
        
        # Generate description
        style_adjectives = {
            "casual": "casual and comfortable",
            "formal": "formal and elegant",
            "sports": "sporty and active",
            "party": "glamorous party",
            "daily": "everyday chic",
            "street": "urban streetwear",
            "boho": "bohemian free-spirited",
            "vintage": "vintage-inspired",
            "minimalist": "minimalist modern",
            "romantic": "romantic feminine",
            "punk": "edgy punk",
            "elegant": "sophisticated elegant",
            "preppy": "preppy collegiate",
            "vacation": "vacation resort",
            "business": "professional business",
            "sexy": "alluring and sensual"
        }
        
        style_desc = style_adjectives.get(style, style)
        outfit_description = f"A {style_desc} outfit featuring {', '.join(outfit_tags)}"
        
        return (", ".join(outfit_tags), outfit_description)


# Node registration
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