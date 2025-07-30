class VideoPromptNode:

    # Complete prompt data from PDF
    PROMPT_DATA = {
        "光源类型": {
            "日光": {"zh": "日光", "en": "daylight"},
            "人工光": {"zh": "人工光", "en": "artificial light"},
            "月光": {"zh": "月光", "en": "moonlight"},
            "实用光": {"zh": "实用光", "en": "practical light"},
            "火光": {"zh": "火光", "en": "firelight"},
            "荧光": {"zh": "荧光", "en": "fluorescent light"}
        },
        "光线类型": {
            "柔光": {"zh": "柔光", "en": "soft light"},
            "硬光": {"zh": "硬光", "en": "hard light"},
            "顶光": {"zh": "顶光", "en": "top light"},
            "侧光": {"zh": "侧光", "en": "side light"},
            "背光": {"zh": "背光", "en": "backlight"},
            "底光": {"zh": "底光", "en": "bottom light"},
            "边缘光": {"zh": "边缘光", "en": "rim light"},
            "剪影": {"zh": "剪影", "en": "silhouette"},
            "低对比度": {"zh": "低对比度", "en": "low contrast"},
            "高对比度": {"zh": "高对比度", "en": "high contrast"},
            "阴天光": {"zh": "阴天光", "en": "overcast light"},
            "混合光": {"zh": "混合光", "en": "mixed light"},
            "晴天光": {"zh": "晴天光", "en": "sunny light"}
        },
        "时间段": {
            "白天": {"zh": "白天", "en": "daytime"},
            "夜晚": {"zh": "夜晚", "en": "night"},
            "黄昏": {"zh": "黄昏", "en": "dusk"},
            "日落": {"zh": "日落", "en": "sunset"},
            "日出": {"zh": "日出", "en": "sunrise"},
            "黎明": {"zh": "黎明", "en": "dawn"}
        },
        "景别": {
            "特写": {"zh": "特写", "en": "extreme close-up"},
            "近景": {"zh": "近景", "en": "close-up"},
            "中景": {"zh": "中景", "en": "medium shot"},
            "中近景": {"zh": "中近景", "en": "medium close-up"},
            "中全景": {"zh": "中全景", "en": "medium full shot"},
            "全景": {"zh": "全景", "en": "full shot"},
            "远景": {"zh": "远景", "en": "long shot"},
            "广角": {"zh": "广角", "en": "wide angle"}
        },
        "构图": {
            "中心构图": {"zh": "中心构图", "en": "center composition"},
            "平衡构图": {"zh": "平衡构图", "en": "balanced composition"},
            "右侧重构图": {"zh": "右侧重构图", "en": "right-weighted composition"},
            "左侧重构图": {"zh": "左侧重构图", "en": "left-weighted composition"},
            "对称构图": {"zh": "对称构图", "en": "symmetrical composition"},
            "短边构图": {"zh": "短边构图", "en": "short side composition"}
        },
        "镜头焦段": {
            "中焦距": {"zh": "中焦距", "en": "medium focal length"},
            "广角": {"zh": "广角", "en": "wide angle"},
            "长焦": {"zh": "长焦", "en": "telephoto"},
            "望远": {"zh": "望远", "en": "telescope"},
            "超广角-鱼眼": {"zh": "超广角-鱼眼", "en": "ultra wide angle fisheye"}
        },
        "机位角度": {
            "平拍": {"zh": "平拍", "en": "eye level"},
            "过肩镜头角度拍摄": {"zh": "过肩镜头角度拍摄", "en": "over shoulder shot"},
            "高角度拍摄": {"zh": "高角度拍摄", "en": "high angle"},
            "低角度拍摄": {"zh": "低角度拍摄", "en": "low angle"},
            "倾斜角度": {"zh": "倾斜角度", "en": "dutch angle"},
            "航拍": {"zh": "航拍", "en": "aerial shot"},
            "俯视角度拍摄": {"zh": "俯视角度拍摄", "en": "bird's eye view"}
        },
        "镜头类型": {
            "干净的单人镜头": {"zh": "干净的单人镜头", "en": "clean single shot"},
            "双人镜头": {"zh": "双人镜头", "en": "two shot"},
            "三人镜头": {"zh": "三人镜头", "en": "three shot"},
            "群像镜头": {"zh": "群像镜头", "en": "group shot"},
            "定场镜头": {"zh": "定场镜头", "en": "establishing shot"}
        },
        "色调": {
            "暖色调": {"zh": "暖色调", "en": "warm tone"},
            "冷色调": {"zh": "冷色调", "en": "cool tone"},
            "高饱和度": {"zh": "高饱和度", "en": "high saturation"},
            "低饱和度": {"zh": "低饱和度", "en": "low saturation"},
            "混合色调": {"zh": "混合色调", "en": "mixed tones"}
        },
        "基础运镜": {
            "镜头推进": {"zh": "镜头推进", "en": "push in"},
            "镜头拉远": {"zh": "镜头拉远", "en": "pull out"},
            "镜头向右移动": {"zh": "镜头向右移动", "en": "pan right"},
            "镜头向左移动": {"zh": "镜头向左移动", "en": "pan left"},
            "镜头上摇": {"zh": "镜头上摇", "en": "tilt up"},
            "镜头下摇": {"zh": "镜头下摇", "en": "tilt down"}
        },
        "高级运镜": {
            "手持镜头": {"zh": "手持镜头", "en": "handheld"},
            "跟随镜头": {"zh": "跟随镜头", "en": "tracking shot"},
            "环绕运镜": {"zh": "环绕运镜", "en": "circular tracking"},
            "复合运镜": {"zh": "复合运镜", "en": "complex camera movement"}
        },
        "人物情绪": {
            "愤怒": {"zh": "愤怒", "en": "angry"},
            "恐惧": {"zh": "恐惧", "en": "fearful"},
            "高兴": {"zh": "高兴", "en": "happy"},
            "悲伤": {"zh": "悲伤", "en": "sad"},
            "惊讶": {"zh": "惊讶", "en": "surprised"}
        },
        "运动类型": {
            "跑步": {"zh": "跑步", "en": "running"},
            "滑滑板": {"zh": "滑滑板", "en": "skateboarding"},
            "踢足球": {"zh": "踢足球", "en": "playing football"},
            "网球": {"zh": "网球", "en": "playing tennis"},
            "篮球": {"zh": "篮球", "en": "playing basketball"},
            "橄榄球": {"zh": "橄榄球", "en": "playing rugby"},
            "顶碗舞": {"zh": "顶碗舞", "en": "bowl dance"},
            "侧手翻": {"zh": "侧手翻", "en": "cartwheel"}
        },
        "视觉风格": {
            "毛毡风格": {"zh": "毛毡风格", "en": "felt style"},
            "3D卡通": {"zh": "3D卡通风格", "en": "3D cartoon style"},
            "像素风格": {"zh": "像素风格", "en": "pixel art style"},
            "木偶动画": {"zh": "木偶动画", "en": "puppet animation"},
            "3D游戏": {"zh": "3D游戏", "en": "3D game style"},
            "黏土风格": {"zh": "黏土风格", "en": "claymation style"},
            "二次元": {"zh": "二次元动画风格", "en": "anime style"},
            "水彩画": {"zh": "水彩画", "en": "watercolor style"},
            "黑白动画": {"zh": "黑白动画", "en": "black and white animation"},
            "油画风格": {"zh": "油画风格", "en": "oil painting style"}
        },
        "特效镜头": {
            "移轴摄影": {"zh": "移轴摄影", "en": "tilt-shift photography"},
            "延时拍摄": {"zh": "延时拍摄", "en": "time-lapse"}
        }
    }
    
    # Pre-defined prompt templates
    TEMPLATES = {
        "电影级画面": {
            "zh": "黄昏，柔光，侧光，边缘光，中景，中心构图，暖色调，低饱和度，干净的单人镜头",
            "en": "dusk, soft light, side light, rim light, medium shot, center composition, warm tone, low saturation, clean single shot"
        },
        "纪录片风格": {
            "zh": "日光，自然光，平拍，中景，手持镜头，跟随镜头，写实风格",
            "en": "daylight, natural light, eye level, medium shot, handheld, tracking shot, realistic style"
        },
        "动作场景": {
            "zh": "硬光，高对比度，低角度拍摄，快速剪辑，动态模糊，运动镜头",
            "en": "hard light, high contrast, low angle, fast cuts, motion blur, dynamic camera movement"
        },
        "浪漫场景": {
            "zh": "黄昏，柔光，背光，近景，暖色调，高饱和度，浅景深",
            "en": "dusk, soft light, backlight, close-up, warm tone, high saturation, shallow depth of field"
        },
        "恐怖氛围": {
            "zh": "夜晚，底光，硬光，倾斜角度，冷色调，低饱和度，手持镜头",
            "en": "night, bottom light, hard light, dutch angle, cool tone, low saturation, handheld"
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Create selection lists for each category
        inputs = {
            "required": {
                "mode": (["custom", "template"], {"default": "custom"}),
                "output_language": (["中文", "English"], {"default": "中文"}),
                "combine_mode": (["逗号分隔", "空格分隔", "句子形式"], {"default": "逗号分隔"}),
            },
            "optional": {
                # Template selection
                "template": (list(cls.TEMPLATES.keys()), {"default": "电影级画面"}),
                
                # All categories as optional inputs
                "光源类型": (["none"] + list(cls.PROMPT_DATA["光源类型"].keys()), {"default": "none"}),
                "光线类型": (["none"] + list(cls.PROMPT_DATA["光线类型"].keys()), {"default": "none"}),
                "时间段": (["none"] + list(cls.PROMPT_DATA["时间段"].keys()), {"default": "none"}),
                "景别": (["none"] + list(cls.PROMPT_DATA["景别"].keys()), {"default": "none"}),
                "构图": (["none"] + list(cls.PROMPT_DATA["构图"].keys()), {"default": "none"}),
                "镜头焦段": (["none"] + list(cls.PROMPT_DATA["镜头焦段"].keys()), {"default": "none"}),
                "机位角度": (["none"] + list(cls.PROMPT_DATA["机位角度"].keys()), {"default": "none"}),
                "镜头类型": (["none"] + list(cls.PROMPT_DATA["镜头类型"].keys()), {"default": "none"}),
                "色调": (["none"] + list(cls.PROMPT_DATA["色调"].keys()), {"default": "none"}),
                "基础运镜": (["none"] + list(cls.PROMPT_DATA["基础运镜"].keys()), {"default": "none"}),
                "高级运镜": (["none"] + list(cls.PROMPT_DATA["高级运镜"].keys()), {"default": "none"}),
                "人物情绪": (["none"] + list(cls.PROMPT_DATA["人物情绪"].keys()), {"default": "none"}),
                "运动类型": (["none"] + list(cls.PROMPT_DATA["运动类型"].keys()), {"default": "none"}),
                "视觉风格": (["none"] + list(cls.PROMPT_DATA["视觉风格"].keys()), {"default": "none"}),
                "特效镜头": (["none"] + list(cls.PROMPT_DATA["特效镜头"].keys()), {"default": "none"}),
                
                # Custom prompt
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "additional_template_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
        
        return inputs
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "🐳Pond/video"
    
    def generate_prompt(self, mode="custom", output_language="中文", combine_mode="逗号分隔", 
                       template="电影级画面", custom_prompt="", additional_template_prompt="", **kwargs):
        """Generate video prompt based on selections"""
        
        # Determine language key
        lang_key = "zh" if output_language == "中文" else "en"
        
        if mode == "template":
            # Use template mode
            base_prompt = self.TEMPLATES[template][lang_key]
            
            if additional_template_prompt.strip():
                separator = "，" if output_language == "中文" else ", "
                final_prompt = base_prompt + separator + additional_template_prompt.strip()
            else:
                final_prompt = base_prompt
                
        else:  # custom mode
            # Collect selected prompts
            prompt_parts = []
            
            # Process each category
            for category, value in kwargs.items():
                if category in self.PROMPT_DATA and value != "none":
                    if value in self.PROMPT_DATA[category]:
                        prompt_parts.append(self.PROMPT_DATA[category][value][lang_key])
            
            # Add custom prompt if provided
            if custom_prompt.strip():
                prompt_parts.append(custom_prompt.strip())
            
            # Combine prompts based on mode
            if not prompt_parts:
                final_prompt = ""
            elif combine_mode == "逗号分隔":
                separator = "，" if output_language == "中文" else ", "
                final_prompt = separator.join(prompt_parts)
            elif combine_mode == "空格分隔":
                final_prompt = " ".join(prompt_parts)
            else:  # 句子形式
                if output_language == "中文":
                    if len(prompt_parts) > 3:
                        final_prompt = "画面采用" + "、".join(prompt_parts[:3])
                        final_prompt += "，" + "，".join(prompt_parts[3:])
                    else:
                        final_prompt = "画面采用" + "、".join(prompt_parts)
                    final_prompt += "。"
                else:
                    if len(prompt_parts) > 3:
                        final_prompt = "The scene uses " + ", ".join(prompt_parts[:3])
                        final_prompt += ", with " + ", ".join(prompt_parts[3:])
                    else:
                        final_prompt = "The scene uses " + ", ".join(prompt_parts)
                    final_prompt += "."
        
        return (final_prompt,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "VideoPromptNode": VideoPromptNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPromptNode": "🐳Wan2.2_Prompt"
}