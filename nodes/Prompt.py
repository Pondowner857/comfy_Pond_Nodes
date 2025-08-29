class PhotographyParametersNode:
    """
    A photography parameters node with single text input/output but all parameter options
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Input your base prompt here / 在此输入基础提示词"
                }),
                
                "output_language": (["English", "中文", "Both/双语"],),
                
                "lighting_type": ([
                    "None/无",
                    "Golden Hour/黄金时刻",
                    "Blue Hour/蓝色时刻",
                    "Hard Light/硬光",
                    "Soft Light/柔光",
                    "Diffused Light/漫射光",
                    "Rim Light/轮廓光",
                    "Back Light/逆光",
                    "Side Light/侧光",
                    "Front Light/顺光",
                    "Top Light/顶光",
                    "Bottom Light/底光",
                    "Rembrandt Light/伦勃朗光",
                    "Split Light/分割光",
                    "Butterfly Light/蝴蝶光",
                    "Loop Light/环形光",
                    "Broad Light/宽光",
                    "Short Light/窄光",
                    "Natural Light/自然光",
                    "Ambient Light/环境光",
                    "Candlelight/烛光",
                    "Firelight/火光",
                    "Moonlight/月光",
                    "Starlight/星光",
                    "Neon Light/霓虹灯光",
                    "LED Light/LED灯光",
                    "Fluorescent Light/荧光灯",
                    "Incandescent Light/白炽灯",
                    "Studio Light/影棚灯光",
                    "Three-Point Lighting/三点照明",
                    "High Key Lighting/高调照明",
                    "Low Key Lighting/低调照明",
                    "Chiaroscuro/明暗对照法",
                    "Tenebrism/暗色调主义",
                    "Volumetric Light/体积光",
                    "God Rays/丁达尔效应",
                    "Caustics/焦散",
                    "Global Illumination/全局照明",
                    "Radiosity/辐射度",
                    "Subsurface Scattering/次表面散射",
                    "Bounce Light/反射光",
                    "Fill Light/补光",
                    "Key Light/主光",
                    "Kicker Light/踢脚光",
                    "Practical Light/实景光",
                    "Motivated Light/动机光",
                    "Available Light/现场光",
                    "Morning Light/晨光",
                    "Afternoon Light/午后光",
                    "Twilight/暮光",
                    "Dusk Light/黄昏光",
                    "Dawn Light/黎明光",
                ],),
                
                "camera_angle": ([
                    "None/无",
                    "Eye Level/平视",
                    "High Angle/俯拍",
                    "Low Angle/仰拍",
                    "Bird's Eye View/鸟瞰",
                    "Worm's Eye View/虫视",
                    "Dutch Angle/荷兰角",
                    "Overhead Shot/顶部俯拍",
                    "Ground Level/地面水平",
                    "Aerial View/航拍视角",
                    "Drone Shot/无人机视角",
                    "Three-Quarter View/四分之三视角",
                    "Profile View/侧面视角",
                    "Front View/正面视角",
                    "Back View/背面视角",
                    "Over-the-Shoulder/过肩视角",
                    "Point-of-View (POV)/第一人称视角",
                    "Establishing Shot/定场镜头",
                    "Wide Shot/远景",
                    "Full Shot/全景",
                    "Medium Shot/中景",
                    "Medium Close-Up/中近景",
                    "Close-Up/近景",
                    "Extreme Close-Up/特写",
                    "Macro Shot/微距",
                    "Two-Shot/双人镜头",
                    "Group Shot/群像镜头",
                    "Tracking Shot/跟踪镜头",
                    "Dolly Shot/推拉镜头",
                    "Crane Shot/摇臂镜头",
                    "Steadicam Shot/斯坦尼康镜头",
                    "Handheld Shot/手持镜头",
                    "Static Shot/固定镜头",
                    "Pan Shot/横摇镜头",
                    "Tilt Shot/纵摇镜头",
                    "Zoom Shot/变焦镜头",
                    "Rack Focus/焦点转换",
                    "Deep Focus/深焦",
                    "Shallow Focus/浅焦",
                    "Split Diopter/分屈光镜",
                    "Tilt-Shift/移轴",
                    "360 Degree/360度全景",
                    "Vertical Shot/垂直拍摄",
                    "Diagonal Shot/对角线拍摄",
                    "Centered Composition/中心构图",
                    "Rule of Thirds/三分法构图",
                    "Golden Ratio/黄金比例构图",
                    "Symmetrical/对称构图",
                    "Asymmetrical/非对称构图",
                ],),
                
                "aesthetic_style": ([
                    "None/无",
                    "Minimalist/极简主义",
                    "Maximalist/极繁主义",
                    "Brutalist/粗野主义",
                    "Art Nouveau/新艺术运动",
                    "Art Deco/装饰艺术",
                    "Bauhaus/包豪斯",
                    "Victorian/维多利亚风格",
                    "Gothic/哥特式",
                    "Baroque/巴洛克",
                    "Rococo/洛可可",
                    "Renaissance/文艺复兴",
                    "Neoclassical/新古典主义",
                    "Romantic/浪漫主义",
                    "Impressionist/印象派",
                    "Post-Impressionist/后印象派",
                    "Expressionist/表现主义",
                    "Abstract Expressionist/抽象表现主义",
                    "Surrealist/超现实主义",
                    "Dadaist/达达主义",
                    "Cubist/立体主义",
                    "Fauvist/野兽派",
                    "Futurist/未来主义",
                    "Constructivist/构成主义",
                    "Pop Art/波普艺术",
                    "Op Art/欧普艺术",
                    "Kinetic Art/动态艺术",
                    "Conceptual/观念艺术",
                    "Photorealism/照相写实主义",
                    "Hyperrealism/超写实主义",
                    "Vaporwave/蒸汽波",
                    "Synthwave/合成波",
                    "Retrowave/复古波",
                    "Cyberpunk/赛博朋克",
                    "Steampunk/蒸汽朋克",
                    "Dieselpunk/柴油朋克",
                    "Biopunk/生物朋克",
                    "Solarpunk/太阳朋克",
                    "Cottagecore/田园风",
                    "Dark Academia/暗黑学院风",
                    "Light Academia/明亮学院风",
                    "Grunge/垃圾摇滚风",
                    "Y2K/千禧风",
                    "Memphis Design/孟菲斯设计",
                    "Scandinavian/斯堪的纳维亚",
                    "Japandi/日式北欧风",
                    "Wabi-Sabi/侘寂",
                    "Industrial/工业风",
                    "Bohemian/波西米亚",
                    "Mid-Century Modern/世纪中期现代",
                    "Contemporary/当代",
                    "Traditional/传统",
                    "Rustic/乡村风",
                    "Urban/都市风",
                    "Eclectic/折衷主义",
                    "Vintage/复古",
                    "Modern/现代",
                    "Postmodern/后现代",
                    "Deconstructivist/解构主义",
                    "Organic/有机风格",
                    "Geometric/几何风格",
                    "Fluid/流体风格",
                    "Glitch Art/故障艺术",
                    "Pixel Art/像素艺术",
                    "Low Poly/低多边形",
                    "Isometric/等距视角",
                ],),
                
                "emotional_mood": ([
                    "None/无",
                    "Joyful/欢乐的",
                    "Ecstatic/狂喜的",
                    "Euphoric/陶醉的",
                    "Cheerful/愉快的",
                    "Content/满足的",
                    "Serene/宁静的",
                    "Peaceful/平和的",
                    "Tranquil/安详的",
                    "Calm/平静的",
                    "Meditative/冥想的",
                    "Melancholic/忧郁的",
                    "Nostalgic/怀旧的",
                    "Wistful/渴望的",
                    "Pensive/沉思的",
                    "Contemplative/冥想的",
                    "Mysterious/神秘的",
                    "Enigmatic/谜一般的",
                    "Ominous/不祥的",
                    "Eerie/诡异的",
                    "Tense/紧张的",
                    "Anxious/焦虑的",
                    "Suspenseful/悬疑的",
                    "Dramatic/戏剧性的",
                    "Intense/强烈的",
                    "Passionate/热情的",
                    "Romantic/浪漫的",
                    "Intimate/亲密的",
                    "Tender/温柔的",
                    "Vulnerable/脆弱的",
                    "Powerful/有力的",
                    "Dominant/主导的",
                    "Aggressive/激进的",
                    "Fierce/凶猛的",
                    "Bold/大胆的",
                    "Confident/自信的",
                    "Proud/骄傲的",
                    "Triumphant/胜利的",
                    "Hopeful/充满希望的",
                    "Optimistic/乐观的",
                    "Pessimistic/悲观的",
                    "Cynical/愤世嫉俗的",
                    "Whimsical/异想天开的",
                    "Playful/俏皮的",
                    "Surreal/超现实的",
                    "Dreamlike/梦幻的",
                    "Ethereal/空灵的",
                    "Spiritual/精神的",
                    "Sublime/崇高的",
                    "Majestic/庄严的",
                    "Epic/史诗般的",
                    "Heroic/英雄的",
                    "Tragic/悲剧的",
                    "Comic/喜剧的",
                    "Absurd/荒诞的",
                    "Chaotic/混乱的",
                    "Harmonious/和谐的",
                    "Dynamic/动态的",
                    "Energetic/充满活力的",
                    "Vibrant/充满生机的",
                    "Muted/柔和的",
                ],),
                
                "camera_model": ([
                    "None/无",
                    "Canon EOS R5/佳能EOS R5",
                    "Canon EOS R3/佳能EOS R3",
                    "Canon EOS 1D X Mark III/佳能1DX3",
                    "Canon EOS 5D Mark IV/佳能5D4",
                    "Nikon Z9/尼康Z9",
                    "Nikon Z7 II/尼康Z7II",
                    "Nikon D850/尼康D850",
                    "Nikon D6/尼康D6",
                    "Sony α1/索尼A1",
                    "Sony α7R V/索尼A7R5",
                    "Sony α7S III/索尼A7S3",
                    "Sony α9 II/索尼A9II",
                    "Fujifilm GFX100 II/富士GFX100II",
                    "Fujifilm GFX100S/富士GFX100S",
                    "Fujifilm X-H2S/富士X-H2S",
                    "Fujifilm X-T5/富士X-T5",
                    "Hasselblad X2D 100C/哈苏X2D",
                    "Hasselblad 907X/哈苏907X",
                    "Hasselblad H6D-100c/哈苏H6D",
                    "Phase One XF IQ4/飞思XF IQ4",
                    "Leica M11/徕卡M11",
                    "Leica SL2-S/徕卡SL2-S",
                    "Leica Q3/徕卡Q3",
                    "Pentax K-3 III/宾得K3III",
                    "Pentax 645Z/宾得645Z",
                    "Olympus OM-1/奥林巴斯OM1",
                    "Panasonic Lumix S1R/松下S1R",
                    "Panasonic Lumix GH6/松下GH6",
                    "Blackmagic URSA Mini Pro 12K/黑魔法12K",
                    "RED Komodo 6K/RED科莫多",
                    "RED V-Raptor 8K/RED猛禽",
                    "ARRI Alexa 65/阿莱65",
                    "ARRI Alexa Mini LF/阿莱Mini LF",
                    "ARRI Alexa 35/阿莱35",
                    "Sony FX9/索尼FX9",
                    "Sony FX6/索尼FX6",
                    "Canon C500 Mark II/佳能C500II",
                    "Canon C300 Mark III/佳能C300III",
                    "Film - 35mm/胶片35毫米",
                    "Film - Medium Format/中画幅胶片",
                    "Film - Large Format/大画幅胶片",
                    "Film - IMAX/IMAX胶片",
                    "Polaroid/宝丽来",
                    "Vintage Film Camera/复古胶片相机",
                    "Pinhole Camera/针孔相机",
                    "Smartphone Camera/智能手机摄像头",
                    "GoPro HERO12/GoPro运动相机",
                    "360° Camera/360度全景相机",
                ],),
                
                "image_quality": ([
                    "None/无",
                    "8K Resolution/8K分辨率",
                    "4K Resolution/4K分辨率",
                    "Full HD 1080p/全高清1080p",
                    "Ultra High Definition/超高清",
                    "Super Resolution/超分辨率",
                    "RAW Format/RAW格式",
                    "Lossless Compression/无损压缩",
                    "High Bitrate/高比特率",
                    "HDR/高动态范围",
                    "HDR10+/HDR10+",
                    "Dolby Vision/杜比视界",
                    "10-bit Color/10位色彩",
                    "12-bit Color/12位色彩",
                    "14-bit Color/14位色彩",
                    "16-bit Color/16位色彩",
                    "Wide Color Gamut/广色域",
                    "ProRes 4444 XQ/ProRes最高质量",
                    "ProRes RAW/ProRes RAW",
                    "Cinema DNG/电影DNG",
                    "High ISO Performance/高ISO性能",
                    "Low Noise/低噪点",
                    "Zero Noise/零噪点",
                    "Crystal Clear/晶莹剔透",
                    "Tack Sharp/锐利无比",
                    "Perfect Focus/完美对焦",
                    "Shallow Depth of Field/浅景深",
                    "Deep Depth of Field/深景深",
                    "Bokeh Quality/焦外成像质量",
                    "Creamy Bokeh/奶油般焦外",
                    "Smooth Gradation/平滑渐变",
                    "Rich Tonality/丰富色调",
                    "High Contrast/高对比度",
                    "Low Contrast/低对比度",
                    "Film Grain/胶片颗粒",
                    "Fine Grain/细腻颗粒",
                    "Digital Clarity/数字清晰度",
                    "Professional Quality/专业品质",
                    "Museum Quality/博物馆品质",
                    "Gallery Quality/画廊品质",
                    "Award-Winning Quality/获奖品质",
                    "Masterpiece Quality/杰作品质",
                    "Cinema Quality/电影级品质",
                    "IMAX Quality/IMAX品质",
                    "Pristine/原始纯净",
                    "Flawless/完美无瑕",
                    "Exquisite Detail/精致细节",
                    "Hyper-Detailed/超级细节",
                    "Ultra-Detailed/极致细节",
                    "Micro Details/微观细节",
                    "Texture Rich/纹理丰富",
                    "Photorealistic/照片级真实",
                ],),
                
                "skin_texture": ([
                    "None/无",
                    "Porcelain Smooth/瓷器般光滑",
                    "Glass Skin/玻璃肌肤",
                    "Silk Smooth/丝绸般光滑",
                    "Velvet Soft/天鹅绒般柔软",
                    "Baby Soft/婴儿般柔嫩",
                    "Matte Finish/哑光质感",
                    "Dewy Glow/露水般光泽",
                    "Natural Pores/自然毛孔",
                    "Visible Pores/可见毛孔",
                    "Fine Lines/细纹",
                    "Wrinkles/皱纹",
                    "Crow's Feet/鱼尾纹",
                    "Laugh Lines/笑纹",
                    "Age Spots/老年斑",
                    "Freckles/雀斑",
                    "Beauty Marks/美人痣",
                    "Moles/痣",
                    "Scars/疤痕",
                    "Acne Scars/痤疮疤痕",
                    "Stretch Marks/妊娠纹",
                    "Goosebumps/鸡皮疙瘩",
                    "Peach Fuzz/绒毛",
                    "Translucent/半透明",
                    "Alabaster/雪花石膏般",
                    "Ivory/象牙色",
                    "Fair/白皙",
                    "Medium/中等肤色",
                    "Olive/橄榄色",
                    "Tan/古铜色",
                    "Bronze/青铜色",
                    "Dark/深色",
                    "Ebony/乌木色",
                    "Sun-Kissed/阳光亲吻",
                    "Sunburned/晒伤",
                    "Flushed/潮红",
                    "Pale/苍白",
                    "Glowing/发光",
                    "Radiant/容光焕发",
                    "Luminous/明亮",
                    "Shimmering/闪烁",
                    "Glossy/有光泽",
                    "Sweaty/出汗",
                    "Oily/油性",
                    "Dry/干燥",
                    "Smooth/光滑",
                    "Rough/粗糙",
                    "Textured/有纹理",
                    "Realistic/真实",
                    "Photorealistic/照片般真实",
                    "Subsurface Scattering/次表面散射",
                ],),
            }
        }
    
    # 只有一个文本输出
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    
    # 节点功能函数名
    FUNCTION = "generate_prompt"
    
    # 节点类别
    CATEGORY = "🐳Pond/text"
    
    # 节点描述
    DESCRIPTION = "Generate detailed photography and artistic parameters with single text I/O"
    
    def generate_prompt(self, input_text, output_language, lighting_type, camera_angle, 
                        aesthetic_style, emotional_mood, camera_model, 
                        image_quality, skin_texture):
        """
        Generate photography parameters prompt with single text I/O
        """
        
        # Helper function to extract language parts
        def extract_parts(param):
            if param == "None/无":
                return None, None
            if "/" in param:
                parts = param.split("/")
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else parts[0].strip()
            return param.strip(), param.strip()
        
        # Extract English and Chinese versions
        lighting_en, lighting_zh = extract_parts(lighting_type)
        angle_en, angle_zh = extract_parts(camera_angle)
        style_en, style_zh = extract_parts(aesthetic_style)
        mood_en, mood_zh = extract_parts(emotional_mood)
        camera_en, camera_zh = extract_parts(camera_model)
        quality_en, quality_zh = extract_parts(image_quality)
        texture_en, texture_zh = extract_parts(skin_texture)
        
        # Build parameter lists based on what's selected
        params_en = []
        params_zh = []
        
        if lighting_en:
            params_en.append(f"{lighting_en} lighting")
            params_zh.append(f"{lighting_zh}光照")
            
        if angle_en:
            params_en.append(f"{angle_en} angle")
            params_zh.append(f"{angle_zh}角度")
            
        if style_en:
            params_en.append(f"{style_en} style")
            params_zh.append(f"{style_zh}风格")
            
        if mood_en:
            params_en.append(f"{mood_en} mood")
            params_zh.append(f"{mood_zh}情绪")
            
        if camera_en:
            params_en.append(f"shot with {camera_en}")
            params_zh.append(f"使用{camera_zh}拍摄")
            
        if quality_en:
            params_en.append(quality_en)
            params_zh.append(quality_zh)
            
        if texture_en:
            params_en.append(f"skin texture: {texture_en}")
            params_zh.append(f"皮肤纹理: {texture_zh}")
        
        # Combine with input text
        if output_language == "English":
            if params_en:
                param_str = ", ".join(params_en)
                if input_text.strip():
                    result = f"{input_text.strip()}, {param_str}"
                else:
                    result = param_str
            else:
                result = input_text.strip()
                
        elif output_language == "中文":
            if params_zh:
                param_str = ", ".join(params_zh)
                if input_text.strip():
                    result = f"{input_text.strip()}, {param_str}"
                else:
                    result = param_str
            else:
                result = input_text.strip()
                
        else:  # Both/双语
            if params_en and params_zh:
                params_combined = []
                for en, zh in zip(params_en, params_zh):
                    params_combined.append(f"{en} ({zh})")
                param_str = ", ".join(params_combined)
                if input_text.strip():
                    result = f"{input_text.strip()}, {param_str}"
                else:
                    result = param_str
            else:
                result = input_text.strip()
        
        return (result,)


# ComfyUI 注册必需的映射
NODE_CLASS_MAPPINGS = {
    "PhotographyParameters": PhotographyParametersNode
}

# 显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotographyParameters": "🐳PhotoShop prompt 📸"
}