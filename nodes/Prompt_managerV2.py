import json
import os
from pathlib import Path

# 获取当前文件所在目录
CURRENT_DIR = Path(__file__).parent
TEMPLATES_DIR = CURRENT_DIR / "prompt_templates"

# 确保模板目录存在
TEMPLATES_DIR.mkdir(exist_ok=True)

class CustomPromptManagerEnhanced:
    
    # 类变量用于存储模板（跨实例共享）
    _templates_cache = None
    
    def __init__(self):
        self.prompts_data = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prompts_json": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "forceInput": False
                }),
                "separator": ("STRING", {
                    "default": ", ",
                    "multiline": False
                }),
                "use_weights": ("BOOLEAN", {
                    "default": True,
                    "label_on": "使用权重",
                    "label_off": "忽略权重"
                }),
                "enable_tags_filter": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用标签过滤",
                    "label_off": "关闭标签过滤"
                }),
                "filter_tags": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "标签过滤（逗号分隔）"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "process_prompts"
    CATEGORY = "🐳Pond/prompt"
    
    def process_prompts(self, prompts_json="[]", separator=", ", use_weights=True, 
                       enable_tags_filter=False, filter_tags=""):
        """
        处理带正负面分类的prompts，支持标签过滤
        """
        try:
            # 解析JSON数据
            if not prompts_json or prompts_json.strip() == "":
                prompts_json = "[]"
            
            prompts_list = json.loads(prompts_json)
            
            # 解析过滤标签
            filter_tags_list = []
            if enable_tags_filter and filter_tags.strip():
                filter_tags_list = [tag.strip().lower() for tag in filter_tags.split(",") if tag.strip()]
            
            positive_prompts = []
            negative_prompts = []
            
            # 处理每个prompt
            for prompt_data in prompts_list:
                if not isinstance(prompt_data, dict):
                    continue
                
                text = prompt_data.get("text", "")
                enabled = prompt_data.get("enabled", False)
                weight = prompt_data.get("weight", 1.0)
                prompt_type = prompt_data.get("type", "positive")
                tags = prompt_data.get("tags", [])
                
                # 如果启用且有内容
                if enabled and text.strip():
                    # 标签过滤逻辑
                    if enable_tags_filter and filter_tags_list:
                        # 检查是否有任何匹配的标签
                        prompt_tags_lower = [tag.lower() for tag in tags]
                        has_matching_tag = any(filter_tag in prompt_tags_lower 
                                             for filter_tag in filter_tags_list)
                        if not has_matching_tag:
                            continue  # 跳过不匹配的prompt
                    
                    # 应用权重
                    if use_weights and weight != 1.0:
                        weighted_text = f"({text.strip()}:{weight:.2f})"
                    else:
                        weighted_text = text.strip()
                    
                    # 根据类型分类
                    if prompt_type == "negative":
                        negative_prompts.append(weighted_text)
                    else:
                        positive_prompts.append(weighted_text)
            
            # 组合结果
            positive_combined = separator.join(positive_prompts) if positive_prompts else ""
            negative_combined = separator.join(negative_prompts) if negative_prompts else ""
            
            # print(f"[Prompt Manager] 输出 - 正面: {len(positive_prompts)}项, 负面: {len(negative_prompts)}项")
            
            return (positive_combined, negative_combined)
            
        except json.JSONDecodeError as e:
            print(f"[Prompt Manager Enhanced] JSON解析错误: {str(e)}")
            print(f"[Prompt Manager Enhanced] JSON内容: {prompts_json}")
            return ("", "")
        except Exception as e:
            print(f"[Prompt Manager Enhanced] 处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return ("", "")
    
    @classmethod
    def load_templates(cls):
        """加载所有模板"""
        if cls._templates_cache is not None:
            return cls._templates_cache
        
        templates = {}
        if TEMPLATES_DIR.exists():
            for template_file in TEMPLATES_DIR.glob("*.json"):
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        templates[template_file.stem] = template_data
                except Exception as e:
                    print(f"[Prompt Manager] 加载模板失败 {template_file.name}: {str(e)}")
        
        cls._templates_cache = templates
        return templates
    
    @classmethod
    def save_template(cls, template_name, template_data):
        """保存模板到文件"""
        try:
            template_file = TEMPLATES_DIR / f"{template_name}.json"
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            
            # 更新缓存
            if cls._templates_cache is None:
                cls._templates_cache = {}
            cls._templates_cache[template_name] = template_data
            
            return True
        except Exception as e:
            print(f"[Prompt Manager] 保存模板失败: {str(e)}")
            return False
    
    @classmethod
    def delete_template(cls, template_name):
        """删除模板"""
        try:
            template_file = TEMPLATES_DIR / f"{template_name}.json"
            if template_file.exists():
                template_file.unlink()
                
                # 更新缓存
                if cls._templates_cache and template_name in cls._templates_cache:
                    del cls._templates_cache[template_name]
                
                return True
            return False
        except Exception as e:
            print(f"[Prompt Manager] 删除模板失败: {str(e)}")
            return False
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """强制ComfyUI检测变化"""
        return float("nan")


# 注册节点
NODE_CLASS_MAPPINGS = {
    "CustomPromptManagerEnhanced": CustomPromptManagerEnhanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomPromptManagerEnhanced": "🐳Prompt管理器星球版",
}
