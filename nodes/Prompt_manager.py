"""
ComfyUI Custom Prompt Manager Node - Positive/Negative Version Only
支持前端动态控制prompt数量、内容和正负面分类
"""
import json

class CustomPromptManagerWithNegative:
    """
    带正负面提示词的自定义Prompt管理器
    支持正面和负面提示词的分别管理和输出
    """
    
    def __init__(self):
        self.prompts_data = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_json": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "dynamicPrompts": False
                }),
            },
            "optional": {
                "separator": ("STRING", {
                    "default": ", ",
                    "multiline": False
                }),
                "use_weights": ("BOOLEAN", {
                    "default": True,
                    "label_on": "使用权重",
                    "label_off": "忽略权重"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "process_prompts"
    CATEGORY = "🐳Pond/prompt"
    
    def process_prompts(self, prompts_json, separator=", ", use_weights=True):
        """
        处理带正负面分类的prompts
        """
        try:
            # 解析JSON数据
            prompts_list = json.loads(prompts_json)
            
            positive_prompts = []
            negative_prompts = []
            
            # 处理每个prompt
            for prompt_data in prompts_list:
                if not isinstance(prompt_data, dict):
                    continue
                
                text = prompt_data.get("text", "")
                enabled = prompt_data.get("enabled", False)
                weight = prompt_data.get("weight", 1.0)
                prompt_type = prompt_data.get("type", "positive")  # positive 或 negative
                
                # 如果启用且有内容
                if enabled and text.strip():
                    # 应用权重
                    if use_weights and weight != 1.0:
                        weighted_text = f"({text.strip()}:{weight:.1f})"
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
            
            return (positive_combined, negative_combined)
            
        except json.JSONDecodeError as e:
            print(f"[Prompt Manager] JSON解析错误: {str(e)}")
            return ("", "")
        except Exception as e:
            print(f"[Prompt Manager] 处理错误: {str(e)}")
            return ("", "")


# 注册节点
NODE_CLASS_MAPPINGS = {
    "CustomPromptManagerWithNegative": CustomPromptManagerWithNegative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomPromptManagerWithNegative": "🐳Prompt管理器",
}
