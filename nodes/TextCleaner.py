import re

class TextCleanerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型
        """
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "tags": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "mode": (["删除包含标签/提示词的句子", "删除标签/提示词"], {"default": "删除包含标签/提示词的句子"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_text",)
    FUNCTION = "clean_text"
    CATEGORY = "🐳Pond/text"

    def parse_tags(self, tags):
        """
        解析标签，支持多种分隔方式，包括中英文标点
        """
        # 中英文标点符号集合
        separators = r'[,;、\n\s。，；：？！,.?!：]+' 
        
        # 分割并清理标签
        tag_list = [
            tag.strip().lower() 
            for tag in re.split(separators, tags) 
            if tag.strip()
        ]
        
        return tag_list

    def clean_text(self, text, tags, mode):
        """
        清理文本的主要函数
        
        :param text: 输入文本
        :param tags: 要删除的标签
        :param mode: 清理模式
        :return: 清理后的文本
        """
        # 如果没有标签，直接返回原文
        if not tags:
            return (text,)

        # 解析标签
        tag_list = self.parse_tags(tags)

        # 处理不同的清理模式
        if mode == "删除标签/提示词":
            # 删除特定标签/提示词
            for tag in tag_list:
                # 使用正则替换，忽略大小写
                text = re.sub(re.escape(tag), '', text, flags=re.IGNORECASE)
        
        elif mode == "删除包含标签/提示词的句子":
            # 分割句子（支持中文和英文的句子分隔）
            sentences = re.split(r'([。！？!?.]+)', text)
            
            # 重新组装句子，排除包含指定标签的句子
            filtered_sentences = []
            for i in range(0, len(sentences), 2):
                # 检查句子是否包含任何标签
                if not any(tag in sentences[i].lower() for tag in tag_list):
                    # 如果句子不包含标签，加入结果
                    filtered_sentences.append(sentences[i])
                    # 如果是最后一个句子前的分隔符，也加入
                    if i + 1 < len(sentences):
                        filtered_sentences.append(sentences[i+1])
            
            # 重新组装文本
            text = ''.join(filtered_sentences)

        # 返回清理后的文本
        return (text.strip(),)

# 定义 WEB_DICT 以支持节点在 ComfyUI 中的展示
NODE_CLASS_MAPPINGS = {
    "TextCleanerNode": TextCleanerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextCleanerNode": "🐳文本清理器"
}