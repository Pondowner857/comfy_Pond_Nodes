import re

class TextFormatParser:
    """
    支持动态输出端口数量的文本格式解析节点
    根据方括号 [content] 自动提取内容
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Category: [clothing]\nColor: [red]\nInstruction: [Put a clothing on the model.]"
                }),
                "output_count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }
    
    # 定义足够多的输出端口支持动态扩展，实际显示数量由JS控制
    RETURN_TYPES = tuple(["STRING"] * 100)
    RETURN_NAMES = tuple([f"输出_{i+1}" for i in range(100)])
    FUNCTION = "parse_text"
    CATEGORY = "🐳Pond/prompt"
    OUTPUT_NODE = False
    
    def parse_text(self, text, output_count):
        """
        解析文本中所有 [content] 格式的内容
        """
        output_count = min(max(output_count, 1), 100)
        
        # 规范化换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 使用正则表达式提取所有 [content] 的内容
        # 匹配方括号内的内容,使用非贪婪匹配
        pattern = r'\[([^\[\]]+)\]'
        
        try:
            matches = re.findall(pattern, text)
            
            # 去除每个结果的首尾空白
            results = [match.strip() for match in matches]
            
            if results:
                print(f"[TextFormatParser] 成功提取了 {len(results)} 个方括号内容")
                for i, r in enumerate(results):
                    print(f"[TextFormatParser] 输出_{i+1}: {r[:50]}{'...' if len(r) > 50 else ''}")
            else:
                print(f"[TextFormatParser] 未找到任何 [content] 格式的内容")
            
            # 补齐或截取到output_count个
            while len(results) < output_count:
                results.append("")
            
            return tuple(results[:output_count])
                
        except Exception as e:
            print(f"[TextFormatParser] 解析错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return tuple([""] * output_count)


NODE_CLASS_MAPPINGS = {
    "TextFormatParser": TextFormatParser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextFormatParser": "🐳Prompt解析"
}