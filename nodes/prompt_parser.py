import re

class TextFormatParser:
    """
    支持动态输出端口数量的文本格式解析节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Category: clothing\nInstruction: Put a clothing on the model."
                }),
                "format_pattern": ("STRING", {
                    "multiline": True,
                    "default": "Category: {0}\nInstruction: {1}"
                }),
                "output_count": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }
    
    # 初始只定义2个输出端口，其他由前端动态添加
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("输出_1", "输出_2")
    FUNCTION = "parse_text"
    CATEGORY = "🐳Pond/prompt"
    OUTPUT_NODE = False
    
    def parse_text(self, text, format_pattern, output_count):
        """
        解析文本并根据占位符数量返回对应输出
        """
        output_count = min(max(output_count, 1), 100)
        
        # 计算占位符数量
        placeholder_count = 0
        for i in range(output_count):
            if f"{{{i}}}" in format_pattern:
                placeholder_count = i + 1
            else:
                break
        
        if placeholder_count == 0:
            print("[TextFormatParser] 格式中没有找到占位符 {0}, {1} 等")
            return tuple([""] * output_count)
        
        # 规范化换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        format_pattern = format_pattern.replace('\r\n', '\n').replace('\r', '\n')
        
        # 构建正则表达式
        regex_pattern = format_pattern
        
        temp_markers = []
        for i in range(placeholder_count):
            marker = f"<!PLACEHOLDER_{i}!>"
            regex_pattern = regex_pattern.replace(f"{{{i}}}", marker)
            temp_markers.append(marker)
        
        regex_pattern = re.escape(regex_pattern)
        
        for marker in temp_markers:
            regex_pattern = regex_pattern.replace(re.escape(marker), "(.+?)")
        
        regex_pattern = regex_pattern.replace(r'\\n', r'\\s*\\n\\s*')
        regex_pattern = r'^\s*' + regex_pattern + r'\s*$'
        
        print(f"[TextFormatParser] 占位符数量: {placeholder_count}")
        print(f"[TextFormatParser] 输出端口数量: {output_count}")
        
        try:
            match = re.search(regex_pattern, text, re.DOTALL | re.MULTILINE)
            
            if match:
                results = list(match.groups())
                results = [r.strip() if r else "" for r in results]
                
                print(f"[TextFormatParser] 匹配成功！提取了 {len(results)} 个结果")
                for i, r in enumerate(results):
                    print(f"[TextFormatParser] 输出_{i+1}: {r[:50]}{'...' if len(r) > 50 else ''}")
                
                # 补齐到output_count个
                while len(results) < output_count:
                    results.append("")
                
                return tuple(results[:output_count])
            else:
                print(f"[TextFormatParser] 无法匹配文本")
                return tuple([""] * output_count)
                
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
