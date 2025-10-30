import re

class TextFormatParser:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Category: clothing\nInstruction: Put a clothing on the model."
                }),
                "format_pattern": ("STRING", {
                    "multiline": True,  # 改为支持多行
                    "default": "Category: {0}\nInstruction: {1}"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("输出_1", "输出_2", "输出_3", "输出_4", "输出_5")
    FUNCTION = "parse_text"
    CATEGORY = "🐳Pond/prompt"
    
    def parse_text(self, text, format_pattern):
        """
        解析文本并根据占位符数量返回对应输出
        """
        # 计算占位符数量
        placeholder_count = 0
        for i in range(5):
            if f"{{{i}}}" in format_pattern:
                placeholder_count = i + 1
            else:
                break
        
        # 如果没有占位符，返回空
        if placeholder_count == 0:
            print("[TextFormatParser] 格式中没有找到占位符 {0}, {1} 等")
            return ("", "", "", "", "")
        
        # 规范化文本和格式（统一处理换行符）
        # 将所有类型的换行符统一为 \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        format_pattern = format_pattern.replace('\r\n', '\n').replace('\r', '\n')
        
        # 构建正则表达式
        regex_pattern = format_pattern
        
        # 先替换占位符为临时标记
        temp_markers = []
        for i in range(placeholder_count):
            marker = f"<!PLACEHOLDER_{i}!>"
            regex_pattern = regex_pattern.replace(f"{{{i}}}", marker)
            temp_markers.append(marker)
        
        # 转义特殊的正则字符
        regex_pattern = re.escape(regex_pattern)
        
        # 将临时标记替换为捕获组
        # 使用 (.+?) 进行非贪婪匹配
        for marker in temp_markers:
            regex_pattern = regex_pattern.replace(re.escape(marker), "(.+?)")
        
        # 处理换行符：允许前后有可选的空白字符
        # \\n 是转义后的换行符
        regex_pattern = regex_pattern.replace(r'\\n', r'\\s*\\n\\s*')
        
        # 允许行首尾有可选空格
        regex_pattern = r'^\s*' + regex_pattern + r'\s*$'
        
        print(f"[TextFormatParser] 占位符数量: {placeholder_count}")
        print(f"[TextFormatParser] 正则模式: {regex_pattern[:200]}...")  # 只显示前200个字符
        
        # 尝试匹配
        try:
            # 使用 DOTALL 和 MULTILINE 标志
            match = re.search(regex_pattern, text, re.DOTALL | re.MULTILINE)
            
            if match:
                results = list(match.groups())
                # 清理结果：去除首尾空白，包括换行符
                results = [r.strip() if r else "" for r in results]
                
                print(f"[TextFormatParser] 匹配成功！提取了 {len(results)} 个结果")
                for i, r in enumerate(results):
                    print(f"[TextFormatParser] 输出_{i+1}: {r[:50]}{'...' if len(r) > 50 else ''}")
                
                # 确保有placeholder_count个结果
                while len(results) < placeholder_count:
                    results.append("")
                
                # 截取实际需要的数量
                results = results[:placeholder_count]
                
                # 补齐到5个（未使用的输出返回空字符串）
                while len(results) < 5:
                    results.append("")
                
                return tuple(results)
            else:
                print(f"[TextFormatParser] 无法匹配文本")
                print(f"[TextFormatParser] 文本内容: {text[:200]}...")
                print(f"[TextFormatParser] 格式模式: {format_pattern[:200]}...")
                print(f"[TextFormatParser] 提示：检查格式模式是否与文本完全匹配")
                return ("", "", "", "", "")
                
        except Exception as e:
            print(f"[TextFormatParser] 解析错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return ("", "", "", "", "")


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "TextFormatParser": TextFormatParser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextFormatParser": "🐳Prompt解析"
}
