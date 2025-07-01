import math
from typing import List, Tuple, Any, Dict

class MultiNumberCompare:
    """多数值比较节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_inputs": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "display": "int"
                }),
                "output_mode": (["max", "min", "median", "average", "sum", "sorted_asc", "sorted_desc", "range"], {
                    "default": "max"
                }),
            },
            "optional": {
                "int_1": ("INT", {"default": 0, "min": -999999, "max": 999999}),
                "int_2": ("INT", {"default": 0, "min": -999999, "max": 999999}),
                "int_3": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
                "int_4": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
                "int_5": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
                "int_6": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
                "int_7": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
                "int_8": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
                "int_9": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
                "int_10": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")  # 改为INT类型
    RETURN_NAMES = ("result", "secondary_result", "info")
    FUNCTION = "compare_numbers"
    CATEGORY = "🐳Pond/Tools"
    
    def compare_numbers(self, num_inputs, output_mode, **kwargs):
        # 收集所有有效的输入数值（保持为整数）
        numbers = []
        for i in range(1, num_inputs + 1):
            key = f"int_{i}"
            if key in kwargs and kwargs[key] is not None:
                numbers.append(int(kwargs[key]))  # 保持为整数
        
        if not numbers:
            return (0, 0, "No valid inputs")
        
        # 根据输出模式处理数值
        result = 0
        secondary_result = 0
        info = ""
        
        if output_mode == "max":
            result = max(numbers)
            info = f"Maximum of {len(numbers)} numbers: {result}"
            
        elif output_mode == "min":
            result = min(numbers)
            info = f"Minimum of {len(numbers)} numbers: {result}"
            
        elif output_mode == "median":
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            if n % 2 == 0:
                # 中位数取两个中间值的平均数，四舍五入为整数
                result = round((sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2)
            else:
                result = sorted_nums[n//2]
            info = f"Median of {len(numbers)} numbers: {result}"
            
        elif output_mode == "average":
            # 平均值四舍五入为整数
            result = round(sum(numbers) / len(numbers))
            # 计算标准差作为secondary_result，也四舍五入为整数
            variance = sum((x - result) ** 2 for x in numbers) / len(numbers)
            secondary_result = round(math.sqrt(variance))
            info = f"Average: {result}, Std Dev: {secondary_result}"
            
        elif output_mode == "sum":
            result = sum(numbers)
            secondary_result = len(numbers)  # 返回数量作为辅助信息
            info = f"Sum of {len(numbers)} numbers: {result}"
            
        elif output_mode == "sorted_asc":
            sorted_nums = sorted(numbers)
            result = sorted_nums[0] if sorted_nums else 0
            secondary_result = sorted_nums[-1] if sorted_nums else 0
            info = f"Sorted ascending: {', '.join([str(n) for n in sorted_nums[:5]])}"
            if len(sorted_nums) > 5:
                info += "..."
                
        elif output_mode == "sorted_desc":
            sorted_nums = sorted(numbers, reverse=True)
            result = sorted_nums[0] if sorted_nums else 0
            secondary_result = sorted_nums[-1] if sorted_nums else 0
            info = f"Sorted descending: {', '.join([str(n) for n in sorted_nums[:5]])}"
            if len(sorted_nums) > 5:
                info += "..."
                
        elif output_mode == "range":
            result = max(numbers) - min(numbers)
            secondary_result = (max(numbers) + min(numbers)) // 2  # 中点，使用整数除法
            info = f"Range: {result}, Midpoint: {secondary_result}"
        
        return (result, secondary_result, info)


class AspectRatioCalculator:
    """宽高比计算器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1920,
                    "min": 1,
                    "max": 10000
                }),
                "height": ("INT", {
                    "default": 1080,
                    "min": 1,
                    "max": 10000
                }),
                "target_value": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 10000
                }),
                "target_mode": (["width", "height", "max_dimension", "min_dimension", "area"], {
                    "default": "width"
                }),
                "constraint_mode": (["keep_ratio", "max_total", "min_total"], {
                    "default": "keep_ratio"
                }),
                "round_to": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "int"
                }),
            },
            "optional": {
                "preset_ratio": (["custom", "1:1", "4:3", "3:2", "16:9", "21:9", "9:16", "2:3", "3:4"], {
                    "default": "custom"
                }),
                "max_width": ("INT", {"default": 4096, "min": 1, "max": 10000}),
                "max_height": ("INT", {"default": 4096, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING", "STRING")  # 将ratio改为STRING类型
    RETURN_NAMES = ("width", "height", "ratio_text", "info")
    FUNCTION = "calculate_aspect_ratio"
    CATEGORY = "🐳Pond/Tools"
    
    def calculate_aspect_ratio(self, width, height, target_value, target_mode, 
                              constraint_mode, round_to, preset_ratio="custom",
                              max_width=4096, max_height=4096):
        
        # 如果选择了预设比例，先应用预设
        if preset_ratio != "custom":
            ratio_map = {
                "1:1": (1, 1),
                "4:3": (4, 3),
                "3:2": (3, 2),
                "16:9": (16, 9),
                "21:9": (21, 9),
                "9:16": (9, 16),
                "2:3": (2, 3),
                "3:4": (3, 4)
            }
            if preset_ratio in ratio_map:
                preset_w, preset_h = ratio_map[preset_ratio]
                width, height = preset_w, preset_h
        
        # 计算原始宽高比
        ratio = width / height
        
        # 根据目标模式计算新的宽高
        width, height = float(width), float(height)
        
        if target_mode == "width":
            width = float(target_value)
            height = target_value / ratio
            
        elif target_mode == "height":
            height = float(target_value)
            width = target_value * ratio
            
        elif target_mode == "max_dimension":
            if width > height:
                width = float(target_value)
                height = target_value / ratio
            else:
                height = float(target_value)
                width = target_value * ratio
                
        elif target_mode == "min_dimension":
            if width < height:
                width = float(target_value)
                height = target_value / ratio
            else:
                height = float(target_value)
                width = target_value * ratio
                
        elif target_mode == "area":
            # 保持比例，使面积接近目标值
            target_area = target_value
            current_area = width * height
            scale = math.sqrt(target_area / current_area)
            width = width * scale
            height = height * scale
        
        # 应用约束
        if constraint_mode == "max_total":
            total = width + height
            if total > target_value:
                scale = target_value / total
                width *= scale
                height *= scale
                
        elif constraint_mode == "min_total":
            total = width + height
            if total < target_value:
                scale = target_value / total
                width *= scale
                height *= scale
        
        # 应用最大值限制
        if width > max_width:
            scale = max_width / width
            width = max_width
            height *= scale
            
        if height > max_height:
            scale = max_height / height
            height = max_height
            width *= scale
        
        # 四舍五入到指定的倍数
        width = round(width / round_to) * round_to
        height = round(height / round_to) * round_to
        
        # 确保不为0
        width = max(round_to, width)
        height = max(round_to, height)
        
        # 转换为整数
        width = int(width)
        height = int(height)
        
        # 计算最终比例并格式化为字符串
        final_ratio = width / height
        ratio_text = f"{final_ratio:.3f}"
        
        # 生成信息字符串
        info = f"Original: {width}x{height} (ratio: {ratio:.3f})\n"
        info += f"New: {width}x{height} (ratio: {final_ratio:.3f})\n"
        info += f"Scale: {width/width:.3f}x"
        
        return (width, height, ratio_text, info)


class MathOperations:
    """数学运算节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -999999, "max": 999999}),
                "b": ("INT", {"default": 0, "min": -999999, "max": 999999}),
                "operation": ([
                    "add", "subtract", "multiply", "divide", "power", 
                    "modulo", "min", "max", "average", "distance",
                    "log", "exp", "sin", "cos", "tan", "atan2"
                ], {"default": "add"}),
            },
            "optional": {
                "c": ("INT", {"default": 0, "min": -999999, "max": 999999, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("result", "formula")
    FUNCTION = "calculate"
    CATEGORY = "🐳Pond/Tools"
    
    def calculate(self, a, b, operation, c=None):
        result = 0
        formula = ""
        
        try:
            if operation == "add":
                result = a + b + (c if c is not None else 0)
                formula = f"{a} + {b}" + (f" + {c}" if c is not None else "") + f" = {result}"
                
            elif operation == "subtract":
                result = a - b - (c if c is not None else 0)
                formula = f"{a} - {b}" + (f" - {c}" if c is not None else "") + f" = {result}"
                
            elif operation == "multiply":
                result = a * b * (c if c is not None else 1)
                formula = f"{a} × {b}" + (f" × {c}" if c is not None else "") + f" = {result}"
                
            elif operation == "divide":
                if b != 0:
                    result_float = a / b
                    if c is not None and c != 0:
                        result_float = result_float / c
                    result = round(result_float)  # 四舍五入
                    formula = f"{a} ÷ {b}" + (f" ÷ {c}" if c is not None else "") + f" = {result}"
                else:
                    result = 0
                    formula = "Division by zero!"
                    
            elif operation == "power":
                result = round(math.pow(a, b))  # 四舍五入
                formula = f"{a}^{b} = {result}"
                
            elif operation == "modulo":
                if b != 0:
                    result = a % b
                    formula = f"{a} mod {b} = {result}"
                else:
                    result = 0
                    formula = "Modulo by zero!"
                    
            elif operation == "min":
                values = [a, b]
                if c is not None:
                    values.append(c)
                result = min(values)
                formula = f"min({', '.join([str(v) for v in values])}) = {result}"
                
            elif operation == "max":
                values = [a, b]
                if c is not None:
                    values.append(c)
                result = max(values)
                formula = f"max({', '.join([str(v) for v in values])}) = {result}"
                
            elif operation == "average":
                values = [a, b]
                if c is not None:
                    values.append(c)
                result = round(sum(values) / len(values))  # 四舍五入
                formula = f"avg({', '.join([str(v) for v in values])}) = {result}"
                
            elif operation == "distance":
                result = round(math.sqrt(a**2 + b**2 + (c**2 if c is not None else 0)))  # 四舍五入
                formula = f"√({a}² + {b}²" + (f" + {c}²" if c is not None else "") + f") = {result}"
                
            elif operation == "log":
                if a > 0 and b > 0:
                    result = round(math.log(a, b))  # 四舍五入
                    formula = f"log_{b}({a}) = {result}"
                else:
                    result = 0
                    formula = "Invalid logarithm input!"
                    
            elif operation == "exp":
                result = round(math.exp(a))  # 四舍五入
                formula = f"e^{a} = {result}"
                
            elif operation == "sin":
                result = round(math.sin(math.radians(a)) * 1000)  # 乘以1000保留精度，四舍五入
                formula = f"sin({a}°) × 1000 = {result}"
                
            elif operation == "cos":
                result = round(math.cos(math.radians(a)) * 1000)  # 乘以1000保留精度，四舍五入
                formula = f"cos({a}°) × 1000 = {result}"
                
            elif operation == "tan":
                result = round(math.tan(math.radians(a)) * 1000)  # 乘以1000保留精度，四舍五入
                formula = f"tan({a}°) × 1000 = {result}"
                
            elif operation == "atan2":
                result = round(math.degrees(math.atan2(b, a)))  # 四舍五入
                formula = f"atan2({b}, {a}) = {result}°"
                
        except Exception as e:
            result = 0
            formula = f"Error: {str(e)}"
        
        # 确保结果是整数
        result = int(result)
        
        return (result, formula)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "MultiNumberCompare": MultiNumberCompare,
    "AspectRatioCalculator": AspectRatioCalculator,
    "MathOperations": MathOperations,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiNumberCompare": "🐳多数值比较",
    "AspectRatioCalculator": "🐳宽高比计算",
    "MathOperations": "🐳数学运算",
}

# Web界面扩展
# 路径：web文件夹在插件目录内
WEB_DIRECTORY = "./web/js"