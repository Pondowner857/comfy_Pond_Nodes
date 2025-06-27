import os
import sys
import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management
from typing import Tuple, List, Dict, Any

# 将插件目录添加到 Python 路径
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.append(plugin_dir)

# 尝试导入所需库
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from huggingface_hub import snapshot_download
except ImportError:
    print("未找到所需库。请安装: pip install transformers huggingface_hub")
    raise

class WatermarkObjectDetector:
    """
    ComfyUI 节点，使用 HuggingFace 的 GroundingDINO 模型
    检测图像中的水印或任意指定对象
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = comfy.model_management.get_torch_device()
        self.model_name = "IDEA-Research/grounding-dino-tiny"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "watermark",
                    "multiline": False,
                    "placeholder": "输入要检测的内容（如：水印、logo、文字）"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "nms_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                "model_name": ("STRING", {
                    "default": "IDEA-Research/grounding-dino-tiny",
                    "multiline": False,
                    "placeholder": "HuggingFace 模型名称"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "BBOXES")
    RETURN_NAMES = ("遮罩", "边界框")
    FUNCTION = "detect"
    CATEGORY = "🐳Pond/image"
    
    def get_model_path(self):
        """获取模型存储路径"""
        models_dir = folder_paths.models_dir
        detector_models_dir = os.path.join(models_dir, "object_detectors")
        os.makedirs(detector_models_dir, exist_ok=True)
        return detector_models_dir
    
    def load_model(self, model_name=None):
        """从 HuggingFace 加载检测模型"""
        if model_name is None:
            model_name = self.model_name
            
        if self.model is None or model_name != self.model_name:
            print(f"正在加载模型: {model_name}")
            
            # 设置缓存目录
            cache_dir = self.get_model_path()
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            
            try:
                # 加载处理器和模型
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                ).to(self.device)
                self.model_name = model_name
                print(f"模型加载成功: {model_name}")
            except Exception as e:
                print(f"模型加载错误: {e}")
                raise
    
    def detect(self, image, prompt, threshold, nms_threshold, model_name=None):
        """
        根据文本提示检测图像中的对象
        
        参数:
            image: 输入图像张量 (B, H, W, C)
            prompt: 描述要检测内容的文本提示
            threshold: 检测置信度阈值
            nms_threshold: 非极大值抑制阈值
            model_name: 可选的自定义模型名称
            
        返回:
            mask: 检测区域的二值遮罩
            bboxes: 边界框，格式为 [[x1, y1, x2, y2, 置信度, 类别ID], ...]
        """
        # 根据需要加载模型
        if model_name and model_name != self.model_name:
            self.load_model(model_name)
        elif self.model is None:
            self.load_model()
        
        # 将 ComfyUI 图像格式转换为 PIL
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np, mode='RGB')
        
        # 使用模型处理图像
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 后处理结果
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(pil_image.height, pil_image.width)]
        )[0]
        
        # 提取边界框和分数
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]
        
        # 如果需要，应用 NMS
        if len(boxes) > 0 and nms_threshold < 1.0:
            from torchvision.ops import nms
            keep = nms(
                torch.tensor(boxes),
                torch.tensor(scores),
                nms_threshold
            ).numpy()
            boxes = boxes[keep]
            scores = scores[keep]
            labels = [labels[i] for i in keep]
        
        # 创建遮罩
        mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
        
        # 创建边界框列表
        bboxes_list = []
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # 添加到遮罩
            mask[y1:y2, x1:x2] = 1.0
            
            # 添加到边界框列表（格式：[x1, y1, x2, y2, 置信度, 类别ID]）
            bboxes_list.append([float(x1), float(y1), float(x2), float(y2), float(score), 0])
        
        # 将遮罩转换为 ComfyUI 格式 (B, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        # 将边界框转换为自定义格式
        bboxes_output = BoundingBoxes(bboxes_list, image_np.shape[1], image_np.shape[0])
        
        return (mask_tensor, bboxes_output)

class BoundingBoxes:
    """
    在 ComfyUI 中存储边界框的自定义类
    """
    def __init__(self, boxes: List[List[float]], width: int, height: int):
        self.boxes = boxes  # [[x1, y1, x2, y2, 置信度, 类别ID], ...]
        self.width = width
        self.height = height
    
    def to_list(self) -> List[List[float]]:
        return self.boxes
    
    def to_normalized(self) -> List[List[float]]:
        """转换为归一化坐标 (0-1)"""
        normalized = []
        for box in self.boxes:
            x1, y1, x2, y2, conf, cls = box
            normalized.append([
                x1 / self.width,
                y1 / self.height,
                x2 / self.width,
                y2 / self.height,
                conf,
                cls
            ])
        return normalized
    
    def __len__(self):
        return len(self.boxes)
    
    def __repr__(self):
        return f"边界框({len(self.boxes)} 个框, {self.width}x{self.height})"

class DrawBoundingBoxes:
    """
    在图像上可视化边界框的辅助节点
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOXES",),
                "color": ("STRING", {"default": "red"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "show_confidence": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "🐳Pond/image"
    
    def draw(self, image, bboxes, color, thickness, show_confidence):
        """在图像上绘制边界框"""
        import cv2
        
        # 颜色映射
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        
        draw_color = color_map.get(color.lower(), (255, 0, 0))
        
        # 将图像转换为 numpy
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        result = image_np.copy()
        
        # 绘制每个边界框
        for box in bboxes.to_list():
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制矩形
            cv2.rectangle(result, (x1, y1), (x2, y2), draw_color, thickness)
            
            # 如果需要，绘制置信度分数
            if show_confidence:
                text = f"{conf:.2f}"
                cv2.putText(result, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
        
        # 转换回 ComfyUI 格式
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "WatermarkObjectDetector": WatermarkObjectDetector,
    "DrawBoundingBoxes": DrawBoundingBoxes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WatermarkObjectDetector": "水印/对象检测器",
    "DrawBoundingBoxes": "绘制边界框",
}

# 自定义类型注册
def register_custom_types():
    """为 ComfyUI 注册自定义类型"""
    # 这允许 BBOXES 在节点之间传递
    if hasattr(comfy, 'supported_types') and "BBOXES" not in comfy.supported_types:
        comfy.supported_types.add("BBOXES")

# 模块加载时尝试注册类型
try:
    register_custom_types()
except:
    pass

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']