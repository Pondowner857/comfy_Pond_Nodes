import torch
import numpy as np
import os
import sys
from PIL import Image
import folder_paths

# 尝试导入ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("警告: 未安装ultralytics库，请运行: pip install ultralytics")
    YOLO = None

class YoloBboxesCropNode:
    """
    集成YOLO检测和裁剪功能的节点
    自动加载YOLO模型，检测并裁剪图像中的对象
    """
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # 获取YOLO模型文件夹路径
        yolo_models_dir = os.path.join(folder_paths.models_dir, "yolo")
        
        # 获取可用的模型列表
        model_files = []
        if os.path.exists(yolo_models_dir):
            for file in os.listdir(yolo_models_dir):
                if file.endswith(('.pt', '.onnx', '.engine')):
                    model_files.append(file)
        
        if not model_files:
            model_files = ["请将YOLO模型放入models/yolo文件夹"]
        
        return {
            "required": {
                "image": ("IMAGE", {"display": "输入图像"}),
                "model_name": (model_files, {
                    "default": model_files[0] if model_files else "yolov8n.pt",
                    "display": "YOLO模型"
                }),
                "confidence": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "置信度阈值"
                }),
                "class_filter": ("STRING", {
                    "default": "全部",
                    "multiline": False,
                    "display": "类别过滤",
                    "tooltip": "要检测的类别，用逗号分隔（如：人,汽车）或'全部'检测所有"
                }),
                "expand_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "display": "宽度扩展"
                }),
                "expand_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "display": "高度扩展"
                }),
                "crop_mode": (["全部对象", "单个对象", "按类别"], {
                    "default": "全部对象",
                    "display": "裁剪模式"
                }),
                "object_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "对象索引"
                }),
                "output_mode": (["单独图像", "拼接图像"], {
                    "default": "单独图像",
                    "display": "输出模式"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOXES", "STRING", "INT")
    RETURN_NAMES = ("裁剪图像", "遮罩", "边界框", "检测信息", "检测数量")
    FUNCTION = "detect_and_crop"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "使用YOLO模型检测图像中的对象并进行智能裁剪"

    def load_model(self, model_name):
        """加载YOLO模型"""
        if YOLO is None:
            raise ImportError("请安装ultralytics库: pip install ultralytics")
        
        # 如果已经加载了相同的模型，直接返回
        if self.model is not None and self.current_model_name == model_name:
            return self.model
        
        # 构建模型路径
        model_path = os.path.join(folder_paths.models_dir, "yolo", model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        print(f"加载YOLO模型: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            self.current_model_name = model_name
            print(f"成功加载模型: {model_name}")
            return self.model
        except Exception as e:
            raise RuntimeError(f"加载YOLO模型失败: {str(e)}")

    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # 从CHW或HWC格式转换为HWC
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:
            tensor = tensor.permute(1, 2, 0)
        
        # 转换为numpy
        img = tensor.cpu().numpy()
        
        # 确保值在0-255范围内
        if img.max() <= 1.0:
            img = img * 255
        
        img = img.astype(np.uint8)
        
        # 如果是单通道，转换为RGB
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        
        return Image.fromarray(img)

    def pil_to_tensor(self, pil_img):
        """将PIL图像转换为tensor"""
        img = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).float()
        return tensor

    def filter_detections(self, results, class_filter):
        """根据类别过滤检测结果"""
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            return []
        
        filtered_boxes = []
        
        # 中文类别名称映射
        class_name_cn = {
            'person': '人',
            'bicycle': '自行车',
            'car': '汽车',
            'motorcycle': '摩托车',
            'airplane': '飞机',
            'bus': '公交车',
            'train': '火车',
            'truck': '卡车',
            'boat': '船',
            'traffic light': '交通灯',
            'fire hydrant': '消防栓',
            'stop sign': '停止标志',
            'parking meter': '停车计时器',
            'bench': '长椅',
            'bird': '鸟',
            'cat': '猫',
            'dog': '狗',
            'horse': '马',
            'sheep': '羊',
            'cow': '牛',
            'elephant': '大象',
            'bear': '熊',
            'zebra': '斑马',
            'giraffe': '长颈鹿',
            'backpack': '背包',
            'umbrella': '雨伞',
            'handbag': '手提包',
            'tie': '领带',
            'suitcase': '手提箱',
            'frisbee': '飞盘',
            'skis': '滑雪板',
            'snowboard': '滑雪板',
            'sports ball': '运动球',
            'kite': '风筝',
            'baseball bat': '棒球棒',
            'baseball glove': '棒球手套',
            'skateboard': '滑板',
            'surfboard': '冲浪板',
            'tennis racket': '网球拍',
            'bottle': '瓶子',
            'wine glass': '酒杯',
            'cup': '杯子',
            'fork': '叉子',
            'knife': '刀',
            'spoon': '勺子',
            'bowl': '碗',
            'banana': '香蕉',
            'apple': '苹果',
            'sandwich': '三明治',
            'orange': '橙子',
            'broccoli': '西兰花',
            'carrot': '胡萝卜',
            'hot dog': '热狗',
            'pizza': '披萨',
            'donut': '甜甜圈',
            'cake': '蛋糕',
            'chair': '椅子',
            'couch': '沙发',
            'potted plant': '盆栽',
            'bed': '床',
            'dining table': '餐桌',
            'toilet': '马桶',
            'tv': '电视',
            'laptop': '笔记本电脑',
            'mouse': '鼠标',
            'remote': '遥控器',
            'keyboard': '键盘',
            'cell phone': '手机',
            'microwave': '微波炉',
            'oven': '烤箱',
            'toaster': '烤面包机',
            'sink': '水槽',
            'refrigerator': '冰箱',
            'book': '书',
            'clock': '时钟',
            'vase': '花瓶',
            'scissors': '剪刀',
            'teddy bear': '泰迪熊',
            'hair drier': '吹风机',
            'toothbrush': '牙刷'
        }
        
        # 如果检测所有类别
        if class_filter.lower() == 'all' or class_filter == '全部':
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 获取类别名称
                class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                class_name_display = class_name_cn.get(class_name, class_name)
                
                filtered_boxes.append({
                    'bbox': xyxy.tolist(),
                    'confidence': float(conf),
                    'class': class_name,
                    'class_display': class_name_display,
                    'class_id': cls
                })
        else:
            # 解析要检测的类别（支持中英文）
            target_classes = [c.strip().lower() for c in class_filter.split(',')]
            
            # 创建反向映射（中文到英文）
            class_name_en = {v: k for k, v in class_name_cn.items()}
            
            # 转换中文类别名到英文
            target_classes_en = []
            for tc in target_classes:
                if tc in class_name_en:
                    target_classes_en.append(class_name_en[tc])
                else:
                    target_classes_en.append(tc)
            
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                
                if class_name.lower() in target_classes_en:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_name_display = class_name_cn.get(class_name, class_name)
                    
                    filtered_boxes.append({
                        'bbox': xyxy.tolist(),
                        'confidence': float(conf),
                        'class': class_name,
                        'class_display': class_name_display,
                        'class_id': cls
                    })
        
        return filtered_boxes

    def crop_single_object(self, image, detection, expand_width, expand_height):
        """裁剪单个检测对象"""
        height, width = image.shape[:2]
        
        # 获取边界框坐标
        x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
        
        # 应用扩展
        x1_expanded = max(0, x1 - expand_width)
        y1_expanded = max(0, y1 - expand_height)
        x2_expanded = min(width, x2 + expand_width)
        y2_expanded = min(height, y2 + expand_height)
        
        # 裁剪图像
        cropped = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        
        # 创建遮罩
        mask = np.zeros((height, width), dtype=np.float32)
        mask[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = 1.0
        
        return cropped, mask, (x1_expanded, y1_expanded, x2_expanded, y2_expanded)

    def detect_and_crop(self, image, model_name, confidence, class_filter, 
                       expand_width, expand_height, crop_mode, object_index, output_mode):
        """执行检测和裁剪"""
        # 确保图像是4维的
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 加载模型
        try:
            model = self.load_model(model_name)
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 返回原图和空遮罩
            empty_mask = torch.zeros((1, image.shape[1], image.shape[2], 1), dtype=torch.float32)
            empty_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            return (image, empty_mask, empty_bboxes, "模型加载失败", 0)
        
        # 转换为PIL图像进行检测
        pil_img = self.tensor_to_pil(image)
        
        # 执行检测
        print(f"执行YOLO检测，置信度阈值: {confidence}")
        results = model(pil_img, conf=confidence, verbose=False)
        
        # 过滤检测结果
        detections = self.filter_detections(results, class_filter)
        
        if not detections:
            print("未检测到任何对象")
            empty_mask = torch.zeros((1, image.shape[1], image.shape[2], 1), dtype=torch.float32)
            empty_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            return (image, empty_mask, empty_bboxes, "未检测到对象", 0)
        
        print(f"检测到 {len(detections)} 个对象")
        
        # 转换为numpy数组进行处理
        img_np = np.array(pil_img)
        
        # 根据裁剪模式选择要处理的对象
        if crop_mode == "单个对象":
            if object_index >= len(detections):
                print(f"警告: 对象索引 {object_index} 超出范围，使用最后一个对象")
                object_index = len(detections) - 1
            selected_detections = [detections[object_index]]
        elif crop_mode == "按类别":
            # 按类别分组，每个类别选择置信度最高的
            class_dict = {}
            for det in detections:
                cls = det['class']
                if cls not in class_dict or det['confidence'] > class_dict[cls]['confidence']:
                    class_dict[cls] = det
            selected_detections = list(class_dict.values())
        else:  # 全部对象
            selected_detections = detections
        
        # 裁剪选中的对象
        cropped_images = []
        masks = []
        bboxes = []
        detection_info = []
        
        for i, det in enumerate(selected_detections):
            cropped, mask, bbox = self.crop_single_object(img_np, det, expand_width, expand_height)
            
            # 转换为tensor
            cropped_tensor = self.pil_to_tensor(Image.fromarray(cropped))
            mask_tensor = torch.from_numpy(mask).float()
            
            cropped_images.append(cropped_tensor.unsqueeze(0))
            masks.append(mask_tensor.unsqueeze(0).unsqueeze(-1))
            bboxes.append(bbox)
            
            # 记录检测信息
            info = f"[{i}] {det['class_display']} (置信度: {det['confidence']:.2f})"
            detection_info.append(info)
        
        # 合并结果
        if output_mode == "拼接图像" and len(cropped_images) > 1:
            # 将所有裁剪图像拼接成批次
            final_images = torch.cat(cropped_images, dim=0)
        else:
            # 单独输出每个图像
            final_images = torch.cat(cropped_images, dim=0)
        
        # 合并遮罩（使用最大值保留重叠区域）
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = torch.maximum(final_mask, mask)
        
        # 转换边界框为tensor
        final_bboxes = torch.tensor(bboxes, dtype=torch.float32)
        
        # 生成检测信息字符串
        info_str = f"检测到 {len(detections)} 个对象，裁剪了 {len(selected_detections)} 个\n"
        info_str += "\n".join(detection_info)
        
        return (final_images, final_mask, final_bboxes, info_str, len(selected_detections))

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloBboxesCropNode": YoloBboxesCropNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloBboxesCropNode": "🐳YOLO检测裁剪"
}