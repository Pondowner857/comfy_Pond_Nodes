import torch
import numpy as np
import os
import sys
from PIL import Image
import folder_paths

# 尝试导入ultralytics
try:
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version
    print(f"Ultralytics版本: {ultralytics_version}")
except ImportError:
    print("警告: 未安装ultralytics库，请运行: pip install ultralytics>=8.2.0")
    YOLO = None

class YoloV11BboxesCropNode:
    """YOLOv11专用检测和裁剪节点"""

    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # 获取YOLO模型文件夹路径
        yolo_models_dir = os.path.join(folder_paths.models_dir, "yolo")
        
        # 获取可用的YOLOv11模型列表
        model_files = []
        if os.path.exists(yolo_models_dir):
            for file in os.listdir(yolo_models_dir):
                # 筛选v11模型
                if file.endswith(('.pt', '.onnx')) and ('v11' in file.lower() or 'yolo11' in file.lower()):
                    model_files.append(file)
        
        # 添加默认YOLOv11模型选项
        default_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        for model in default_models:
            if model not in model_files:
                model_files.append(model + " (需下载)")
        
        if not model_files:
            model_files = ["请将YOLOv11模型放入models/yolo文件夹"]
        
        return {
            "required": {
                "image": ("IMAGE", {"display": "输入图像"}),
                "model_name": (model_files, {
                    "default": model_files[0] if model_files else "yolo11n.pt",
                    "display": "YOLOv11模型"
                }),
                "device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "display": "运行设备"
                }),
                "confidence": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "置信度阈值"
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "IOU阈值"
                }),
                "imgsz": ("INT", {
                    "default": 640,
                    "min": 320,
                    "max": 1280,
                    "step": 32,
                    "display": "推理尺寸"
                }),
                "max_det": ("INT", {
                    "default": 300,
                    "min": 1,
                    "max": 1000,
                    "step": 10,
                    "display": "最大检测数"
                }),
                "class_filter": ("STRING", {
                    "default": "全部",
                    "multiline": False,
                    "display": "类别过滤",
                    "tooltip": "要检测的类别，用逗号分隔（如：人,汽车）或'全部'检测所有"
                }),
                "square_size": ("FLOAT", {
                    "default": 100.0,
                    "min": 10.0,
                    "max": 200.0,
                    "step": 1.0,
                    "display": "方框大小%",
                    "tooltip": "基于检测对象大小的百分比调整"
                }),
                "object_margin": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "对象边距系数",
                    "tooltip": "在检测对象周围添加的额外边距系数"
                }),
                "vertical_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -50.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "垂直偏移%"
                }),
                "horizontal_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -50.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "水平偏移%"
                }),
                "sort_by": (["默认", "从左到右", "从右到左", "从上到下", "从下到上", 
                           "置信度降序", "置信度升序", "面积降序", "面积升序"], {
                    "default": "从左到右",
                    "display": "排序方式"
                }),
                "crop_mode": (["全部对象", "单个对象", "按类别", "批量处理"], {
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
                "augment": ("BOOLEAN", {
                    "default": False,
                    "display": "测试时增强"
                }),
                "agnostic_nms": ("BOOLEAN", {
                    "default": False,
                    "display": "类别无关NMS"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOXES", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("裁剪图像", "遮罩", "边界框", "检测信息", "检测数量", "平均置信度")
    OUTPUT_IS_LIST = (True, False, False, False, False, False)
    FUNCTION = "detect_and_crop_v11"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "YOLOv11专用版本，支持最新的YOLOv11模型特性，包括改进的检测性能和更多参数控制"

    def get_device(self, device_preference):
        """智能选择设备"""
        if device_preference == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_preference

    def load_model_v11(self, model_name, device="auto"):
        """加载YOLOv11模型"""
        if YOLO is None:
            raise ImportError("请安装ultralytics库: pip install ultralytics>=8.2.0")
        
        # 处理设备选择
        actual_device = self.get_device(device)
        
        # 如果已经加载了相同的模型和设备，直接返回
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.device == actual_device):
            return self.model
        
        # 清理"需下载"标记
        clean_model_name = model_name.replace(" (需下载)", "")
        
        # 构建模型路径
        model_path = os.path.join(folder_paths.models_dir, "yolo", clean_model_name)
        
        # 如果本地文件不存在，尝试自动下载官方模型
        if not os.path.exists(model_path) and clean_model_name in ["yolo11n.pt", "yolo11s.pt", 
                                                                   "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]:
            print(f"自动下载YOLOv11模型: {clean_model_name}")
            try:
                # 设置环境变量
                os.environ['YOLO_VERBOSE'] = 'False'
                
                # 直接使用模型名称，让ultralytics自动下载
                self.model = YOLO(clean_model_name)
                self.model.to(actual_device)
                self.current_model_name = model_name
                self.device = actual_device
                
                # 保存到本地以供后续使用
                save_path = model_path
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.model.save(save_path)
                print(f"模型已保存到: {save_path}")
                
                return self.model
            except Exception as e:
                print(f"自动下载失败: {e}")
                raise FileNotFoundError(f"无法下载或加载模型: {clean_model_name}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        try:
            print(f"加载YOLOv11模型: {clean_model_name} 在设备: {actual_device}")
            
            # 设置环境变量
            os.environ['YOLO_VERBOSE'] = 'False'
            
            # 加载模型
            self.model = YOLO(model_path)
            self.model.to(actual_device)
            
            # 验证是否为v11模型
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml'):
                model_yaml = self.model.model.yaml
                if isinstance(model_yaml, dict) and 'model' in model_yaml:
                    model_info = model_yaml.get('model', '')
                    print(f"模型信息: {model_info}")
            
            self.current_model_name = model_name
            self.device = actual_device
            
            return self.model
            
        except Exception as e:
            print(f"加载YOLOv11模型失败: {e}")
            raise RuntimeError(f"加载YOLOv11模型失败: {str(e)}")

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

    def get_class_names_v11(self):
        """获取YOLOv11的类别名称映射"""
        # COCO数据集的80个类别（YOLOv11默认）
        class_name_cn = {
            'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车',
            'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车',
            'boat': '船', 'traffic light': '交通灯', 'fire hydrant': '消防栓',
            'stop sign': '停止标志', 'parking meter': '停车计时器', 'bench': '长椅',
            'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊',
            'cow': '牛', 'elephant': '大象', 'bear': '熊', 'zebra': '斑马',
            'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
            'handbag': '手提包', 'tie': '领带', 'suitcase': '手提箱',
            'frisbee': '飞盘', 'skis': '滑雪板', 'snowboard': '滑雪板',
            'sports ball': '运动球', 'kite': '风筝', 'baseball bat': '棒球棒',
            'baseball glove': '棒球手套', 'skateboard': '滑板', 'surfboard': '冲浪板',
            'tennis racket': '网球拍', 'bottle': '瓶子', 'wine glass': '酒杯',
            'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺子',
            'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治',
            'orange': '橙子', 'broccoli': '西兰花', 'carrot': '胡萝卜',
            'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈', 'cake': '蛋糕',
            'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽',
            'bed': '床', 'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视',
            'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器',
            'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉',
            'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水槽',
            'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 'vase': '花瓶',
            'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机',
            'toothbrush': '牙刷'
        }
        return class_name_cn

    def filter_detections_v11(self, results, class_filter):
        """根据类别过滤YOLOv11检测结果"""
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        # 检查YOLOv11的输出格式
        if not hasattr(result, 'boxes') or result.boxes is None:
            return []
        
        filtered_boxes = []
        class_name_cn = self.get_class_names_v11()
        
        # 创建反向映射（中文到英文）
        class_name_en = {v: k for k, v in class_name_cn.items()}
        
        # 处理检测结果
        boxes = result.boxes
        
        # 检测所有类别
        if class_filter.lower() == 'all' or class_filter == '全部':
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
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
            # 解析要检测的类别
            target_classes = [c.strip().lower() for c in class_filter.split(',')]
            
            # 转换中文类别名到英文
            target_classes_en = []
            for tc in target_classes:
                if tc in class_name_en:
                    target_classes_en.append(class_name_en[tc])
                else:
                    target_classes_en.append(tc)
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].cpu().numpy())
                class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                
                if class_name.lower() in target_classes_en:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    class_name_display = class_name_cn.get(class_name, class_name)
                    
                    filtered_boxes.append({
                        'bbox': xyxy.tolist(),
                        'confidence': float(conf),
                        'class': class_name,
                        'class_display': class_name_display,
                        'class_id': cls
                    })
        
        return filtered_boxes

    def sort_detections(self, detections, sort_by):
        """根据指定方式对检测结果排序"""
        if not detections or sort_by == "默认":
            return detections
        
        if sort_by == "从左到右":
            return sorted(detections, key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2)
        elif sort_by == "从右到左":
            return sorted(detections, key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2, reverse=True)
        elif sort_by == "从上到下":
            return sorted(detections, key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2)
        elif sort_by == "从下到上":
            return sorted(detections, key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2, reverse=True)
        elif sort_by == "置信度降序":
            return sorted(detections, key=lambda d: d['confidence'], reverse=True)
        elif sort_by == "置信度升序":
            return sorted(detections, key=lambda d: d['confidence'])
        elif sort_by == "面积降序":
            def get_area(d):
                x1, y1, x2, y2 = d['bbox']
                return (x2 - x1) * (y2 - y1)
            return sorted(detections, key=get_area, reverse=True)
        elif sort_by == "面积升序":
            def get_area(d):
                x1, y1, x2, y2 = d['bbox']
                return (x2 - x1) * (y2 - y1)
            return sorted(detections, key=get_area)
        
        return detections

    def crop_single_object(self, image, detection, square_size, object_margin, 
                          vertical_offset, horizontal_offset, original_height, 
                          original_width, scale_x=1.0, scale_y=1.0):
        """智能裁剪单个检测对象"""
        height, width = image.shape[:2]
        
        # 获取检测框坐标
        xmin, ymin, xmax, ymax = detection['bbox']
        
        # 计算检测到的对象框的宽度和高度
        object_width = xmax - xmin
        object_height = ymax - ymin
        object_size = max(object_width, object_height)
        
        # 根据对象尺寸计算合适的方块大小
        actual_square_size = object_size * object_margin
        actual_square_size = actual_square_size * (square_size / 100.0)
        
        # 计算边界框的中心
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        # 使用实际的方块大小
        half_size = actual_square_size / 2
        
        # 计算偏移后的坐标
        vertical_offset_px = actual_square_size * vertical_offset / 100
        horizontal_offset_px = actual_square_size * horizontal_offset / 100
        
        # 计算新的边界框坐标，保持正方形大小
        x1_new = max(0, int(center_x - half_size + horizontal_offset_px))
        x2_new = min(width, int(center_x + half_size + horizontal_offset_px))
        y1_new = max(0, int(center_y - half_size + vertical_offset_px))
        y2_new = min(height, int(center_y + half_size + vertical_offset_px))
        
        # 确保裁剪区域是正方形
        crop_width = x2_new - x1_new
        crop_height = y2_new - y1_new
        
        if crop_width != crop_height:
            target_size = min(crop_width, crop_height)
            center_x_new = (x1_new + x2_new) / 2
            center_y_new = (y1_new + y2_new) / 2
            half_target = target_size / 2
            
            x1_new = max(0, int(center_x_new - half_target))
            x2_new = min(width, int(center_x_new + half_target))
            y1_new = max(0, int(center_y_new - half_target))
            y2_new = min(height, int(center_y_new + half_target))
        
        # 裁剪图像
        cropped = image[y1_new:y2_new, x1_new:x2_new]
        
        # 创建原图尺寸的遮罩
        mask = np.zeros((original_height, original_width), dtype=np.float32)
        
        # 计算在原始尺寸上的坐标
        mask_x1 = int(x1_new * scale_x)
        mask_y1 = int(y1_new * scale_y)
        mask_x2 = int(x2_new * scale_x)
        mask_y2 = int(y2_new * scale_y)
        
        # 确保坐标在有效范围内
        mask_x1 = max(0, min(mask_x1, original_width))
        mask_y1 = max(0, min(mask_y1, original_height))
        mask_x2 = max(mask_x1, min(mask_x2, original_width))
        mask_y2 = max(mask_y1, min(mask_y2, original_height))
        
        if mask_x2 > mask_x1 and mask_y2 > mask_y1:
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1.0
        
        return cropped, mask, (x1_new, y1_new, x2_new, y2_new)

    def detect_and_crop_v11(self, image, model_name, device, confidence, iou_threshold,
                           imgsz, max_det, class_filter, square_size, object_margin, 
                           vertical_offset, horizontal_offset, sort_by, crop_mode, 
                           object_index, augment, agnostic_nms):
        """执行YOLOv11检测和裁剪"""
        
        # 确保图像是4维的
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 保存原始图像尺寸
        batch_size, tensor_height, tensor_width, channels = image.shape
        
        # 加载模型
        try:
            model = self.load_model_v11(model_name, device)
        except Exception as e:
            print(f"模型加载失败: {e}")
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = []
            return ([image], empty_mask, empty_bboxes, f"模型加载失败: {str(e)}", 0, 0.0)
        
        # 转换为PIL图像进行检测
        pil_img = self.tensor_to_pil(image)
        
        # 执行YOLOv11检测
        try:
            results = model.predict(
                source=pil_img,
                conf=confidence,
                iou=iou_threshold,
                imgsz=imgsz,
                max_det=max_det,
                augment=augment,
                agnostic_nms=agnostic_nms,
                verbose=False
            )
        except Exception as e:
            print(f"检测失败: {e}")
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = []
            return ([image], empty_mask, empty_bboxes, f"检测失败: {str(e)}", 0, 0.0)
        
        # 过滤检测结果
        detections = self.filter_detections_v11(results, class_filter)
        
        # 对检测结果进行排序
        detections = self.sort_detections(detections, sort_by)
        
        if not detections:
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = []
            return ([image], empty_mask, empty_bboxes, "未检测到对象", 0, 0.0)
        
        # 转换为numpy数组进行处理
        img_np = np.array(pil_img)
        
        # 使用tensor的原始尺寸作为遮罩尺寸
        original_height, original_width = tensor_height, tensor_width
        
        # 计算缩放比例
        pil_height, pil_width = img_np.shape[:2]
        scale_x = original_width / pil_width
        scale_y = original_height / pil_height
        
        # 根据裁剪模式选择要处理的对象
        if crop_mode == "单个对象":
            if object_index >= len(detections):
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
        elif crop_mode == "批量处理":
            # 批量处理模式：选择置信度最高的前N个
            selected_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:10]
        else:  # 全部对象
            selected_detections = detections
        
        # 裁剪选中的对象
        cropped_images = []
        bboxes = []
        detection_info = []
        total_confidence = 0.0
        
        # 创建一个与原图尺寸相同的合并遮罩
        combined_mask = torch.zeros((original_height, original_width), dtype=torch.float32)
        
        for i, det in enumerate(selected_detections):
            cropped, mask, bbox = self.crop_single_object(
                img_np, det, square_size, object_margin, 
                vertical_offset, horizontal_offset,
                original_height, original_width, scale_x, scale_y
            )
            
            # 转换为tensor
            cropped_tensor = self.pil_to_tensor(Image.fromarray(cropped))
            mask_tensor = torch.from_numpy(mask).float()
            
            # 为裁剪的图像添加批次维度
            cropped_images.append(cropped_tensor.unsqueeze(0))
            
            # 合并遮罩
            combined_mask = torch.maximum(combined_mask, mask_tensor)
            
            bboxes.append(bbox)
            total_confidence += det['confidence']
            
            # 记录检测信息
            center_x = int((det['bbox'][0] + det['bbox'][2]) / 2)
            center_y = int((det['bbox'][1] + det['bbox'][3]) / 2)
            object_size = max(det['bbox'][2] - det['bbox'][0], det['bbox'][3] - det['bbox'][1])
            info = f"[{i}] {det['class_display']} (置信度: {det['confidence']:.2f}, 中心: x={center_x},y={center_y}, 尺寸: {object_size:.0f})"
            detection_info.append(info)
        
        # 为合并遮罩添加批次维度
        final_mask = combined_mask.unsqueeze(0)
        final_mask = final_mask.clamp(0.0, 1.0)
        
        # 确保边界框格式正确
        final_bboxes = []
        for bbox in bboxes:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                final_bboxes.append(list(bbox))
        
        # 计算平均置信度
        avg_confidence = total_confidence / len(selected_detections) if selected_detections else 0.0
        
        # 生成检测信息字符串
        info_str = f"YOLOv11 检测结果\n"
        info_str += f"模型: {model_name} | 设备: {self.device}\n"
        info_str += f"检测到 {len(detections)} 个对象，裁剪了 {len(selected_detections)} 个\n"
        info_str += f"平均置信度: {avg_confidence:.3f}\n"
        info_str += f"推理尺寸: {imgsz} | IOU阈值: {iou_threshold}\n"
        info_str += f"裁剪设置: 大小={square_size}%, 边距={object_margin}x, 垂直偏移={vertical_offset}%, 水平偏移={horizontal_offset}%\n"
        info_str += "\n".join(detection_info)
        
        # 返回裁剪的图像列表和合并的遮罩
        return (cropped_images, final_mask, final_bboxes, info_str, len(selected_detections), avg_confidence)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloV11BboxesCropNode": YoloV11BboxesCropNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloV11BboxesCropNode": "🐳YOLOv11智能裁剪"
}