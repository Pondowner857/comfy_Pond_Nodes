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
                "sort_by": (["默认", "从左到右", "从右到左", "从上到下", "从下到上", "置信度降序", "置信度升序", "面积降序", "面积升序"], {
                    "default": "从左到右",
                    "display": "排序方式"
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
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOXES", "STRING", "INT")
    RETURN_NAMES = ("裁剪图像", "遮罩", "边界框", "检测信息", "检测数量")
    OUTPUT_IS_LIST = (True, False, False, False, False)  # 只有图像输出为列表
    FUNCTION = "detect_and_crop"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "使用YOLO模型检测图像中的对象并进行智能裁剪，根据对象大小自动调整裁剪框，支持多种排序方式"

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
        
        try:
            # 对于PyTorch 2.6+，需要添加安全全局变量
            if hasattr(torch.serialization, 'add_safe_globals'):
                safe_list = []
                
                # 添加PyTorch基础模块
                try:
                    safe_list.extend([
                        torch.nn.modules.container.Sequential,
                        torch.nn.modules.container.ModuleList,
                        torch.nn.modules.container.ModuleDict,
                    ])
                except:
                    pass
                
                # 添加YOLO相关模块
                try:
                    from ultralytics.nn.tasks import DetectionModel
                    from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
                    safe_list.extend([DetectionModel, Conv, C2f, SPPF, Detect])
                except ImportError:
                    pass
                
                # 添加更多可能需要的模块
                try:
                    safe_list.extend([
                        torch.nn.Conv2d,
                        torch.nn.BatchNorm2d,
                        torch.nn.SiLU,
                        torch.nn.Upsample,
                        torch.nn.MaxPool2d,
                    ])
                except:
                    pass
                
                if safe_list:
                    torch.serialization.add_safe_globals(safe_list)
            
            # 强制设置weights_only=False的另一种方法
            import os as os_module
            os_module.environ['YOLO_AUTOINSTALL'] = 'False'
            
            # 使用上下文管理器临时修改torch.load
            class LoadContext:
                def __enter__(self):
                    self.original_load = torch.load
                    torch.load = lambda *args, **kwargs: self.original_load(
                        *args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, 
                        weights_only=False
                    )
                    return self
                
                def __exit__(self, *args):
                    torch.load = self.original_load
            
            with LoadContext():
                self.model = YOLO(model_path)
                
            self.current_model_name = model_name
            return self.model
            
        except Exception as e:
            print(f"加载失败: {e}")
            # 最后的备选方案：使用monkey patch
            try:
                import ultralytics.engine.model
                original_torch_load = torch.load
                
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)
                
                torch.load = patched_load
                self.model = YOLO(model_path)
                torch.load = original_torch_load
                
                self.current_model_name = model_name
                return self.model
            except Exception as e2:
                raise RuntimeError(f"加载YOLO模型失败: {str(e2)}")

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
        """使用智能尺寸调整裁剪单个检测对象"""
        height, width = image.shape[:2]
        
        # 获取检测框坐标
        xmin, ymin, xmax, ymax = detection['bbox']
        
        # 计算检测到的对象框的宽度和高度
        object_width = xmax - xmin
        object_height = ymax - ymin
        object_size = max(object_width, object_height)  # 使用较大的边作为基准
        
        # 根据对象尺寸计算合适的方块大小
        # 添加额外的边距
        actual_square_size = object_size * object_margin
        
        # 应用用户指定的百分比调整
        actual_square_size = actual_square_size * (square_size / 100.0)
        
        # 计算边界框的中心
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        # 使用实际的方块大小
        half_size = actual_square_size / 2
        
        # 计算偏移后的坐标
        vertical_offset_px = (actual_square_size) * vertical_offset / 100
        horizontal_offset_px = (actual_square_size) * horizontal_offset / 100
        
        # 计算新的边界框坐标，保持正方形大小
        x1_new = max(0, int(center_x - half_size + horizontal_offset_px))
        x2_new = min(width, int(center_x + half_size + horizontal_offset_px))
        y1_new = max(0, int(center_y - half_size + vertical_offset_px))
        y2_new = min(height, int(center_y + half_size + vertical_offset_px))
        
        # 确保裁剪区域是正方形
        crop_width = x2_new - x1_new
        crop_height = y2_new - y1_new
        
        if crop_width != crop_height:
            # 如果不是正方形，调整为正方形
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

    def detect_and_crop(self, image, model_name, confidence, class_filter, 
                       square_size, object_margin, vertical_offset, horizontal_offset,
                       sort_by, crop_mode, object_index):
        """执行检测和裁剪"""
        # 确保图像是4维的
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 保存原始图像尺寸
        batch_size, tensor_height, tensor_width, channels = image.shape
        
        # 加载模型
        try:
            model = self.load_model(model_name)
        except Exception as e:
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = []
            return ([image], empty_mask, empty_bboxes, "模型加载失败", 0)
        
        # 转换为PIL图像进行检测
        pil_img = self.tensor_to_pil(image)
        
        # 执行检测
        results = model(pil_img, conf=confidence, verbose=False)
        
        # 过滤检测结果
        detections = self.filter_detections(results, class_filter)
        
        # 对检测结果进行排序
        detections = self.sort_detections(detections, sort_by)
        
        if not detections:
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = []
            return ([image], empty_mask, empty_bboxes, "未检测到对象", 0)
        
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
        else:  # 全部对象
            selected_detections = detections
        
        # 裁剪选中的对象
        cropped_images = []
        bboxes = []
        detection_info = []
        
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
        
        # 生成检测信息字符串
        info_str = f"检测到 {len(detections)} 个对象，裁剪了 {len(selected_detections)} 个\n"
        info_str += f"裁剪设置: 大小={square_size}%, 边距={object_margin}x, 垂直偏移={vertical_offset}%, 水平偏移={horizontal_offset}%\n"
        info_str += "\n".join(detection_info)
        
        # 返回裁剪的图像列表和合并的遮罩
        return (cropped_images, final_mask, final_bboxes, info_str, len(selected_detections))

# 节点注册
NODE_CLASS_MAPPINGS = {
    "YoloBboxesCropNode": YoloBboxesCropNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloBboxesCropNode": "🐳YOLO智能裁剪"
}