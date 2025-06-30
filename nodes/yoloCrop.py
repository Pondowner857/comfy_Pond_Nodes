import torch
import numpy as np
import os
import sys
from PIL import Image
import folder_paths

# Try to import ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics library not installed, please run: pip install ultralytics")
    YOLO = None

class YoloBboxesCropNode:

    def __init__(self):
        self.model = None
        self.current_model_name = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # Get YOLO models folder path
        yolo_models_dir = os.path.join(folder_paths.models_dir, "yolo")
        
        # Get available model list
        model_files = []
        if os.path.exists(yolo_models_dir):
            for file in os.listdir(yolo_models_dir):
                if file.endswith(('.pt', '.onnx', '.engine')):
                    model_files.append(file)
        
        if not model_files:
            model_files = ["Please place YOLO models in models/yolo folder"]
        
        return {
            "required": {
                "image": ("IMAGE", {"display": "Input Image"}),
                "model_name": (model_files, {
                    "default": model_files[0] if model_files else "yolov8n.pt",
                    "display": "YOLO Model"
                }),
                "confidence": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "Confidence Threshold"
                }),
                "class_filter": ("STRING", {
                    "default": "all",
                    "multiline": False,
                    "display": "Class Filter",
                    "tooltip": "Classes to detect, comma-separated (e.g.: person,car) or 'all' to detect all"
                }),
                "expand_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "display": "Width Expansion"
                }),
                "expand_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "display": "Height Expansion"
                }),
                "sort_by": (["default", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "confidence_desc", "confidence_asc", "area_desc", "area_asc"], {
                    "default": "left_to_right",
                    "display": "Sort By"
                }),
                "crop_mode": (["all_objects", "single_object", "by_class"], {
                    "default": "all_objects",
                    "display": "Crop Mode"
                }),
                "object_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "Object Index"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOXES", "STRING", "INT")
    RETURN_NAMES = ("cropped_images", "mask", "bboxes", "detection_info", "detection_count")
    OUTPUT_IS_LIST = (True, False, False, False, False)  # Only image output is list
    FUNCTION = "detect_and_crop"
    CATEGORY = "🐳Pond/yolo"
    DESCRIPTION = "Detect objects in images using YOLO models and perform intelligent cropping with various sorting options, outputs single mask"

    def load_model(self, model_name):
        """Load YOLO model"""
        if YOLO is None:
            raise ImportError("Please install ultralytics library: pip install ultralytics")
        
        # If same model already loaded, return it
        if self.model is not None and self.current_model_name == model_name:
            return self.model
        
        # Build model path
        model_path = os.path.join(folder_paths.models_dir, "yolo", model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading YOLO model: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            self.current_model_name = model_name
            print(f"Successfully loaded model: {model_name}")
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL image"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # Convert from CHW or HWC format to HWC
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy
        img = tensor.cpu().numpy()
        
        # Ensure values are in 0-255 range
        if img.max() <= 1.0:
            img = img * 255
        
        img = img.astype(np.uint8)
        
        # If single channel, convert to RGB
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        
        return Image.fromarray(img)

    def pil_to_tensor(self, pil_img):
        """Convert PIL image to tensor"""
        img = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).float()
        return tensor

    def filter_detections(self, results, class_filter):
        """Filter detections based on class"""
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            return []
        
        filtered_boxes = []
        
        # If detecting all classes
        if class_filter.lower() == 'all':
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                
                filtered_boxes.append({
                    'bbox': xyxy.tolist(),
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': cls
                })
        else:
            # Parse classes to detect
            target_classes = [c.strip().lower() for c in class_filter.split(',')]
            
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                
                if class_name.lower() in target_classes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    filtered_boxes.append({
                        'bbox': xyxy.tolist(),
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': cls
                    })
        
        return filtered_boxes

    def sort_detections(self, detections, sort_by):
        """Sort detections by specified method"""
        if not detections or sort_by == "default":
            return detections
        
        if sort_by == "left_to_right":
            # Sort by x-coordinate of bbox center
            return sorted(detections, key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2)
        elif sort_by == "right_to_left":
            # Sort by x-coordinate of bbox center descending
            return sorted(detections, key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2, reverse=True)
        elif sort_by == "top_to_bottom":
            # Sort by y-coordinate of bbox center
            return sorted(detections, key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2)
        elif sort_by == "bottom_to_top":
            # Sort by y-coordinate of bbox center descending
            return sorted(detections, key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2, reverse=True)
        elif sort_by == "confidence_desc":
            # Sort by confidence descending
            return sorted(detections, key=lambda d: d['confidence'], reverse=True)
        elif sort_by == "confidence_asc":
            # Sort by confidence ascending
            return sorted(detections, key=lambda d: d['confidence'])
        elif sort_by == "area_desc":
            # Sort by bbox area descending
            def get_area(d):
                x1, y1, x2, y2 = d['bbox']
                return (x2 - x1) * (y2 - y1)
            return sorted(detections, key=get_area, reverse=True)
        elif sort_by == "area_asc":
            # Sort by bbox area ascending
            def get_area(d):
                x1, y1, x2, y2 = d['bbox']
                return (x2 - x1) * (y2 - y1)
            return sorted(detections, key=get_area)
        
        return detections

    def crop_single_object(self, image, detection, expand_width, expand_height, original_height, original_width, scale_x=1.0, scale_y=1.0):
        """Crop single detected object"""
        height, width = image.shape[:2]
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
        
        # Apply expansion
        x1_expanded = max(0, x1 - expand_width)
        y1_expanded = max(0, y1 - expand_height)
        x2_expanded = min(width, x2 + expand_width)
        y2_expanded = min(height, y2 + expand_height)
        
        # Crop image
        cropped = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
        
        # Create mask at original image size, considering possible scaling
        mask = np.zeros((original_height, original_width), dtype=np.float32)
        
        # Calculate coordinates on original size
        mask_x1 = int(x1_expanded * scale_x)
        mask_y1 = int(y1_expanded * scale_y)
        mask_x2 = int(x2_expanded * scale_x)
        mask_y2 = int(y2_expanded * scale_y)
        
        # Ensure coordinates are within valid range
        mask_x1 = max(0, min(mask_x1, original_width - 1))
        mask_y1 = max(0, min(mask_y1, original_height - 1))
        mask_x2 = max(mask_x1 + 1, min(mask_x2, original_width))
        mask_y2 = max(mask_y1 + 1, min(mask_y2, original_height))
        
        if mask_x2 > mask_x1 and mask_y2 > mask_y1:
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1.0
        
        return cropped, mask, (x1_expanded, y1_expanded, x2_expanded, y2_expanded)

    def detect_and_crop(self, image, model_name, confidence, class_filter, 
                       expand_width, expand_height, sort_by, crop_mode, object_index):
        """Execute detection and cropping"""
        # Ensure image is 4-dimensional
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Save original image dimensions (from tensor)
        batch_size, tensor_height, tensor_width, channels = image.shape
        
        # Load model
        try:
            model = self.load_model(model_name)
        except Exception as e:
            print(f"Model loading failed: {e}")
            # Return original image and empty mask
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            return ([image], empty_mask, empty_bboxes, "Model loading failed", 0)
        
        # Convert to PIL image for detection
        pil_img = self.tensor_to_pil(image)
        
        # Execute detection
        print(f"Executing YOLO detection with confidence threshold: {confidence}")
        results = model(pil_img, conf=confidence, verbose=False)
        
        # Filter detection results
        detections = self.filter_detections(results, class_filter)
        
        # Sort detection results
        detections = self.sort_detections(detections, sort_by)
        
        if not detections:
            print("No objects detected")
            # Return original image and empty mask of original size
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            return ([image], empty_mask, empty_bboxes, "No objects detected", 0)
        
        print(f"Detected {len(detections)} objects, sort by: {sort_by}")
        
        # Convert to numpy array for processing
        img_np = np.array(pil_img)
        
        # Use tensor's original dimensions as mask dimensions
        original_height, original_width = tensor_height, tensor_width
        
        # Calculate scale ratio (handle size mismatch between PIL image and original tensor)
        pil_height, pil_width = img_np.shape[:2]
        scale_x = original_width / pil_width
        scale_y = original_height / pil_height
        
        # Select objects to process based on crop mode
        if crop_mode == "single_object":
            if object_index >= len(detections):
                print(f"Warning: Object index {object_index} out of range, using last object")
                object_index = len(detections) - 1
            selected_detections = [detections[object_index]]
        elif crop_mode == "by_class":
            # Group by class, select highest confidence for each class
            class_dict = {}
            for det in detections:
                cls = det['class']
                if cls not in class_dict or det['confidence'] > class_dict[cls]['confidence']:
                    class_dict[cls] = det
            selected_detections = list(class_dict.values())
        else:  # all_objects
            selected_detections = detections
        
        # Crop selected objects
        cropped_images = []
        bboxes = []
        detection_info = []
        
        # Create a combined mask with same size as original image
        combined_mask = torch.zeros((original_height, original_width), dtype=torch.float32)
        
        for i, det in enumerate(selected_detections):
            cropped, mask, bbox = self.crop_single_object(
                img_np, det, expand_width, expand_height, 
                original_height, original_width, scale_x, scale_y
            )
            
            # Ensure cropped image is valid
            if cropped.size == 0:
                print(f"Warning: Empty crop for object {i}, skipping")
                continue
            
            # Convert to tensor
            cropped_tensor = self.pil_to_tensor(Image.fromarray(cropped))
            mask_tensor = torch.from_numpy(mask).float()
            
            # Add batch dimension to cropped image
            cropped_images.append(cropped_tensor.unsqueeze(0))
            
            # Merge masks (use maximum to preserve all detected regions)
            combined_mask = torch.maximum(combined_mask, mask_tensor)
            
            bboxes.append(bbox)
            
            # Record detection info (show sorted index and position)
            center_x = int((det['bbox'][0] + det['bbox'][2]) / 2)
            center_y = int((det['bbox'][1] + det['bbox'][3]) / 2)
            info = f"[{i}] {det['class']} (confidence: {det['confidence']:.2f}, center: x={center_x},y={center_y})"
            detection_info.append(info)
        
        # Handle case where no valid crops were produced
        if not cropped_images:
            print("No valid crops produced")
            empty_mask = torch.zeros((1, tensor_height, tensor_width), dtype=torch.float32)
            empty_bboxes = torch.zeros((0, 4), dtype=torch.float32)
            return ([image], empty_mask, empty_bboxes, "No valid crops produced", 0)
        
        # Add batch dimension to combined mask
        # MASK format should be (batch, height, width), no channel dimension needed
        final_mask = combined_mask.unsqueeze(0)
        
        # Ensure mask data type and range are correct
        final_mask = final_mask.clamp(0.0, 1.0)
        
        # Convert bounding boxes to tensor
        final_bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4), dtype=torch.float32)
        
        # Generate detection info string
        info_str = f"Detected {len(detections)} objects, cropped {len(selected_detections)}\n"
        info_str += "\n".join(detection_info)
        
        # Return list of cropped images and combined mask
        return (cropped_images, final_mask, final_bboxes, info_str, len(selected_detections))

# Node registration
NODE_CLASS_MAPPINGS = {
    "YoloBboxesCropNode": YoloBboxesCropNode
}

# Node display name mapping
NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloBboxesCropNode": "🐳YOLO Detection Crop"
}