import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
from typing import Tuple, Optional
import os
import sys

# 导入ComfyUI相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class AutoCensorWithOpenPose:
    """
    使用OpenPose检测骨骼并自动打码的节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入需要处理的图像"}),
                "censor_face": ("BOOLEAN", {
                    "default": False,
                    "label_on": "打码脸部",
                    "label_off": "不打码脸部",
                    "tooltip": "是否对脸部进行打码"
                }),
                "censor_chest": ("BOOLEAN", {
                    "default": True,
                    "label_on": "打码胸部",
                    "label_off": "不打码胸部",
                    "tooltip": "是否对胸部区域进行打码"
                }),
                "censor_groin": ("BOOLEAN", {
                    "default": True,
                    "label_on": "打码腿根部",
                    "label_off": "不打码腿根部",
                    "tooltip": "是否对腿根部区域进行打码"
                }),
                "blur_strength": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "step": 5,
                    "display": "slider",
                    "tooltip": "模糊强度，数值越大模糊效果越强"
                }),
                "censor_size_multiplier": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.8,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "打码区域大小倍数，用于调整打码范围"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("打码后的图像",)
    FUNCTION = "process_image"
    CATEGORY = "🐳Pond/image"

    def __init__(self):
        # OpenPose关键点索引
        self.POSE_BODY_25_BODY_PARTS = {
            "Nose": 0,
            "Neck": 1,
            "RShoulder": 2,
            "RElbow": 3,
            "RWrist": 4,
            "LShoulder": 5,
            "LElbow": 6,
            "LWrist": 7,
            "MidHip": 8,
            "RHip": 9,
            "RKnee": 10,
            "RAnkle": 11,
            "LHip": 12,
            "LKnee": 13,
            "LAnkle": 14,
            "REye": 15,
            "LEye": 16,
            "REar": 17,
            "LEar": 18,
            "LBigToe": 19,
            "LSmallToe": 20,
            "LHeel": 21,
            "RBigToe": 22,
            "RSmallToe": 23,
            "RHeel": 24
        }

    def detect_openpose_keypoints(self, image_np):
        """
        使用MediaPipe检测人体关键点并映射到OpenPose格式
        """
        try:
            import mediapipe as mp
            
            # 初始化MediaPipe
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
            
            # 转换图像格式
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            h, w = image_np.shape[:2]
            keypoints = np.zeros((25, 3))  # OpenPose BODY_25格式
            
            if results.pose_landmarks:
                # MediaPipe到OpenPose的映射
                mp_to_op_mapping = {
                    0: 0,   # NOSE -> Nose
                    11: 5,  # LEFT_SHOULDER -> LShoulder
                    12: 2,  # RIGHT_SHOULDER -> RShoulder
                    23: 12, # LEFT_HIP -> LHip
                    24: 9,  # RIGHT_HIP -> RHip
                    25: 13, # LEFT_KNEE -> LKnee
                    26: 10, # RIGHT_KNEE -> RKnee
                    2: 16,  # LEFT_EYE -> LEye
                    5: 15,  # RIGHT_EYE -> REye
                    7: 18,  # LEFT_EAR -> LEar
                    8: 17,  # RIGHT_EAR -> REar
                }
                
                # 映射关键点
                for mp_idx, op_idx in mp_to_op_mapping.items():
                    if mp_idx < len(results.pose_landmarks.landmark):
                        landmark = results.pose_landmarks.landmark[mp_idx]
                        keypoints[op_idx] = [
                            landmark.x * w,
                            landmark.y * h,
                            landmark.visibility
                        ]
                
                # 计算额外的关键点
                # Neck (OpenPose idx 1) - 两肩膀中点
                if keypoints[2][2] > 0.1 and keypoints[5][2] > 0.1:
                    keypoints[1] = [
                        (keypoints[2][0] + keypoints[5][0]) / 2,
                        (keypoints[2][1] + keypoints[5][1]) / 2,
                        (keypoints[2][2] + keypoints[5][2]) / 2
                    ]
                
                # MidHip (OpenPose idx 8) - 两臀部中点
                if keypoints[9][2] > 0.1 and keypoints[12][2] > 0.1:
                    keypoints[8] = [
                        (keypoints[9][0] + keypoints[12][0]) / 2,
                        (keypoints[9][1] + keypoints[12][1]) / 2,
                        (keypoints[9][2] + keypoints[12][2]) / 2
                    ]
            
            pose.close()
            return keypoints
            
        except ImportError:
            print("MediaPipe未安装，使用模拟数据。请运行: pip install mediapipe")
            print("警告：当前使用模拟数据进行测试，实际使用时请安装MediaPipe")
            # 返回模拟数据用于测试
            h, w = image_np.shape[:2]
            keypoints = np.zeros((25, 3))
            # 模拟一些关键点用于测试
            keypoints[0] = [w/2, h*0.15, 0.9]  # Nose
            keypoints[1] = [w/2, h*0.2, 0.9]  # Neck
            keypoints[2] = [w*0.4, h*0.25, 0.9]  # RShoulder  
            keypoints[5] = [w*0.6, h*0.25, 0.9]  # LShoulder
            keypoints[8] = [w/2, h*0.5, 0.9]  # MidHip
            keypoints[9] = [w*0.45, h*0.5, 0.9]  # RHip
            keypoints[12] = [w*0.55, h*0.5, 0.9]  # LHip
            keypoints[15] = [w*0.45, h*0.12, 0.9]  # REye
            keypoints[16] = [w*0.55, h*0.12, 0.9]  # LEye
            return keypoints

    def get_face_region(self, keypoints, image_shape, size_multiplier):
        """
        根据面部关键点计算脸部区域
        """
        h, w = image_shape[:2]
        
        # 获取面部相关关键点
        nose = keypoints[self.POSE_BODY_25_BODY_PARTS["Nose"]]
        neck = keypoints[self.POSE_BODY_25_BODY_PARTS["Neck"]]
        left_eye = keypoints[self.POSE_BODY_25_BODY_PARTS["LEye"]]
        right_eye = keypoints[self.POSE_BODY_25_BODY_PARTS["REye"]]
        left_ear = keypoints[self.POSE_BODY_25_BODY_PARTS["LEar"]]
        right_ear = keypoints[self.POSE_BODY_25_BODY_PARTS["REar"]]
        
        # 检查关键点是否有效
        if nose[2] > 0.1:
            # 计算脸部中心（使用鼻子位置）
            face_center_x = nose[0]
            face_center_y = nose[1]
            
            # 估算脸部大小
            face_width = 0
            face_height = 0
            
            # 如果有眼睛关键点，使用眼睛间距估算宽度
            if left_eye[2] > 0.1 and right_eye[2] > 0.1:
                eye_distance = abs(right_eye[0] - left_eye[0])
                face_width = eye_distance * 2.5
            # 如果有耳朵关键点，使用耳朵间距
            elif left_ear[2] > 0.1 and right_ear[2] > 0.1:
                ear_distance = abs(right_ear[0] - left_ear[0])
                face_width = ear_distance * 1.2
            # 否则使用默认估算
            else:
                face_width = w * 0.15  # 假设脸宽约为图像宽度的15%
            
            # 如果有颈部关键点，使用它来估算高度
            if neck[2] > 0.1:
                face_height = abs(neck[1] - nose[1]) * 2.5
            else:
                face_height = face_width * 1.3  # 脸部高度通常是宽度的1.3倍
            
            # 应用大小倍数
            face_width *= size_multiplier
            face_height *= size_multiplier
            
            # 计算边界框，稍微向上偏移以更好地覆盖额头
            x1 = int(max(0, face_center_x - face_width/2))
            y1 = int(max(0, face_center_y - face_height/2 - face_height*0.2))
            x2 = int(min(w, face_center_x + face_width/2))
            y2 = int(min(h, face_center_y + face_height/2))
            
            return (x1, y1, x2, y2)
        
        return None

    def get_chest_region(self, keypoints, image_shape, size_multiplier):
        """
        根据肩膀和臀部关键点计算胸部区域
        """
        h, w = image_shape[:2]
        
        # 获取相关关键点
        left_shoulder = keypoints[self.POSE_BODY_25_BODY_PARTS["LShoulder"]]
        right_shoulder = keypoints[self.POSE_BODY_25_BODY_PARTS["RShoulder"]]
        mid_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["MidHip"]]
        neck = keypoints[self.POSE_BODY_25_BODY_PARTS["Neck"]]
        
        # 检查关键点是否有效
        if (left_shoulder[2] > 0.1 and right_shoulder[2] > 0.1 and 
            mid_hip[2] > 0.1 and neck[2] > 0.1):
            
            # 计算胸部中心位置
            chest_x = (left_shoulder[0] + right_shoulder[0]) / 2
            chest_y = (neck[1] + mid_hip[1]) / 2
            
            # 计算胸部区域大小
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            chest_height = abs(mid_hip[1] - neck[1]) * 0.4
            
            # 应用大小倍数
            width = shoulder_width * size_multiplier
            height = chest_height * size_multiplier
            
            # 计算边界框
            x1 = int(max(0, chest_x - width/2))
            y1 = int(max(0, chest_y - height/2))
            x2 = int(min(w, chest_x + width/2))
            y2 = int(min(h, chest_y + height/2))
            
            return (x1, y1, x2, y2)
        
        return None

    def get_groin_region(self, keypoints, image_shape, size_multiplier):
        """
        根据臀部和膝盖关键点计算腿根部区域
        """
        h, w = image_shape[:2]
        
        # 获取相关关键点
        mid_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["MidHip"]]
        left_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["LHip"]]
        right_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["RHip"]]
        left_knee = keypoints[self.POSE_BODY_25_BODY_PARTS["LKnee"]]
        right_knee = keypoints[self.POSE_BODY_25_BODY_PARTS["RKnee"]]
        
        # 检查关键点是否有效
        if (mid_hip[2] > 0.1 and left_hip[2] > 0.1 and right_hip[2] > 0.1):
            
            # 计算腿根部中心位置
            groin_x = mid_hip[0]
            groin_y = mid_hip[1]
            
            # 如果膝盖关键点有效，使用它们来调整位置
            if left_knee[2] > 0.1 and right_knee[2] > 0.1:
                knee_y = (left_knee[1] + right_knee[1]) / 2
                groin_y = (mid_hip[1] + knee_y) / 2
            
            # 计算腿根部区域大小
            hip_width = abs(right_hip[0] - left_hip[0])
            groin_height = hip_width * 0.8
            
            # 应用大小倍数
            width = hip_width * size_multiplier
            height = groin_height * size_multiplier
            
            # 计算边界框
            x1 = int(max(0, groin_x - width/2))
            y1 = int(max(0, groin_y - height/2))
            x2 = int(min(w, groin_x + width/2))
            y2 = int(min(h, groin_y + height/2))
            
            return (x1, y1, x2, y2)
        
        return None

    def apply_blur(self, image, region, blur_strength):
        """
        对指定区域应用模糊效果
        """
        if region is None:
            return image
        
        x1, y1, x2, y2 = region
        
        # 提取区域
        region_img = image[y1:y2, x1:x2]
        
        # 应用高斯模糊
        blurred_region = cv2.GaussianBlur(region_img, (blur_strength*2+1, blur_strength*2+1), 0)
        
        # 将模糊区域放回原图
        result = image.copy()
        result[y1:y2, x1:x2] = blurred_region
        
        return result

    def apply_elliptical_blur(self, image, region, blur_strength, is_face=False):
        """
        对指定区域应用椭圆形模糊效果（用于脸部等需要更自然效果的区域）
        """
        if region is None:
            return image
        
        x1, y1, x2, y2 = region
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # 创建椭圆形遮罩
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 255, -1)
        
        # 创建模糊版本
        blurred = cv2.GaussianBlur(image, (blur_strength*2+1, blur_strength*2+1), 0)
        
        # 使用遮罩混合原图和模糊图
        result = image.copy()
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (1 - mask_3channel) * image + mask_3channel * blurred
        
        return result.astype(np.uint8)

    def process_image(self, image, censor_face, censor_chest, censor_groin, blur_strength, censor_size_multiplier):
        """
        处理图像的主函数
        """
        # 参数验证
        if not censor_face and not censor_chest and not censor_groin:
            print("提示：未选择任何打码区域，将返回原图")
            return (image,)
        
        # 转换图像格式
        image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # 检测OpenPose关键点
        print("正在检测人体姿态...")
        keypoints = self.detect_openpose_keypoints(image_np)
        
        # 检查是否检测到人体
        valid_keypoints = np.sum(keypoints[:, 2] > 0.1)
        if valid_keypoints < 3:
            print("警告：未检测到有效的人体姿态，请确保图像中包含清晰的人体")
            return (image,)
        
        # 根据选项决定要打码的区域
        regions_to_blur = []
        face_region = None
        chest_region = None
        groin_region = None
        
        if censor_face:
            print("正在定位脸部区域...")
            face_region = self.get_face_region(keypoints, image_np.shape, censor_size_multiplier)
            if face_region:
                regions_to_blur.append(face_region)
                print("✓ 成功定位脸部区域")
            else:
                print("✗ 无法定位脸部区域")
        
        if censor_chest:
            print("正在定位胸部区域...")
            chest_region = self.get_chest_region(keypoints, image_np.shape, censor_size_multiplier)
            if chest_region:
                regions_to_blur.append(chest_region)
                print("✓ 成功定位胸部区域")
            else:
                print("✗ 无法定位胸部区域")
        
        if censor_groin:
            print("正在定位腿根部区域...")
            groin_region = self.get_groin_region(keypoints, image_np.shape, censor_size_multiplier)
            if groin_region:
                regions_to_blur.append(groin_region)
                print("✓ 成功定位腿根部区域")
            else:
                print("✗ 无法定位腿根部区域")
        
        # 如果没有找到任何区域
        if not regions_to_blur:
            print("警告：未能定位任何需要打码的区域")
            return (image,)
        
        # 应用模糊效果
        print(f"正在应用模糊效果（强度：{blur_strength}）...")
        result_image = image_np.copy()
        
        # 分别处理不同类型的区域
        region_index = 0
        
        # 处理脸部（使用椭圆形模糊）
        if censor_face and face_region:
            result_image = self.apply_elliptical_blur(result_image, face_region, blur_strength, is_face=True)
            region_index += 1
            print(f"✓ 已处理脸部区域（椭圆形模糊）")
        
        # 处理其他区域（使用矩形模糊）
        other_regions = []
        if censor_chest and chest_region:
            other_regions.append(chest_region)
        if censor_groin and groin_region:
            other_regions.append(groin_region)
        
        for region in other_regions:
            result_image = self.apply_blur(result_image, region, blur_strength)
            region_index += 1
            print(f"✓ 已处理区域 {region_index}/{len(regions_to_blur)}")
        
        # 转换回torch张量格式
        result_tensor = torch.from_numpy(result_image.astype(np.float32) / 255.0).unsqueeze(0)
        
        print("✓ 打码处理完成！")
        return (result_tensor,)


# 用于实际集成OpenPose的辅助类
class OpenPoseDetector:
    """
    实际使用时，这个类应该包含真正的OpenPose集成
    """
    def __init__(self):
        # 初始化OpenPose模型
        # 可以使用以下选项之一：
        # 1. openpose-python
        # 2. mediapipe
        # 3. mmpose
        # 4. 其他姿态估计库
        pass
    
    def detect(self, image):
        """
        检测图像中的人体关键点
        返回格式：numpy array of shape (25, 3) for BODY_25 model
        """
        # 实现实际的检测逻辑
        pass


# 节点注册
NODE_CLASS_MAPPINGS = {
    "AutoCensorWithOpenPose": AutoCensorWithOpenPose
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCensorWithOpenPose": "🐳自动打码"
}