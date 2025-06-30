import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
from typing import Tuple, Optional
import os
import sys

# Import ComfyUI related modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class AutoCensorWithOpenPose:
    """
    Node that uses OpenPose to detect skeleton and automatically censor
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to process"}),
                "censor_face": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Censor Face",
                    "label_off": "Don't Censor Face",
                    "tooltip": "Whether to censor face area"
                }),
                "censor_chest": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Censor Chest",
                    "label_off": "Don't Censor Chest",
                    "tooltip": "Whether to censor chest area"
                }),
                "censor_groin": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Censor Groin",
                    "label_off": "Don't Censor Groin",
                    "tooltip": "Whether to censor groin area"
                }),
                "blur_strength": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "step": 5,
                    "display": "slider",
                    "tooltip": "Blur strength, higher values mean stronger blur"
                }),
                "censor_size_multiplier": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.8,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Censor area size multiplier for adjusting coverage"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("censored_image",)
    FUNCTION = "process_image"
    CATEGORY = "🐳Pond/image"

    def __init__(self):
        # OpenPose keypoint indices
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
        Use MediaPipe to detect body keypoints and map to OpenPose format
        """
        try:
            import mediapipe as mp
            
            # Initialize MediaPipe
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
            
            # Convert image format
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            h, w = image_np.shape[:2]
            keypoints = np.zeros((25, 3))  # OpenPose BODY_25 format
            
            if results.pose_landmarks:
                # MediaPipe to OpenPose mapping
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
                
                # Map keypoints
                for mp_idx, op_idx in mp_to_op_mapping.items():
                    if mp_idx < len(results.pose_landmarks.landmark):
                        landmark = results.pose_landmarks.landmark[mp_idx]
                        keypoints[op_idx] = [
                            landmark.x * w,
                            landmark.y * h,
                            landmark.visibility
                        ]
                
                # Calculate additional keypoints
                # Neck (OpenPose idx 1) - midpoint between shoulders
                if keypoints[2][2] > 0.1 and keypoints[5][2] > 0.1:
                    keypoints[1] = [
                        (keypoints[2][0] + keypoints[5][0]) / 2,
                        (keypoints[2][1] + keypoints[5][1]) / 2,
                        (keypoints[2][2] + keypoints[5][2]) / 2
                    ]
                
                # MidHip (OpenPose idx 8) - midpoint between hips
                if keypoints[9][2] > 0.1 and keypoints[12][2] > 0.1:
                    keypoints[8] = [
                        (keypoints[9][0] + keypoints[12][0]) / 2,
                        (keypoints[9][1] + keypoints[12][1]) / 2,
                        (keypoints[9][2] + keypoints[12][2]) / 2
                    ]
            
            pose.close()
            return keypoints
            
        except ImportError:
            print("MediaPipe not installed, using simulated data. Please run: pip install mediapipe")
            print("Warning: Currently using simulated data for testing, please install MediaPipe for actual use")
            # Return simulated data for testing
            h, w = image_np.shape[:2]
            keypoints = np.zeros((25, 3))
            # Simulate some keypoints for testing
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
        Calculate face region based on facial keypoints
        """
        h, w = image_shape[:2]
        
        # Get face-related keypoints
        nose = keypoints[self.POSE_BODY_25_BODY_PARTS["Nose"]]
        neck = keypoints[self.POSE_BODY_25_BODY_PARTS["Neck"]]
        left_eye = keypoints[self.POSE_BODY_25_BODY_PARTS["LEye"]]
        right_eye = keypoints[self.POSE_BODY_25_BODY_PARTS["REye"]]
        left_ear = keypoints[self.POSE_BODY_25_BODY_PARTS["LEar"]]
        right_ear = keypoints[self.POSE_BODY_25_BODY_PARTS["REar"]]
        
        # Check if keypoints are valid
        if nose[2] > 0.1:
            # Calculate face center (using nose position)
            face_center_x = nose[0]
            face_center_y = nose[1]
            
            # Estimate face size
            face_width = 0
            face_height = 0
            
            # If we have eye keypoints, use eye distance to estimate width
            if left_eye[2] > 0.1 and right_eye[2] > 0.1:
                eye_distance = abs(right_eye[0] - left_eye[0])
                face_width = eye_distance * 2.5
            # If we have ear keypoints, use ear distance
            elif left_ear[2] > 0.1 and right_ear[2] > 0.1:
                ear_distance = abs(right_ear[0] - left_ear[0])
                face_width = ear_distance * 1.2
            # Otherwise use default estimation
            else:
                face_width = w * 0.15  # Assume face width is about 15% of image width
            
            # If we have neck keypoint, use it to estimate height
            if neck[2] > 0.1:
                face_height = abs(neck[1] - nose[1]) * 2.5
            else:
                face_height = face_width * 1.3  # Face height is usually 1.3 times the width
            
            # Apply size multiplier
            face_width *= size_multiplier
            face_height *= size_multiplier
            
            # Calculate bounding box, slightly shift upward to better cover forehead
            x1 = int(max(0, face_center_x - face_width/2))
            y1 = int(max(0, face_center_y - face_height/2 - face_height*0.2))
            x2 = int(min(w, face_center_x + face_width/2))
            y2 = int(min(h, face_center_y + face_height/2))
            
            return (x1, y1, x2, y2)
        
        return None

    def get_chest_region(self, keypoints, image_shape, size_multiplier):
        """
        Calculate chest region based on shoulder and hip keypoints
        """
        h, w = image_shape[:2]
        
        # Get relevant keypoints
        left_shoulder = keypoints[self.POSE_BODY_25_BODY_PARTS["LShoulder"]]
        right_shoulder = keypoints[self.POSE_BODY_25_BODY_PARTS["RShoulder"]]
        mid_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["MidHip"]]
        neck = keypoints[self.POSE_BODY_25_BODY_PARTS["Neck"]]
        
        # Check if keypoints are valid
        if (left_shoulder[2] > 0.1 and right_shoulder[2] > 0.1 and 
            mid_hip[2] > 0.1 and neck[2] > 0.1):
            
            # Calculate chest center position
            chest_x = (left_shoulder[0] + right_shoulder[0]) / 2
            chest_y = (neck[1] + mid_hip[1]) / 2
            
            # Calculate chest area size
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            chest_height = abs(mid_hip[1] - neck[1]) * 0.4
            
            # Apply size multiplier
            width = shoulder_width * size_multiplier
            height = chest_height * size_multiplier
            
            # Calculate bounding box
            x1 = int(max(0, chest_x - width/2))
            y1 = int(max(0, chest_y - height/2))
            x2 = int(min(w, chest_x + width/2))
            y2 = int(min(h, chest_y + height/2))
            
            return (x1, y1, x2, y2)
        
        return None

    def get_groin_region(self, keypoints, image_shape, size_multiplier):
        """
        Calculate groin region based on hip and knee keypoints
        """
        h, w = image_shape[:2]
        
        # Get relevant keypoints
        mid_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["MidHip"]]
        left_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["LHip"]]
        right_hip = keypoints[self.POSE_BODY_25_BODY_PARTS["RHip"]]
        left_knee = keypoints[self.POSE_BODY_25_BODY_PARTS["LKnee"]]
        right_knee = keypoints[self.POSE_BODY_25_BODY_PARTS["RKnee"]]
        
        # Check if keypoints are valid
        if (mid_hip[2] > 0.1 and left_hip[2] > 0.1 and right_hip[2] > 0.1):
            
            # Calculate groin center position
            groin_x = mid_hip[0]
            groin_y = mid_hip[1]
            
            # If knee keypoints are valid, use them to adjust position
            if left_knee[2] > 0.1 and right_knee[2] > 0.1:
                knee_y = (left_knee[1] + right_knee[1]) / 2
                groin_y = (mid_hip[1] + knee_y) / 2
            
            # Calculate groin area size
            hip_width = abs(right_hip[0] - left_hip[0])
            groin_height = hip_width * 0.8
            
            # Apply size multiplier
            width = hip_width * size_multiplier
            height = groin_height * size_multiplier
            
            # Calculate bounding box
            x1 = int(max(0, groin_x - width/2))
            y1 = int(max(0, groin_y - height/2))
            x2 = int(min(w, groin_x + width/2))
            y2 = int(min(h, groin_y + height/2))
            
            return (x1, y1, x2, y2)
        
        return None

    def apply_blur(self, image, region, blur_strength):
        """
        Apply blur effect to specified region
        """
        if region is None:
            return image
        
        x1, y1, x2, y2 = region
        
        # Extract region
        region_img = image[y1:y2, x1:x2]
        
        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(region_img, (blur_strength*2+1, blur_strength*2+1), 0)
        
        # Put blurred region back into original image
        result = image.copy()
        result[y1:y2, x1:x2] = blurred_region
        
        return result

    def apply_elliptical_blur(self, image, region, blur_strength, is_face=False):
        """
        Apply elliptical blur effect to specified region (for more natural effect on face etc.)
        """
        if region is None:
            return image
        
        x1, y1, x2, y2 = region
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Create elliptical mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 255, -1)
        
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (blur_strength*2+1, blur_strength*2+1), 0)
        
        # Use mask to blend original and blurred images
        result = image.copy()
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (1 - mask_3channel) * image + mask_3channel * blurred
        
        return result.astype(np.uint8)

    def process_image(self, image, censor_face, censor_chest, censor_groin, blur_strength, censor_size_multiplier):
        """
        Main function to process image
        """
        # Parameter validation
        if not censor_face and not censor_chest and not censor_groin:
            print("Note: No censor areas selected, returning original image")
            return (image,)
        
        # Convert image format
        image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # Detect OpenPose keypoints
        print("Detecting body pose...")
        keypoints = self.detect_openpose_keypoints(image_np)
        
        # Check if body is detected
        valid_keypoints = np.sum(keypoints[:, 2] > 0.1)
        if valid_keypoints < 3:
            print("Warning: No valid body pose detected, please ensure image contains clear human body")
            return (image,)
        
        # Determine regions to censor based on options
        regions_to_blur = []
        face_region = None
        chest_region = None
        groin_region = None
        
        if censor_face:
            print("Locating face area...")
            face_region = self.get_face_region(keypoints, image_np.shape, censor_size_multiplier)
            if face_region:
                regions_to_blur.append(face_region)
                print("✓ Successfully located face area")
            else:
                print("✗ Unable to locate face area")
        
        if censor_chest:
            print("Locating chest area...")
            chest_region = self.get_chest_region(keypoints, image_np.shape, censor_size_multiplier)
            if chest_region:
                regions_to_blur.append(chest_region)
                print("✓ Successfully located chest area")
            else:
                print("✗ Unable to locate chest area")
        
        if censor_groin:
            print("Locating groin area...")
            groin_region = self.get_groin_region(keypoints, image_np.shape, censor_size_multiplier)
            if groin_region:
                regions_to_blur.append(groin_region)
                print("✓ Successfully located groin area")
            else:
                print("✗ Unable to locate groin area")
        
        # If no regions found
        if not regions_to_blur:
            print("Warning: Unable to locate any areas to censor")
            return (image,)
        
        # Apply blur effect
        print(f"Applying blur effect (strength: {blur_strength})...")
        result_image = image_np.copy()
        
        # Process different types of regions separately
        region_index = 0
        
        # Process face (using elliptical blur)
        if censor_face and face_region:
            result_image = self.apply_elliptical_blur(result_image, face_region, blur_strength, is_face=True)
            region_index += 1
            print(f"✓ Processed face area (elliptical blur)")
        
        # Process other regions (using rectangular blur)
        other_regions = []
        if censor_chest and chest_region:
            other_regions.append(chest_region)
        if censor_groin and groin_region:
            other_regions.append(groin_region)
        
        for region in other_regions:
            result_image = self.apply_blur(result_image, region, blur_strength)
            region_index += 1
            print(f"✓ Processed region {region_index}/{len(regions_to_blur)}")
        
        # Convert back to torch tensor format
        result_tensor = torch.from_numpy(result_image.astype(np.float32) / 255.0).unsqueeze(0)
        
        print("✓ Censoring process complete!")
        return (result_tensor,)


# Helper class for actual OpenPose integration
class OpenPoseDetector:
    """
    For actual use, this class should contain real OpenPose integration
    """
    def __init__(self):
        # Initialize OpenPose model
        # Can use one of the following options:
        # 1. openpose-python
        # 2. mediapipe
        # 3. mmpose
        # 4. Other pose estimation libraries
        pass
    
    def detect(self, image):
        """
        Detect body keypoints in image
        Return format: numpy array of shape (25, 3) for BODY_25 model
        """
        # Implement actual detection logic
        pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "AutoCensorWithOpenPose": AutoCensorWithOpenPose
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCensorWithOpenPose": "🐳Auto Censor"
}