import numpy as np
import torch
import cv2
from PIL import Image
import os

class VideoFrameExtractor:
    """
    A ComfyUI node to extract a specific frame from a video file
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),  # ComfyUI的视频加载器输出的是IMAGE序列
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "total_frames": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_frame"
    CATEGORY = "🐳Pond/video"
    
    def extract_frame(self, video, frame_index, total_frames=-1):
        """
        Extract a specific frame from the video
        
        Args:
            video: Video frames as IMAGE tensor (B, H, W, C)
            frame_index: Index of the frame to extract
            total_frames: Optional limit on total frames to consider
        
        Returns:
            Single frame as IMAGE tensor
        """
        # 获取视频的总帧数
        video_frames = video.shape[0]
        
        # 如果设置了总帧数限制，使用较小的值
        if total_frames > 0:
            video_frames = min(video_frames, total_frames)
        
        # 确保帧索引在有效范围内
        if frame_index >= video_frames:
            print(f"Warning: Frame index {frame_index} exceeds video length {video_frames}. Using last frame.")
            frame_index = video_frames - 1
        
        # 提取指定帧
        extracted_frame = video[frame_index:frame_index+1]
        
        return (extracted_frame,)


class VideoFrameExtractorAdvanced:
    """
    Advanced video frame extractor with more options
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "mode": (["index", "percentage", "time"],),
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 9999.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("image", "frame_index",)
    FUNCTION = "extract_frame_advanced"
    CATEGORY = "🐳Pond/video"
    
    def extract_frame_advanced(self, video, mode, value, fps=30.0):
        """
        Extract frame based on different modes
        
        Args:
            video: Video frames as IMAGE tensor
            mode: Extraction mode (index, percentage, time)
            value: Value based on mode
            fps: Frames per second (for time mode)
        
        Returns:
            Extracted frame and its index
        """
        video_frames = video.shape[0]
        
        # 根据模式计算帧索引
        if mode == "index":
            frame_index = int(value)
        elif mode == "percentage":
            # 百分比模式：0-100%
            percentage = min(max(value, 0), 100)
            frame_index = int((percentage / 100.0) * (video_frames - 1))
        elif mode == "time":
            # 时间模式：根据秒数和FPS计算
            frame_index = int(value * fps)
        else:
            frame_index = 0
        
        # 确保帧索引在有效范围内
        frame_index = min(max(frame_index, 0), video_frames - 1)
        
        # 提取帧
        extracted_frame = video[frame_index:frame_index+1]
        
        return (extracted_frame, frame_index)


class VideoFrameRangeExtractor:
    """
    Extract a range of frames from video
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "end_frame": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "extract_range"
    CATEGORY = "🐳Pond/video"
    
    def extract_range(self, video, start_frame, end_frame, step):
        """
        Extract a range of frames with optional step
        
        Args:
            video: Video frames as IMAGE tensor
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)
            step: Step between frames
        
        Returns:
            Selected frames as IMAGE tensor
        """
        video_frames = video.shape[0]
        
        # 确保索引在有效范围内
        start_frame = max(0, min(start_frame, video_frames - 1))
        end_frame = max(start_frame, min(end_frame, video_frames - 1))
        
        # 提取帧范围
        indices = list(range(start_frame, end_frame + 1, step))
        if not indices:
            indices = [start_frame]
        
        # 收集选定的帧
        selected_frames = []
        for idx in indices:
            if idx < video_frames:
                selected_frames.append(video[idx])
        
        # 堆叠帧
        if selected_frames:
            result = torch.stack(selected_frames, dim=0)
        else:
            result = video[0:1]  # 返回第一帧作为默认
        
        return (result,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoFrameExtractor": VideoFrameExtractor,
    "VideoFrameExtractorAdvanced": VideoFrameExtractorAdvanced,
    "VideoFrameRangeExtractor": VideoFrameRangeExtractor,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameExtractor": "🐳Video Frame Extractor",
    "VideoFrameExtractorAdvanced": "🐳Video Frame Extractor (Advanced)",
    "VideoFrameRangeExtractor": "🐳Video Frame Range Extractor",
}