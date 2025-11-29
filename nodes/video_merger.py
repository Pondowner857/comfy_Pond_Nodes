import torch

class VideoMerger:

    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("IMAGE",),
                "video_2": ("IMAGE",),
                "input_count": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
                "remove_last_frame": ("BOOLEAN", {
                    "default": True,
                    "label_on": "删除最后帧",
                    "label_off": "保留最后帧"
                }),
            },
            "optional": {
                "video_3": ("IMAGE",),
                "video_4": ("IMAGE",),
                "video_5": ("IMAGE",),
                "video_6": ("IMAGE",),
                "video_7": ("IMAGE",),
                "video_8": ("IMAGE",),
                "video_9": ("IMAGE",),
                "video_10": ("IMAGE",),
                "video_11": ("IMAGE",),
                "video_12": ("IMAGE",),
                "video_13": ("IMAGE",),
                "video_14": ("IMAGE",),
                "video_15": ("IMAGE",),
                "video_16": ("IMAGE",),
                "video_17": ("IMAGE",),
                "video_18": ("IMAGE",),
                "video_19": ("IMAGE",),
                "video_20": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_video",)
    FUNCTION = "merge_videos"
    CATEGORY = "🐳Pond/video"
    OUTPUT_NODE = False
    
    def merge_videos(self, input_count, remove_last_frame, **kwargs):

        # 收集所有有效的视频输入
        videos = []
        for i in range(1, input_count + 1):
            video_key = f"video_{i}"
            if video_key in kwargs and kwargs[video_key] is not None:
                videos.append(kwargs[video_key])
            else:
                print(f"[VideoMerger] 警告: {video_key} 未连接或为空")
        
        if len(videos) == 0:
            print("[VideoMerger] 错误: 没有有效的视频输入")
            # 返回一个空的图像张量
            return (torch.zeros((1, 64, 64, 3)),)
        
        if len(videos) == 1:
            print("[VideoMerger] 只有一个视频输入，直接返回")
            return (videos[0],)
        
        # 处理每个视频
        processed_videos = []
        for idx, video in enumerate(videos):
            # 检查是否是最后一个视频
            is_last_video = (idx == len(videos) - 1)
            
            if remove_last_frame and not is_last_video:
                # 删除最后一帧
                if video.shape[0] > 1:  # 确保至少有2帧
                    processed_video = video[:-1]
                    print(f"[VideoMerger] 视频{idx+1}: 删除最后一帧，剩余 {processed_video.shape[0]} 帧")
                else:
                    processed_video = video
                    print(f"[VideoMerger] 视频{idx+1}: 只有1帧，保留")
            else:
                processed_video = video
                if is_last_video:
                    print(f"[VideoMerger] 视频{idx+1}: 最后一个视频，保留所有 {video.shape[0]} 帧")
                else:
                    print(f"[VideoMerger] 视频{idx+1}: 保留所有 {video.shape[0]} 帧")
            
            processed_videos.append(processed_video)
        
        # 合并所有视频
        try:
            merged_video = torch.cat(processed_videos, dim=0)
            total_frames = merged_video.shape[0]
            print(f"[VideoMerger] ✅ 成功合并 {len(videos)} 个视频，总帧数: {total_frames}")
            
            # 打印每个视频的帧数信息
            frame_info = " + ".join([str(v.shape[0]) for v in processed_videos])
            print(f"[VideoMerger] 帧数详情: {frame_info} = {total_frames}")
            
            return (merged_video,)
            
        except Exception as e:
            print(f"[VideoMerger] ❌ 合并失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回第一个视频作为fallback
            return (videos[0],)


NODE_CLASS_MAPPINGS = {
    "VideoMerger": VideoMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoMerger": "🐳视频合并"
}




