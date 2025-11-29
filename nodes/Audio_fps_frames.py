import torch

class AudioFrameCalculator:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("fps", "num_frames")
    FUNCTION = "calculate_frames"
    CATEGORY = "🐳Pond/audio"
    OUTPUT_NODE = False
    
    def calculate_frames(self, audio, fps):

        try:
            # 获取音频波形和采样率
            waveform = audio['waveform']  # shape: [batch, channels, samples]
            sample_rate = audio['sample_rate']
            
            # 获取音频样本数（取第一个batch的数据）
            num_samples = waveform.shape[2]
            
            # 计算音频时长（秒）
            duration = num_samples / sample_rate
            
            # 计算总帧数
            num_frames = int(duration * fps)            
            
            return (fps, num_frames)
            
        except KeyError as e:
            # 返回默认值
            return (fps, 1)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 返回默认值
            return (fps, 1)


NODE_CLASS_MAPPINGS = {
    "AudioFrameCalculator": AudioFrameCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFrameCalculator": "🐳音频帧数计算"
}




