import torch
import os
import folder_paths
from PIL import Image
import numpy as np
from transformers.generation import GenerationConfig
import gc
import json

# 尝试导入不同版本的类
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    QWEN2VL_AVAILABLE = True
except ImportError:
    QWEN2VL_AVAILABLE = False
    print("[Qwen Image Captioner] Qwen2VL not available, trying Qwen2_5VL...")

try:
    from transformers import Qwen2_5VLForConditionalGeneration, AutoProcessor as Qwen2_5VLProcessor
    QWEN2_5VL_AVAILABLE = True
except ImportError:
    QWEN2_5VL_AVAILABLE = False
    print("[Qwen Image Captioner] Qwen2_5VL not available...")

# 基础类导入
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


class QwenImageCaptioner:
    """
    A ComfyUI node for generating image captions/prompts using Qwen vision-language models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取Qwen模型目录
        qwen_models_dir = os.path.join(folder_paths.models_dir, "Qwen")
        available_models = cls._scan_available_models(qwen_models_dir)
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (available_models,),
                "prompt_type": (["detailed", "brief", "technical", "artistic", "custom"],),
                "language": (["English", "中文"],),
                "device": (["auto", "cuda", "cpu"],),
                "max_length": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 2048,
                    "step": 32
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "auto_unload": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto Unload",
                    "label_off": "Keep Loaded"
                }),
            },
            "optional": {
                "custom_instruction": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail for use as a prompt in image generation."
                }),
                "max_image_size": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 128,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "🐳Pond/Qwen"
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model = None
        self.current_device = None
    
    @staticmethod
    def _scan_available_models(qwen_models_dir):
        """扫描可用的Qwen VL模型"""
        available_models = []
        
        if not os.path.exists(qwen_models_dir):
            return ["No models found in models/Qwen/"]
        
        for item in os.listdir(qwen_models_dir):
            model_path = os.path.join(qwen_models_dir, item)
            
            if not os.path.isdir(model_path):
                continue
                
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                continue
            
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                
                # 检查是否是VL模型
                if "vl" in model_type or "VL" in item:
                    available_models.append(item)
                else:
                    available_models.append(f"[非VL模型] {item}")
            except:
                available_models.append(item)
        
        return available_models if available_models else ["No models found in models/Qwen/"]
    
    def _determine_device(self, device):
        """确定使用的设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _cleanup_previous_model(self):
        """清理之前加载的模型"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.current_model = None
        self.current_device = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Qwen Image Captioner] Model unloaded and memory cleared")
    
    def _get_model_config(self, model_path):
        """读取模型配置"""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _prepare_model_kwargs(self, device):
        """准备模型加载参数"""
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # 设置设备映射
        model_kwargs["device_map"] = device
        
        # 设置精度 - 简化版本，自动选择最佳精度
        if device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
            print("[Qwen Image Captioner] Using FP32 for CPU")
        else:
            # GPU上优先使用bf16，否则使用fp16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
                print("[Qwen Image Captioner] Using BF16 for GPU")
            else:
                model_kwargs["torch_dtype"] = torch.float16
                print("[Qwen Image Captioner] Using FP16 for GPU")
        
        return model_kwargs
    
    def _load_qwen2_5_vl(self, model_path, model_kwargs):
        """加载Qwen2.5-VL模型"""
        if not QWEN2_5VL_AVAILABLE:
            return False
            
        try:
            print("[Qwen Image Captioner] Loading as Qwen2.5-VL model...")
            self.processor = Qwen2_5VLProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs
            ).eval()
            return True
        except Exception as e:
            print(f"[Qwen Image Captioner] Failed to load as Qwen2.5-VL: {e}")
            return False
    
    def _load_qwen2_vl(self, model_path, model_kwargs):
        """加载Qwen2-VL模型"""
        if not QWEN2VL_AVAILABLE:
            return False
            
        try:
            print("[Qwen Image Captioner] Loading as Qwen2-VL model...")
            self.processor = Qwen2VLProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs
            ).eval()
            return True
        except Exception as e:
            print(f"[Qwen Image Captioner] Failed to load as Qwen2-VL: {e}")
            return False
    
    def _load_auto_model(self, model_path, model_kwargs, model_type):
        """使用AutoModel加载模型"""
        try:
            print("[Qwen Image Captioner] Loading with AutoModel...")
            
            if "qwen2_5_vl" in model_type:
                # 使用AutoModel的from_pretrained
                from transformers import AutoModelForVision2Seq
                try:
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_path,
                        **model_kwargs
                    ).eval()
                except:
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(
                        model_path,
                        **model_kwargs
                    ).eval()
                
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            else:
                # 旧版Qwen-VL
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                ).eval()
                
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                except:
                    self.processor = None
            
            return True
        except Exception as e:
            print(f"[Qwen Image Captioner] Failed to load with AutoModel: {e}")
            return False
    
    def load_model(self, model_name, device="auto"):
        """加载Qwen视觉语言模型"""
        # 检查是否是不支持的格式
        if "[非VL模型]" in model_name:
            raise ValueError(
                "这不是一个视觉语言(VL)模型！\n"
                "图像描述需要使用包含视觉编码器的VL模型。\n"
                "请下载Qwen-VL或Qwen2-VL系列模型。"
            )
        
        # 确定设备
        self.device = self._determine_device(device)
        
        # 检查是否需要重新加载
        if self.current_model == model_name and self.current_device == device:
            print(f"[Qwen Image Captioner] Model {model_name} already loaded on {device}")
            return
        
        # 清理之前的模型
        self._cleanup_previous_model()
        
        # 模型路径
        model_path = os.path.join(folder_paths.models_dir, "Qwen", model_name)
        
        try:
            # 读取配置
            config = self._get_model_config(model_path)
            model_type = config.get("model_type", "").lower()
            architectures = config.get("architectures", [])
            
            print(f"[Qwen Image Captioner] Model type: {model_type}")
            print(f"[Qwen Image Captioner] Architectures: {architectures}")
            
            # 准备模型加载参数
            model_kwargs = self._prepare_model_kwargs(self.device)
            
            # 按优先级尝试不同的加载方式
            loaded = False
            
            # 1. 尝试Qwen2.5-VL
            if "qwen2_5_vl" in model_type:
                loaded = self._load_qwen2_5_vl(model_path, model_kwargs)
            
            # 2. 尝试Qwen2-VL
            if not loaded and ("qwen2_vl" in model_type or "Qwen2VL" in str(architectures)):
                loaded = self._load_qwen2_vl(model_path, model_kwargs)
            
            # 3. 尝试使用AutoModel
            if not loaded:
                loaded = self._load_auto_model(model_path, model_kwargs, model_type)
            
            if not loaded:
                raise ValueError(f"无法加载模型 {model_name}。请确保模型格式正确。")
            
            self.current_model = model_name
            self.current_device = device
            print(f"[Qwen Image Captioner] Successfully loaded model: {model_name}")
            
        except Exception as e:
            print(f"[Qwen Image Captioner] Error loading model {model_name}: {str(e)}")
            raise
    
    def _resize_image(self, pil_image, max_size=1024):
        """等比例缩放图像到指定最大尺寸"""
        width, height = pil_image.size
        
        # 检查是否需要缩放
        if width <= max_size and height <= max_size:
            print(f"[Qwen Image Captioner] Image size ({width}x{height}) is within limit, no resizing needed")
            return pil_image
        
        # 计算缩放比例
        if width > height:
            scale = max_size / width
        else:
            scale = max_size / height
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 使用高质量的重采样方法
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"[Qwen Image Captioner] Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return resized_image
    
    def _prepare_image(self, image, max_size=1024):
        """准备图像输入，包括格式转换和尺寸调整"""
        if isinstance(image, torch.Tensor):
            # ComfyUI图像格式: [batch, height, width, channels]
            if len(image.shape) == 4:
                image = image[0]  # 取第一张图
            # 转换为PIL Image
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
        else:
            pil_image = image
        
        # 等比例缩放图像
        pil_image = self._resize_image(pil_image, max_size)
        
        return pil_image
    
    def _get_instruction(self, prompt_type, language, custom_instruction):
        """根据类型和语言生成指令"""
        instructions = {
            "中文": {
                "detailed": "请详细描述这张图片，包括所有的物体、人物、颜色、纹理、构图、光线和氛围。请具体说明空间关系和视觉元素。",
                "brief": "请简洁地描述图片的主要主题和关键元素。",
                "technical": "请从技术角度描述这张图片：构图、光线、色彩搭配、拍摄角度、景深以及任何后期处理效果。",
                "artistic": "请从艺术角度描述这张图片，重点关注艺术风格、情绪、美学特质和情感影响。包括艺术技巧和视觉叙事的细节。",
                "custom": custom_instruction if custom_instruction else "请描述这张图片。"
            },
            "English": {
                "detailed": "Describe this image in great detail, including all objects, people, colors, textures, composition, lighting, and atmosphere. Be specific about spatial relationships and visual elements.",
                "brief": "Provide a concise description of the main subject and key elements in this image.",
                "technical": "Describe this image focusing on technical aspects: composition, lighting, color palette, camera angle, depth of field, and any post-processing effects.",
                "artistic": "Describe this image with focus on artistic style, mood, aesthetic qualities, and emotional impact. Include details about artistic techniques and visual storytelling.",
                "custom": custom_instruction if custom_instruction else "Describe this image."
            }
        }
        
        return instructions[language][prompt_type]
    
    def _generate_with_qwen2_vl(self, pil_image, instruction, max_length, temperature):
        """使用Qwen2-VL格式生成描述"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=temperature > 0.1,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=generation_config)
        
        return outputs
    
    def _generate_with_qwen_vl(self, pil_image, instruction, max_length, temperature):
        """使用旧版Qwen-VL格式生成描述"""
        query = self.tokenizer.from_list_format([
            {'image': pil_image},
            {'text': instruction}
        ])
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=temperature > 0.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=generation_config)
        
        return outputs
    
    def _decode_output(self, outputs, instruction):
        """解码模型输出"""
        if hasattr(outputs, 'cpu'):
            outputs = outputs.cpu()
        
        # 使用正确的解码器
        if self.processor and hasattr(self.processor, 'decode'):
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
        elif self.tokenizer:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的文本（去除输入部分）
        if instruction in response:
            caption = response.split(instruction)[-1].strip()
        else:
            caption = response.split('\n')[-1].strip() if '\n' in response else response
        
        return caption
    
    def generate_caption(self, image, model_name, prompt_type, language, device, 
                        max_length, temperature, auto_unload=True, 
                        custom_instruction="", max_image_size=1024):
        """生成图像描述"""
        
        try:
            # 加载模型
            self.load_model(model_name, device)
            
            # 准备图像（包括缩放）
            pil_image = self._prepare_image(image, max_image_size)
            
            # 获取指令
            instruction = self._get_instruction(prompt_type, language, custom_instruction)
            
            # 根据不同的模型类型使用不同的生成方法
            if self.processor and hasattr(self.processor, 'apply_chat_template'):
                # Qwen2-VL格式
                outputs = self._generate_with_qwen2_vl(pil_image, instruction, max_length, temperature)
            elif self.tokenizer and hasattr(self.tokenizer, 'from_list_format'):
                # 旧版Qwen-VL格式
                outputs = self._generate_with_qwen_vl(pil_image, instruction, max_length, temperature)
            else:
                # 通用格式（简化处理）
                raise NotImplementedError("Unsupported model format")
            
            # 解码输出
            caption = self._decode_output(outputs, instruction)
            
            print(f"[Qwen Image Captioner] Caption generated successfully")
            
            return (caption,)
            
        except Exception as e:
            error_msg = f"Error generating caption: {str(e)}"
            print(f"[Qwen Image Captioner] {error_msg}")
            
            # 提供调试建议
            if "CUDA" in str(e):
                print("[Qwen Image Captioner] CUDA error detected. Suggestions:")
                print("1. Try using CPU instead of GPU")
                print("2. Try reducing max_length")
                print("3. Check GPU memory usage")
            
            return (error_msg,)
        
        finally:
            # 如果启用了自动卸载，在生成完成后卸载模型
            if auto_unload:
                print("[Qwen Image Captioner] Auto-unloading model after generation...")
                self._cleanup_previous_model()


# 节点映射
NODE_CLASS_MAPPINGS = {
    "QwenImageCaptioner": QwenImageCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageCaptioner": "🐳Qwen Image Captioner",
}