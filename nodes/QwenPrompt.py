import torch
import os
import folder_paths
from PIL import Image
import numpy as np
from transformers.generation import GenerationConfig
import gc
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from typing import Optional, Dict, Any, Tuple
from functools import partial
import asyncio
from pathlib import Path

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

# 检测FlashAttention2是否可用
FLASH_ATTENTION_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    print("[Qwen Image Captioner] FlashAttention2 is available ✓")
except ImportError:
    print("[Qwen Image Captioner] FlashAttention2 not available - Install with: pip install flash-attn")

# 尝试导入加速库
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    ACCELERATE_AVAILABLE = True
    print("[Qwen Image Captioner] Accelerate is available ✓")
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("[Qwen Image Captioner] Accelerate not available - Install for faster loading: pip install accelerate")


class FastModelLoader:
    """优化的模型加载器，支持多线程和并行加载"""
    
    @staticmethod
    def parallel_load_safetensors(model_path, device, dtype):
        """并行加载safetensors格式的模型权重"""
        try:
            import safetensors.torch
            from safetensors import safe_open
            
            # 查找所有的safetensors文件
            safetensor_files = list(Path(model_path).glob("*.safetensors"))
            if not safetensor_files:
                return None
                
            print(f"[Qwen Image Captioner] Found {len(safetensor_files)} safetensor files")
            
            # 使用多线程并行加载
            state_dict = {}
            
            def load_single_file(file_path):
                with safe_open(file_path, framework="pt", device=str(device)) as f:
                    return {key: f.get_tensor(key) for key in f.keys()}
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(load_single_file, f) for f in safetensor_files]
                for future in as_completed(futures):
                    state_dict.update(future.result())
            
            return state_dict
        except Exception as e:
            print(f"[Qwen Image Captioner] Safetensors loading failed: {e}")
            return None
    
    @staticmethod
    def optimize_model_loading_params(device):
        """获取优化的模型加载参数"""
        params = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
        }
        
        # 设备配置
        if device == "cuda" and torch.cuda.is_available():
            params["device_map"] = "auto"  # 自动设备映射
            
            # 选择最佳精度
            if torch.cuda.is_bf16_supported():
                params["torch_dtype"] = torch.bfloat16
                print("[Qwen Image Captioner] Using BF16 precision")
            else:
                params["torch_dtype"] = torch.float16
                print("[Qwen Image Captioner] Using FP16 precision")
            
            # FlashAttention2配置
            if FLASH_ATTENTION_AVAILABLE:
                params["attn_implementation"] = "flash_attention_2"
                print("[Qwen Image Captioner] FlashAttention2 enabled ✓")
            else:
                params["attn_implementation"] = "sdpa"
                print("[Qwen Image Captioner] Using SDPA (FlashAttention2 not available)")
        else:
            params["device_map"] = "cpu"
            params["torch_dtype"] = torch.float32
            print("[Qwen Image Captioner] Using CPU with FP32")
        
        return params
    
    @staticmethod
    def load_with_accelerate(model_class, model_path, device, dtype):
        """使用accelerate库加速加载"""
        if not ACCELERATE_AVAILABLE:
            return None
            
        try:
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            
            print("[Qwen Image Captioner] Loading with Accelerate...")
            
            # 初始化空权重模型
            with init_empty_weights():
                model = model_class.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype
                )
            
            # 并行加载权重
            model = load_checkpoint_and_dispatch(
                model,
                model_path,
                device_map="auto" if device == "cuda" else "cpu",
                no_split_module_classes=["Qwen2VLDecoderLayer"],
                dtype=dtype
            )
            
            return model
        except Exception as e:
            print(f"[Qwen Image Captioner] Accelerate loading failed: {e}")
            return None


class QwenImageCaptioner:
    """
    优化版的ComfyUI节点，专注于FlashAttention2加速和快速模型加载
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
                "use_flash_attention": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Use FlashAttention2",
                    "label_off": "Use Default"
                }),
                "fast_load": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Fast Load",
                    "label_off": "Normal Load"
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
                "num_beams": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "display": "number"
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Use KV Cache",
                    "label_off": "No Cache"
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
        self.load_time_stats = {}
    
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
    
    def _fast_load_processor(self, model_path):
        """快速加载处理器"""
        def load_processor():
            return AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        
        # 使用线程并行加载
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_processor)
            return future.result()
    
    def _load_model_fast(self, model_path, model_type, use_flash, fast_load):
        """优化的模型加载方法"""
        start_time = time.time()
        
        # 确定设备
        device = self.device
        
        # 获取优化的加载参数
        model_kwargs = FastModelLoader.optimize_model_loading_params(device)
        
        # 如果不使用FlashAttention，回退到SDPA
        if not use_flash or not FLASH_ATTENTION_AVAILABLE:
            model_kwargs["attn_implementation"] = "sdpa"
            print("[Qwen Image Captioner] Using SDPA attention")
        
        try:
            # 并行加载处理器
            processor_thread = threading.Thread(
                target=lambda: setattr(self, 'processor', self._fast_load_processor(model_path))
            )
            processor_thread.start()
            
            # 根据模型类型选择加载方式
            if "qwen2_5_vl" in model_type and QWEN2_5VL_AVAILABLE:
                print("[Qwen Image Captioner] Loading Qwen2.5-VL model...")
                
                if fast_load and ACCELERATE_AVAILABLE:
                    # 尝试使用accelerate加速加载
                    self.model = FastModelLoader.load_with_accelerate(
                        Qwen2_5VLForConditionalGeneration,
                        model_path,
                        device,
                        model_kwargs.get("torch_dtype", torch.float32)
                    )
                
                if self.model is None:
                    # 标准加载方式
                    self.model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    
            elif "qwen2_vl" in model_type and QWEN2VL_AVAILABLE:
                print("[Qwen Image Captioner] Loading Qwen2-VL model...")
                
                if fast_load and ACCELERATE_AVAILABLE:
                    self.model = FastModelLoader.load_with_accelerate(
                        Qwen2VLForConditionalGeneration,
                        model_path,
                        device,
                        model_kwargs.get("torch_dtype", torch.float32)
                    )
                
                if self.model is None:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
            else:
                # 使用AutoModel
                print("[Qwen Image Captioner] Loading with AutoModel...")
                from transformers import AutoModelForVision2Seq
                
                try:
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
            
            # 等待处理器加载完成
            processor_thread.join()
            
            # 设置为评估模式
            self.model.eval()
            
            # 启用优化
            if device == "cuda":
                # 启用cudnn优化
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # 编译模型（如果支持）
                if hasattr(torch, 'compile') and fast_load:
                    try:
                        print("[Qwen Image Captioner] Compiling model with torch.compile...")
                        self.model = torch.compile(
                            self.model, 
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                    except Exception as e:
                        print(f"[Qwen Image Captioner] Compilation failed: {e}")
            
            load_time = time.time() - start_time
            self.load_time_stats[model_path] = load_time
            print(f"[Qwen Image Captioner] Model loaded in {load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"[Qwen Image Captioner] Error loading model: {e}")
            return False
    
    def load_model(self, model_name, device="auto", use_flash=True, fast_load=True):
        """加载Qwen视觉语言模型（优化版）"""
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
            print(f"[Qwen Image Captioner] Model {model_name} already loaded")
            return
        
        # 清理之前的模型
        self._cleanup_previous_model()
        
        # 模型路径
        model_path = os.path.join(folder_paths.models_dir, "Qwen", model_name)
        
        try:
            # 读取配置
            config = self._get_model_config(model_path)
            model_type = config.get("model_type", "").lower()
            
            print(f"[Qwen Image Captioner] Loading model: {model_name}")
            print(f"[Qwen Image Captioner] Model type: {model_type}")
            
            # 加载模型
            success = self._load_model_fast(model_path, model_type, use_flash, fast_load)
            
            if not success:
                raise ValueError(f"Failed to load model {model_name}")
            
            self.current_model = model_name
            self.current_device = device
            
            # 显示加载统计
            if model_path in self.load_time_stats:
                avg_time = sum(self.load_time_stats.values()) / len(self.load_time_stats)
                print(f"[Qwen Image Captioner] Average load time: {avg_time:.2f}s")
            
        except Exception as e:
            print(f"[Qwen Image Captioner] Error: {str(e)}")
            raise
    
    def _resize_image(self, pil_image, max_size=1024):
        """等比例缩放图像到指定最大尺寸"""
        width, height = pil_image.size
        
        if width <= max_size and height <= max_size:
            return pil_image
        
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    
    def _prepare_image(self, image, max_size=1024):
        """准备图像输入"""
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0]
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
        else:
            pil_image = image
        
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
    
    def _generate_optimized(self, pil_image, instruction, max_length, temperature, 
                           num_beams=1, use_cache=True):
        """优化的生成方法"""
        if self.processor and hasattr(self.processor, 'apply_chat_template'):
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
        elif self.tokenizer and hasattr(self.tokenizer, 'from_list_format'):
            query = self.tokenizer.from_list_format([
                {'image': pil_image},
                {'text': instruction}
            ])
            inputs = self.tokenizer(query, return_tensors='pt')
        else:
            raise NotImplementedError("Unsupported model format")
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=temperature > 0.1,
            num_beams=num_beams,
            use_cache=use_cache,
            pad_token_id=self.processor.tokenizer.pad_token_id if self.processor else self.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id if self.processor else self.tokenizer.eos_token_id,
        )
        
        # 使用混合精度推理
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=generation_config)
        
        return outputs
    
    def _decode_output(self, outputs, instruction):
        """解码模型输出"""
        if hasattr(outputs, 'cpu'):
            outputs = outputs.cpu()
        
        if self.processor and hasattr(self.processor, 'decode'):
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
        elif self.tokenizer:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if instruction in response:
            caption = response.split(instruction)[-1].strip()
        else:
            caption = response.split('\n')[-1].strip() if '\n' in response else response
        
        return caption
    
    def generate_caption(self, image, model_name, prompt_type, language, device, 
                        max_length, temperature, auto_unload=True, 
                        custom_instruction="", max_image_size=1024,
                        use_flash_attention=True, fast_load=True,
                        num_beams=1, use_cache=True):
        """生成图像描述（优化版）"""
        
        try:
            total_start = time.time()
            
            # 加载模型
            load_start = time.time()
            self.load_model(model_name, device, use_flash_attention, fast_load)
            load_time = time.time() - load_start
            
            # 准备图像
            prep_start = time.time()
            pil_image = self._prepare_image(image, max_image_size)
            prep_time = time.time() - prep_start
            
            # 获取指令
            instruction = self._get_instruction(prompt_type, language, custom_instruction)
            
            # 生成描述
            gen_start = time.time()
            outputs = self._generate_optimized(
                pil_image, instruction, max_length, temperature,
                num_beams, use_cache
            )
            gen_time = time.time() - gen_start
            
            # 解码输出
            decode_start = time.time()
            caption = self._decode_output(outputs, instruction)
            decode_time = time.time() - decode_start
            
            total_time = time.time() - total_start
            
            # 显示性能统计
            print(f"[Qwen Image Captioner] Performance breakdown:")
            print(f"  - Model loading: {load_time:.2f}s")
            print(f"  - Image prep: {prep_time:.3f}s")
            print(f"  - Generation: {gen_time:.2f}s")
            print(f"  - Decoding: {decode_time:.3f}s")
            print(f"  - Total time: {total_time:.2f}s")
            
            if FLASH_ATTENTION_AVAILABLE and use_flash_attention:
                print(f"[Qwen Image Captioner] FlashAttention2 enabled ✓")
            
            return (caption,)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[Qwen Image Captioner] {error_msg}")
            
            if "CUDA" in str(e):
                print("[Qwen Image Captioner] Tips:")
                print("1. Try using CPU instead")
                print("2. Reduce max_length")
                print("3. Check GPU memory")
            
            return (error_msg,)
        
        finally:
            if auto_unload:
                print("[Qwen Image Captioner] Auto-unloading model...")
                self._cleanup_previous_model()


# 节点映射
NODE_CLASS_MAPPINGS = {
    "QwenImageCaptioner": QwenImageCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageCaptioner": "🐳Qwen Image Captioner",
}