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
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass

# ============================================================================
# 依赖检测
# ============================================================================

class DependencyStatus:
    """依赖状态管理"""
    qwen2vl_available: bool = False
    qwen2_5vl_available: bool = False
    flash_attention_available: bool = False
    accelerate_available: bool = False
    bitsandbytes_available: bool = False

_deps = DependencyStatus()

# Qwen2VL
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    _deps.qwen2vl_available = True
except ImportError:
    pass

# Qwen2.5VL
try:
    from transformers import Qwen2_5VLForConditionalGeneration, AutoProcessor as Qwen2_5VLProcessor
    _deps.qwen2_5vl_available = True
except ImportError:
    pass

# 基础类
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

# FlashAttention2
try:
    from flash_attn import flash_attn_func
    _deps.flash_attention_available = True
    print("[Qwen Captioner] ✓ FlashAttention2 available")
except ImportError:
    print("[Qwen Captioner] ✗ FlashAttention2 not available (pip install flash-attn)")

# Accelerate
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    _deps.accelerate_available = True
    print("[Qwen Captioner] ✓ Accelerate available")
except ImportError:
    print("[Qwen Captioner] ✗ Accelerate not available (pip install accelerate)")

# bitsandbytes (量化支持)
try:
    import bitsandbytes as bnb
    _deps.bitsandbytes_available = True
    print("[Qwen Captioner] ✓ BitsAndBytes available (INT8/INT4 quantization)")
except ImportError:
    print("[Qwen Captioner] ✗ BitsAndBytes not available (pip install bitsandbytes)")


# ============================================================================
# 错误类型定义
# ============================================================================

class QwenCaptionerError(Exception):
    """基础异常类"""
    pass

class ModelNotFoundError(QwenCaptionerError):
    """模型未找到"""
    pass

class ModelLoadError(QwenCaptionerError):
    """模型加载失败"""
    pass

class InvalidModelTypeError(QwenCaptionerError):
    """无效的模型类型"""
    pass

class ImageProcessingError(QwenCaptionerError):
    """图像处理错误"""
    pass

class GenerationError(QwenCaptionerError):
    """生成错误"""
    pass

class CUDAOutOfMemoryError(QwenCaptionerError):
    """显存不足"""
    pass

class QuantizationError(QwenCaptionerError):
    """量化错误"""
    pass


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    model_type: str
    is_vl_model: bool
    config: Dict[str, Any]


@dataclass
class LoadedModel:
    """已加载的模型"""
    model: Any
    processor: Any
    tokenizer: Any
    device: str
    dtype: torch.dtype
    quantization: str = "none"
    load_time: float = 0.0


@dataclass
class PerformanceStats:
    """性能统计"""
    model_load_time: float = 0.0
    image_prep_time: float = 0.0
    generation_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    peak_memory_mb: float = 0.0
    
    def __str__(self) -> str:
        mem_str = f"  • Peak memory:   {self.peak_memory_mb:.1f}MB\n" if self.peak_memory_mb > 0 else ""
        return (
            f"Performance Stats:\n"
            f"  • Model loading: {self.model_load_time:.2f}s\n"
            f"  • Image prep:    {self.image_prep_time:.3f}s\n"
            f"  • Generation:    {self.generation_time:.2f}s\n"
            f"  • Decoding:      {self.decode_time:.3f}s\n"
            f"{mem_str}"
            f"  • Total:         {self.total_time:.2f}s"
        )


# ============================================================================
# LRU模型缓存
# ============================================================================

class ModelCache:
    """LRU模型缓存，支持多模型热切换"""
    
    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[LoadedModel]:
        """获取缓存的模型"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def put(self, key: str, model: LoadedModel) -> None:
        """缓存模型"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = model
            else:
                if len(self._cache) >= self.max_size:
                    oldest_key, oldest_model = self._cache.popitem(last=False)
                    self._unload_model(oldest_model, oldest_key)
                self._cache[key] = model
    
    def remove(self, key: str) -> None:
        """移除指定模型"""
        with self._lock:
            if key in self._cache:
                model = self._cache.pop(key)
                self._unload_model(model, key)
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            for key, model in list(self._cache.items()):
                self._unload_model(model, key)
            self._cache.clear()
    
    def _unload_model(self, loaded_model: LoadedModel, key: str) -> None:
        """卸载模型并释放内存"""
        print(f"[Qwen Captioner] Unloading cached model: {key}")
        
        if loaded_model.model is not None:
            del loaded_model.model
        if loaded_model.processor is not None:
            del loaded_model.processor
        if loaded_model.tokenizer is not None:
            del loaded_model.tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @property
    def cached_models(self) -> List[str]:
        """返回缓存的模型列表"""
        with self._lock:
            return list(self._cache.keys())
    
    def __len__(self) -> int:
        return len(self._cache)


# 全局模型缓存
_model_cache = ModelCache(max_size=2)


# ============================================================================
# 工具函数
# ============================================================================

def get_dtype_from_precision(precision: str, device: str) -> torch.dtype:
    """根据精度设置获取数据类型"""
    if device == "cpu":
        return torch.float32
    
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[Qwen Captioner] BF16 not supported, using FP16")
        return torch.float16
    elif precision in ("int8", "int4"):
        # 量化模型使用bf16作为计算类型
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    else:
        return torch.float32


def get_attention_implementation(attention_mode: str) -> str:
    """获取注意力实现方式"""
    if attention_mode == "auto":
        # 自动选择：优先FlashAttention2，其次SDPA
        if _deps.flash_attention_available:
            return "flash_attention_2"
        return "sdpa"
    elif attention_mode == "flash_attention_2":
        if not _deps.flash_attention_available:
            print("[Qwen Captioner] ⚠ FlashAttention2 not available, falling back to SDPA")
            return "sdpa"
        return "flash_attention_2"
    elif attention_mode == "sdpa":
        return "sdpa"
    elif attention_mode == "eager":
        return "eager"
    else:
        return "sdpa"


def clear_cuda_memory():
    """清理CUDA内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_usage() -> float:
    """获取GPU内存使用量(MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory_stats():
    """重置GPU内存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# 图像处理器
# ============================================================================

class ImageProcessor:
    """优化的图像处理器"""
    
    @staticmethod
    def tensor_to_pil(image: torch.Tensor) -> Image.Image:
        """将tensor转换为PIL图像 - 原地操作优化"""
        if len(image.shape) == 4:
            image = image[0]
        
        # 避免不必要的拷贝，直接操作
        if image.device.type == "cuda":
            image = image.cpu()
        
        # 使用 numpy 的原地操作
        image_np = image.numpy()
        np.multiply(image_np, 255, out=image_np)
        image_np = image_np.astype(np.uint8)
        
        return Image.fromarray(image_np)
    
    @staticmethod
    def resize_image(
        pil_image: Image.Image, 
        max_size: int = 1024,
        resample: Image.Resampling = Image.Resampling.BILINEAR
    ) -> Image.Image:
        """等比例缩放图像"""
        width, height = pil_image.size
        
        if width <= max_size and height <= max_size:
            return pil_image
        
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return pil_image.resize((new_width, new_height), resample)
    
    @staticmethod
    def prepare_single(
        image: Union[torch.Tensor, Image.Image], 
        max_size: int = 1024
    ) -> Image.Image:
        """准备单张图像"""
        if isinstance(image, torch.Tensor):
            pil_image = ImageProcessor.tensor_to_pil(image)
        else:
            pil_image = image
        
        # 确保是RGB模式
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        return ImageProcessor.resize_image(pil_image, max_size)


# ============================================================================
# 量化配置
# ============================================================================

class QuantizationConfig:
    """量化配置管理"""
    
    @staticmethod
    def get_config(precision: str) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if precision == "int8":
            if not _deps.bitsandbytes_available:
                raise QuantizationError(
                    "INT8量化需要bitsandbytes库\n"
                    "安装: pip install bitsandbytes"
                )
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        
        elif precision == "int4":
            if not _deps.bitsandbytes_available:
                raise QuantizationError(
                    "INT4量化需要bitsandbytes库\n"
                    "安装: pip install bitsandbytes"
                )
            # 优先使用bf16作为计算类型
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        return None
    
    @staticmethod
    def get_memory_estimate(precision: str, model_params_b: float = 7.0) -> str:
        """估算内存使用"""
        estimates = {
            "bf16": model_params_b * 2,
            "int8": model_params_b * 1,
            "int4": model_params_b * 0.5,
        }
        mem = estimates.get(precision, model_params_b * 2)
        return f"~{mem:.1f}GB VRAM"


# ============================================================================
# 模型加载器
# ============================================================================

class ModelLoader:
    """优化的模型加载器"""
    
    @staticmethod
    def scan_models(qwen_dir: str) -> List[str]:
        """扫描可用模型"""
        available = []
        
        if not os.path.exists(qwen_dir):
            return ["No models found in models/Qwen/"]
        
        for item in os.listdir(qwen_dir):
            model_path = os.path.join(qwen_dir, item)
            
            if not os.path.isdir(model_path):
                continue
            
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                continue
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                
                if "vl" in model_type or "VL" in item:
                    available.append(item)
                else:
                    available.append(f"[非VL] {item}")
            except (json.JSONDecodeError, IOError):
                available.append(f"[配置错误] {item}")
        
        return available if available else ["No models found in models/Qwen/"]
    
    @staticmethod
    def get_model_info(model_path: str) -> ModelInfo:
        """获取模型信息"""
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise ModelNotFoundError(f"Config not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelLoadError(f"Invalid config.json: {e}")
        
        model_type = config.get("model_type", "").lower()
        name = os.path.basename(model_path)
        is_vl = "vl" in model_type or "VL" in name
        
        return ModelInfo(
            name=name,
            path=model_path,
            model_type=model_type,
            is_vl_model=is_vl,
            config=config
        )
    
    @staticmethod
    def get_loading_params(
        device: str, 
        attention_mode: str, 
        precision: str
    ) -> Dict[str, Any]:
        """获取模型加载参数"""
        params = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if device == "cuda" and torch.cuda.is_available():
            params["device_map"] = "auto"
            params["torch_dtype"] = get_dtype_from_precision(precision, device)
            
            # 量化模式下强制使用SDPA，FlashAttention2与量化不兼容
            is_quantized = precision in ("int8", "int4")
            if is_quantized:
                if attention_mode == "flash_attention_2":
                    print("[Qwen Captioner] ⚠ 量化模式不支持FlashAttention2，自动切换到SDPA")
                params["attn_implementation"] = "sdpa"
            else:
                params["attn_implementation"] = get_attention_implementation(attention_mode)
            
            # 量化配置
            quant_config = QuantizationConfig.get_config(precision)
            if quant_config is not None:
                params["quantization_config"] = quant_config
                print(f"[Qwen Captioner] Using {precision.upper()} quantization")
        else:
            params["device_map"] = "cpu"
            params["torch_dtype"] = torch.float32
        
        return params
    
    @staticmethod
    def load_processor(model_path: str) -> Any:
        """加载处理器"""
        return AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    
    @staticmethod
    def load_model(
        model_info: ModelInfo,
        device: str,
        attention_mode: str,
        precision: str
    ) -> Tuple[Any, Any, Any]:
        """加载模型和处理器"""
        
        model_path = model_info.path
        model_type = model_info.model_type
        params = ModelLoader.get_loading_params(device, attention_mode, precision)
        
        model = None
        processor = None
        tokenizer = None
        
        # 并行加载处理器
        processor_result = [None]
        processor_error = [None]
        
        def load_processor_async():
            try:
                processor_result[0] = ModelLoader.load_processor(model_path)
            except Exception as e:
                processor_error[0] = e
        
        processor_thread = threading.Thread(target=load_processor_async)
        processor_thread.start()
        
        print(f"[Qwen Captioner] Loading model with {precision.upper()} precision...")
        
        try:
            # 根据模型类型选择加载方式
            if "qwen2_5_vl" in model_type and _deps.qwen2_5vl_available:
                print("[Qwen Captioner] Loading Qwen2.5-VL model...")
                model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                    model_path, **params
                )
            elif "qwen2_vl" in model_type and _deps.qwen2vl_available:
                print("[Qwen Captioner] Loading Qwen2-VL model...")
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path, **params
                )
            else:
                print("[Qwen Captioner] Loading with AutoModel...")
                try:
                    from transformers import AutoModelForVision2Seq
                    model = AutoModelForVision2Seq.from_pretrained(
                        model_path, **params
                    )
                except (ImportError, ValueError):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, **params
                    )
                    
        except torch.cuda.OutOfMemoryError as e:
            raise CUDAOutOfMemoryError(
                f"GPU内存不足，无法加载模型。\n"
                f"建议：\n"
                f"1. 使用INT8或INT4量化\n"
                f"2. 使用更小的模型\n"
                f"3. 关闭其他GPU程序\n"
                f"原始错误: {e}"
            )
        except Exception as e:
            raise ModelLoadError(f"模型加载失败: {e}")
        
        # 等待处理器加载完成
        processor_thread.join()
        
        if processor_error[0]:
            raise ModelLoadError(f"处理器加载失败: {processor_error[0]}")
        
        processor = processor_result[0]
        
        # 设置评估模式
        model.eval()
        
        # CUDA优化
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        return model, processor, tokenizer


# ============================================================================
# 指令生成器
# ============================================================================

class InstructionGenerator:
    """指令生成器"""
    
    INSTRUCTIONS = {
        "中文": {
            "detailed": "请详细描述这张图片，包括所有的物体、人物、颜色、纹理、构图、光线和氛围。请具体说明空间关系和视觉元素。",
            "brief": "请简洁地描述图片的主要主题和关键元素。",
            "technical": "请从技术角度描述这张图片：构图、光线、色彩搭配、拍摄角度、景深以及任何后期处理效果。",
            "artistic": "请从艺术角度描述这张图片，重点关注艺术风格、情绪、美学特质和情感影响。包括艺术技巧和视觉叙事的细节。",
        },
        "English": {
            "detailed": "Describe this image in great detail, including all objects, people, colors, textures, composition, lighting, and atmosphere. Be specific about spatial relationships and visual elements.",
            "brief": "Provide a concise description of the main subject and key elements in this image.",
            "technical": "Describe this image focusing on technical aspects: composition, lighting, color palette, camera angle, depth of field, and any post-processing effects.",
            "artistic": "Describe this image with focus on artistic style, mood, aesthetic qualities, and emotional impact. Include details about artistic techniques and visual storytelling.",
        }
    }
    
    @classmethod
    def get_instruction(
        cls, 
        prompt_type: str, 
        language: str, 
        custom_instruction: str = ""
    ) -> str:
        """获取指令"""
        if prompt_type == "custom":
            return custom_instruction or cls.INSTRUCTIONS[language].get("detailed", "")
        
        return cls.INSTRUCTIONS.get(language, cls.INSTRUCTIONS["English"]).get(
            prompt_type, 
            cls.INSTRUCTIONS["English"]["detailed"]
        )


# ============================================================================
# 生成器
# ============================================================================

class CaptionGenerator:
    """描述生成器 - 内存优化版"""
    
    def __init__(self, loaded_model: LoadedModel):
        self.model = loaded_model.model
        self.processor = loaded_model.processor
        self.tokenizer = loaded_model.tokenizer
        self.device = loaded_model.device
        self.dtype = loaded_model.dtype
        self.quantization = loaded_model.quantization
    
    def _get_pad_token_id(self) -> int:
        """获取pad token id"""
        if self.processor and hasattr(self.processor, 'tokenizer'):
            return self.processor.tokenizer.pad_token_id
        if self.tokenizer:
            return self.tokenizer.pad_token_id
        return 0
    
    def _get_eos_token_id(self) -> int:
        """获取eos token id"""
        if self.processor and hasattr(self.processor, 'tokenizer'):
            return self.processor.tokenizer.eos_token_id
        if self.tokenizer:
            return self.tokenizer.eos_token_id
        return 0
    
    def _prepare_inputs_inplace(
        self, 
        pil_image: Image.Image, 
        instruction: str
    ) -> Dict[str, torch.Tensor]:
        """准备模型输入 - 原地操作优化"""
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
            raise GenerationError("不支持的模型格式")
        
        # 原地移动到设备，避免额外内存分配
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(self.device, non_blocking=True)
        
        return inputs
    
    def generate(
        self,
        pil_image: Image.Image,
        instruction: str,
        max_length: int = 256,
        temperature: float = 0.7,
        num_beams: int = 1,
        use_cache: bool = True
    ) -> str:
        """生成图像描述 - 内存优化版"""
        
        inputs = None
        outputs = None
        
        try:
            # 准备输入
            inputs = self._prepare_inputs_inplace(pil_image, instruction)
            
            # 生成配置
            # 量化模式下禁用采样，避免multinomial CUDA错误
            is_quantized = self.quantization in ("int8", "int4")
            do_sample = temperature > 0.1 and num_beams == 1 and not is_quantized
            
            if is_quantized and temperature > 0.1:
                print("[Qwen Captioner] ⚠ 量化模式下已禁用采样，使用贪婪解码")
            
            generation_config = GenerationConfig(
                max_new_tokens=max_length,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=use_cache,
                pad_token_id=self._get_pad_token_id(),
                eos_token_id=self._get_eos_token_id(),
            )
            
            # 混合精度推理（使用新API，兼容CPU）
            autocast_enabled = self.device == "cuda"
            autocast_device = "cuda" if autocast_enabled else "cpu"
            
            with torch.amp.autocast(device_type=autocast_device, enabled=autocast_enabled):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        generation_config=generation_config
                    )
            
            # 解码
            caption = self._decode_output(outputs, instruction)
            
            return caption
            
        finally:
            # 积极清理中间张量
            if inputs is not None:
                for k in list(inputs.keys()):
                    del inputs[k]
                del inputs
            
            if outputs is not None:
                del outputs
            
            # 清理CUDA缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def _decode_output(self, outputs: torch.Tensor, instruction: str) -> str:
        """解码输出"""
        # 移动到CPU以释放GPU内存
        outputs_cpu = outputs.cpu()
        
        # 统一的解码逻辑
        decoder = (
            self.processor if self.processor and hasattr(self.processor, 'decode')
            else self.tokenizer if self.tokenizer
            else self.processor.tokenizer if self.processor and hasattr(self.processor, 'tokenizer')
            else None
        )
        
        if decoder is None:
            raise GenerationError("无法找到解码器")
        
        response = decoder.decode(outputs_cpu[0], skip_special_tokens=True)
        
        del outputs_cpu
        
        # 提取实际回答
        if instruction in response:
            caption = response.split(instruction)[-1].strip()
        elif '\n' in response:
            caption = response.split('\n')[-1].strip()
        else:
            caption = response.strip()
        
        return caption


# ============================================================================
# 主节点类
# ============================================================================

class QwenImageCaptioner:
    """
    优化版ComfyUI节点 - Qwen图像描述生成 v2
    
    特性：
    - 量化推理 (BF16/INT8/INT4)
    - 原地张量操作优化
    - LRU多模型缓存
    - FlashAttention2/SDPA加速
    - 详细性能统计
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        qwen_dir = os.path.join(folder_paths.models_dir, "Qwen")
        available_models = ModelLoader.scan_models(qwen_dir)
        
        # 精度选项（用户自行选择，如果选择int8/int4需要安装bitsandbytes）
        precision_options = ["bf16", "int8", "int4"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (available_models,),
                "prompt_type": (["detailed", "brief", "technical", "artistic", "custom"],),
                "language": (["English", "中文"],),
                "device": (["auto", "cuda", "cpu"],),
                "precision": (precision_options, {
                    "default": "bf16"
                }),
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
                "attention_mode": (["auto", "flash_attention_2", "sdpa", "eager"], {
                    "default": "auto"
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
                    "step": 128
                }),
                "num_beams": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
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
        self.stats = PerformanceStats()
    
    def _determine_device(self, device: str) -> str:
        """确定设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _get_cache_key(
        self, 
        model_name: str, 
        device: str, 
        attention_mode: str, 
        precision: str
    ) -> str:
        """生成缓存键"""
        return f"{model_name}_{device}_{attention_mode}_{precision}"
    
    def _load_or_get_model(
        self,
        model_name: str,
        device: str,
        attention_mode: str,
        precision: str,
        auto_unload: bool
    ) -> LoadedModel:
        """加载或获取缓存的模型"""
        
        # 验证模型名称
        if model_name.startswith("[非VL]"):
            raise InvalidModelTypeError(
                "这不是视觉语言(VL)模型！\n"
                "图像描述需要使用Qwen-VL系列模型。"
            )
        
        if model_name.startswith("[配置错误]"):
            raise ModelLoadError(f"模型配置文件损坏: {model_name}")
        
        cache_key = self._get_cache_key(model_name, device, attention_mode, precision)
        
        # 检查缓存
        cached = _model_cache.get(cache_key)
        if cached is not None:
            print(f"[Qwen Captioner] Using cached model: {model_name}")
            return cached
        
        # 加载新模型
        model_path = os.path.join(folder_paths.models_dir, "Qwen", model_name)
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"模型路径不存在: {model_path}")
        
        model_info = ModelLoader.get_model_info(model_path)
        
        if not model_info.is_vl_model:
            raise InvalidModelTypeError(
                f"'{model_name}' 不是视觉语言模型。\n"
                f"检测到的类型: {model_info.model_type}"
            )
        
        # 重置内存统计
        reset_gpu_memory_stats()
        
        start_time = time.perf_counter()
        
        model, processor, tokenizer = ModelLoader.load_model(
            model_info, device, attention_mode, precision
        )
        
        load_time = time.perf_counter() - start_time
        
        loaded_model = LoadedModel(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            dtype=get_dtype_from_precision(precision, device),
            quantization=precision,
            load_time=load_time
        )
        
        # 缓存模型（如果不是自动卸载模式）
        if not auto_unload:
            _model_cache.put(cache_key, loaded_model)
        
        print(f"[Qwen Captioner] Model loaded in {load_time:.2f}s ({precision.upper()})")
        
        return loaded_model
    
    def generate_caption(
        self,
        image: torch.Tensor,
        model_name: str,
        prompt_type: str,
        language: str,
        device: str,
        precision: str,
        max_length: int,
        temperature: float,
        auto_unload: bool = True,
        attention_mode: str = "auto",
        custom_instruction: str = "",
        max_image_size: int = 1024,
        num_beams: int = 1,
        use_cache: bool = True,
    ) -> Tuple[str]:
        """生成图像描述"""
        
        total_start = time.perf_counter()
        loaded_model = None
        
        try:
            # 确定设备
            device = self._determine_device(device)
            
            # 重置内存统计
            reset_gpu_memory_stats()
            
            # 加载模型
            load_start = time.perf_counter()
            loaded_model = self._load_or_get_model(
                model_name, device, attention_mode, 
                precision, auto_unload
            )
            self.stats.model_load_time = time.perf_counter() - load_start
            
            # 准备图像
            prep_start = time.perf_counter()
            pil_image = ImageProcessor.prepare_single(image, max_image_size)
            self.stats.image_prep_time = time.perf_counter() - prep_start
            
            # 获取指令
            instruction = InstructionGenerator.get_instruction(
                prompt_type, language, custom_instruction
            )
            
            # 创建生成器
            generator = CaptionGenerator(loaded_model)
            
            # 生成描述
            gen_start = time.perf_counter()
            caption = generator.generate(
                pil_image, instruction, max_length,
                temperature, num_beams, use_cache
            )
            self.stats.generation_time = time.perf_counter() - gen_start
            
            # 计算总时间和内存
            self.stats.total_time = time.perf_counter() - total_start
            self.stats.peak_memory_mb = get_gpu_memory_usage()
            
            # 打印性能统计
            print(f"[Qwen Captioner] {self.stats}")
            
            # 显示配置信息
            attn_impl = get_attention_implementation(attention_mode)
            print(f"[Qwen Captioner] Config: {precision.upper()}, {attn_impl}")
            
            return (caption,)
            
        except QuantizationError as e:
            error_msg = str(e)
            print(f"[Qwen Captioner] Quantization Error: {error_msg}")
            return (f"Error: {error_msg}",)
            
        except CUDAOutOfMemoryError as e:
            error_msg = str(e)
            print(f"[Qwen Captioner] CUDA OOM: {error_msg}")
            clear_cuda_memory()
            return (f"Error: {error_msg}",)
            
        except (ModelNotFoundError, ModelLoadError, InvalidModelTypeError) as e:
            error_msg = str(e)
            print(f"[Qwen Captioner] Model Error: {error_msg}")
            return (f"Error: {error_msg}",)
            
        except GenerationError as e:
            error_msg = str(e)
            print(f"[Qwen Captioner] Generation Error: {error_msg}")
            return (f"Error: {error_msg}",)
            
        except Exception as e:
            error_msg = f"未知错误: {type(e).__name__}: {str(e)}"
            print(f"[Qwen Captioner] {error_msg}")
            import traceback
            traceback.print_exc()
            return (f"Error: {error_msg}",)
            
        finally:
            # 自动卸载
            if auto_unload and loaded_model is not None:
                print("[Qwen Captioner] Auto-unloading model...")
                if loaded_model.model is not None:
                    del loaded_model.model
                if loaded_model.processor is not None:
                    del loaded_model.processor
                if loaded_model.tokenizer is not None:
                    del loaded_model.tokenizer
                clear_cuda_memory()


# ============================================================================
# 节点注册
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "QwenImageCaptioner": QwenImageCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageCaptioner": "🐳 Qwen Image Captioner (Optimized)",
}