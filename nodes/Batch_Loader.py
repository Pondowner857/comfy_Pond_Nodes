import os
import random
import numpy as np
from PIL import Image
import torch
import json
import hashlib
from datetime import datetime
import re
from typing import List, Tuple, Dict, Optional
import folder_paths

class FileCache:
    """文件缓存管理器"""
    def __init__(self, max_cache_size=100):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_times = {}
    
    def get(self, key):
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_cache_size:
            # 移除最久未使用的项
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

# 全局缓存实例
image_cache = FileCache(max_cache_size=50)
text_cache = FileCache(max_cache_size=200)

class BaseFolderLoader:
    """基础文件夹加载器类"""
    
    def __init__(self):
        self.supported_image_formats = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif']
        self.supported_text_formats = ['.txt', '.prompt', '.caption']
        self.encoding_list = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'shift_jis', 'euc_kr', 'latin1']
    
    def get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值用于缓存"""
        stat = os.stat(file_path)
        return hashlib.md5(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
    
    def natural_sort_key(self, filename: str):
        """自然排序键函数"""
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return alphanum_key(filename)
    
    def get_files_from_folder(self, folder_path: str, file_type: str, recursive: bool = False, 
                            pattern: str = "*", sort_by: str = "name") -> List[str]:
        """获取文件夹中的文件列表（增强版）"""
        if not os.path.exists(folder_path):
            return []
        
        files = []
        
        if recursive:
            for root, dirs, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if self._match_file(file_path, file_type, pattern):
                        files.append(file_path)
        else:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and self._match_file(file_path, file_type, pattern):
                    files.append(file_path)
        
        # 排序
        if sort_by == "name":
            files.sort(key=lambda x: self.natural_sort_key(os.path.basename(x)))
        elif sort_by == "date":
            files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_by == "size":
            files.sort(key=lambda x: os.path.getsize(x))
        
        return files
    
    def _match_file(self, file_path: str, file_type: str, pattern: str) -> bool:
        """检查文件是否匹配条件"""
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        
        # 检查文件类型
        type_match = False
        if file_type == "image" and ext in self.supported_image_formats:
            type_match = True
        elif file_type == "text" and ext in self.supported_text_formats:
            type_match = True
        elif file_type == "auto":
            if ext in self.supported_image_formats + self.supported_text_formats:
                type_match = True
        
        if not type_match:
            return False
        
        # 检查文件名模式
        if pattern != "*":
            import fnmatch
            if not fnmatch.fnmatch(filename, pattern):
                return False
        
        return True
    
    def load_image_with_cache(self, image_path: str) -> torch.Tensor:
        """加载图像（带缓存）"""
        file_hash = self.get_file_hash(image_path)
        cached = image_cache.get(file_hash)
        
        if cached is not None:
            return cached.clone()
        
        try:
            img = Image.open(image_path)
            
            # 处理不同的图像模式
            if img.mode == 'RGBA':
                # 将RGBA转换为RGB，保留透明度信息
                rgb = Image.new('RGB', img.size, (255, 255, 255))
                rgb.paste(img, mask=img.split()[3])
                img = rgb
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            if img.mode == 'L':
                img = img.convert('RGB')
            
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]
            
            image_cache.set(file_hash, img_tensor)
            return img_tensor.clone()
            
        except Exception as e:
            print(f"加载图像出错 {image_path}: {e}")
            return torch.zeros((1, 64, 64, 3))
    
    def load_text_with_cache(self, text_path: str) -> str:
        """加载文本文件（带缓存和多编码支持）"""
        file_hash = self.get_file_hash(text_path)
        cached = text_cache.get(file_hash)
        
        if cached is not None:
            return cached
        
        content = ""
        for encoding in self.encoding_list:
            try:
                with open(text_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    text_cache.set(file_hash, content)
                    return content
            except:
                continue
        
        print(f"无法使用任何编码读取文本文件 {text_path}")
        return ""
    
    def find_paired_text(self, image_path: str) -> str:
        """查找图像对应的文本文件"""
        # 获取不带扩展名的文件路径
        base_path = os.path.splitext(image_path)[0]
        
        # 尝试多种常见的文本扩展名
        for ext in ['.txt', '.caption', '.prompt']:
            text_path = base_path + ext
            if os.path.exists(text_path):
                return self.load_text_with_cache(text_path)
        
        return ""  # 如果没找到配对文本，返回空字符串


class AdvancedFolderLoader(BaseFolderLoader):
    """高级文件夹加载器"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入文件夹路径"
                }),
                "mode": (["随机", "索引", "顺序"], {
                    "default": "随机"
                }),
                "file_type": (["图像", "文本", "自动", "图像+文本"], {
                    "default": "自动"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "display": "number"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "pattern": ("STRING", {
                    "default": "*",
                    "multiline": False,
                    "placeholder": "文件名模式 (如: *.png)"
                }),
                "recursive": ("BOOLEAN", {"default": False, "label_on": "递归", "label_off": "仅当前"}),
                "sort_by": (["名称", "日期", "大小"], {"default": "名称"}),
            },
            "optional": {
                "previous_index": ("INT", {"default": -1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "DICT")
    RETURN_NAMES = ("图像", "文本", "文件名", "索引", "文件总数", "元数据")
    FUNCTION = "load_from_folder"
    CATEGORY = "🐳Pond/Tools"
    OUTPUT_NODE = True
    
    def __init__(self):
        super().__init__()
        self.file_lists = {}  # 缓存文件列表
    
    def load_from_folder(self, folder_path, mode, file_type, index=0, seed=-1, pattern="*", 
                        recursive=False, sort_by="名称", previous_index=-1):
        """主加载函数"""
        
        # 处理可能为 None 的参数
        if index is None:
            index = 0
        if seed is None:
            seed = -1
        if pattern is None:
            pattern = "*"
        if previous_index is None:
            previous_index = -1
        
        # 转换中文参数为英文（内部使用）
        mode_map = {"随机": "random", "索引": "index", "顺序": "sequential"}
        file_type_map = {"图像": "image", "文本": "text", "自动": "auto", "图像+文本": "image_text"}
        sort_by_map = {"名称": "name", "日期": "date", "大小": "size"}
        
        mode = mode_map.get(mode, mode)
        file_type = file_type_map.get(file_type, file_type)
        sort_by = sort_by_map.get(sort_by, sort_by)
        
        # 处理图像+文本模式
        if file_type == "image_text":
            # 只获取图像文件
            files = self.get_files_from_folder(folder_path, "image", recursive, pattern, sort_by)
        else:
            # 获取或更新文件列表
            cache_key = f"{folder_path}_{file_type}_{pattern}_{recursive}_{sort_by}"
            
            if cache_key not in self.file_lists:
                files = self.get_files_from_folder(folder_path, file_type, recursive, pattern, sort_by)
                self.file_lists[cache_key] = files
            else:
                files = self.file_lists[cache_key]
        
        if not files:
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, "", "未找到文件", 0, 0, {})
        
        # 选择文件
        if mode == "random":
            if seed >= 0:
                random.seed(seed)
            selected_index = random.randint(0, len(files) - 1)
        elif mode == "sequential":
            selected_index = (previous_index + 1) % len(files) if previous_index >= 0 else 0
        else:  # index mode
            selected_index = index % len(files)
        
        selected_file = files[selected_index]
        filename = os.path.basename(selected_file)
        ext = os.path.splitext(filename)[1].lower()
        
        # 构建元数据
        metadata = {
            "完整路径": selected_file,
            "文件名": filename,
            "扩展名": ext,
            "文件大小": os.path.getsize(selected_file),
            "修改时间": datetime.fromtimestamp(os.path.getmtime(selected_file)).isoformat(),
            "索引": selected_index,
            "总文件数": len(files)
        }
        
        # 加载内容
        if file_type == "image_text":
            # 图像+文本模式
            image = self.load_image_with_cache(selected_file)
            paired_text = self.find_paired_text(selected_file)
            
            if paired_text:
                text = paired_text
                metadata["配对文本"] = "找到"
                metadata["文本长度"] = len(paired_text)
            else:
                text = f"未找到 {filename} 的配对文本文件"
                metadata["配对文本"] = "未找到"
            
            # 添加图像元数据
            try:
                with Image.open(selected_file) as img:
                    metadata["宽度"] = img.width
                    metadata["高度"] = img.height
                    metadata["模式"] = img.mode
            except:
                pass
                
        elif ext in self.supported_image_formats:
            image = self.load_image_with_cache(selected_file)
            text = f"图像: {filename}"
            
            # 添加图像元数据
            try:
                with Image.open(selected_file) as img:
                    metadata["宽度"] = img.width
                    metadata["高度"] = img.height
                    metadata["模式"] = img.mode
                    if hasattr(img, 'info'):
                        metadata["EXIF信息"] = str(img.info)
            except:
                pass
                
        elif ext in self.supported_text_formats:
            image = torch.zeros((1, 64, 64, 3))
            text = self.load_text_with_cache(selected_file)
            metadata["文本长度"] = len(text)
            metadata["行数"] = text.count('\n') + 1
        else:
            image = torch.zeros((1, 64, 64, 3))
            text = "不支持的文件类型"
        
        return (image, text, filename, selected_index, len(files), metadata)


class SmartBatchLoader(BaseFolderLoader):
    """智能批量加载器"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "placeholder": "输入文件夹路径"}),
                "file_type": (["图像", "文本", "混合", "图像+文本"], {"default": "图像"}),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "display": "number"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "display": "number"
                }),
                "shuffle": ("BOOLEAN", {"default": False, "label_on": "打乱", "label_off": "顺序"}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "group_by": (["无", "扩展名", "前缀", "日期"], {"default": "无"}),
                "resize_mode": (["无", "缩放", "裁剪", "填充"], {"default": "无"}),
                "target_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "LIST", "DICT")
    RETURN_NAMES = ("图像批次", "文本内容", "文件总数", "文件信息", "统计信息")
    FUNCTION = "load_batch"
    CATEGORY = "🐳Pond/Tools"
    OUTPUT_NODE = True
    
    def group_files(self, files: List[str], group_by: str) -> Dict[str, List[str]]:
        """根据条件分组文件"""
        groups = {}
        
        for file_path in files:
            if group_by == "extension":
                key = os.path.splitext(file_path)[1].lower()
            elif group_by == "prefix":
                basename = os.path.basename(file_path)
                key = basename.split('_')[0] if '_' in basename else basename[:3]
            elif group_by == "date":
                mtime = os.path.getmtime(file_path)
                key = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
            else:
                key = "all"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(file_path)
        
        return groups
    
    def resize_image(self, img: Image.Image, resize_mode: str, target_size: int) -> Image.Image:
        """调整图像大小"""
        if resize_mode == "none":
            return img
        
        if resize_mode == "resize":
            # 保持宽高比缩放
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        elif resize_mode == "crop":
            # 中心裁剪
            width, height = img.size
            if width > height:
                left = (width - height) // 2
                img = img.crop((left, 0, left + height, height))
            else:
                top = (height - width) // 2
                img = img.crop((0, top, width, top + width))
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        elif resize_mode == "pad":
            # 填充到目标大小
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            paste_x = (target_size - img.width) // 2
            paste_y = (target_size - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        
        return img
    
    def load_batch(self, folder_path, file_type, batch_size=1, start_index=0, shuffle=False, 
                   seed=-1, group_by="无", resize_mode="无", target_size=512):
        """批量加载文件"""
        
        # 处理可能为 None 的参数
        if batch_size is None:
            batch_size = 1
        if start_index is None:
            start_index = 0
        if shuffle is None:
            shuffle = False
        if seed is None:
            seed = -1
        if target_size is None:
            target_size = 512
        
        # 转换中文参数为英文（内部使用）
        file_type_map = {"图像": "image", "文本": "text", "混合": "mixed", "图像+文本": "image_text"}
        group_by_map = {"无": "none", "扩展名": "extension", "前缀": "prefix", "日期": "date"}
        resize_mode_map = {"无": "none", "缩放": "resize", "裁剪": "crop", "填充": "pad"}
        
        file_type = file_type_map.get(file_type, file_type)
        group_by = group_by_map.get(group_by, group_by)
        resize_mode = resize_mode_map.get(resize_mode, resize_mode)
        
        # 获取文件列表
        if file_type == "mixed":
            files = self.get_files_from_folder(folder_path, "auto", recursive=False)
        elif file_type == "image_text":
            # 图像+文本模式，只获取图像文件
            files = self.get_files_from_folder(folder_path, "image", recursive=False)
        else:
            files = self.get_files_from_folder(folder_path, file_type, recursive=False)
        
        if not files:
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image, "", 0, [], {})
        
        # 分组处理
        if group_by != "none":
            groups = self.group_files(files, group_by)
            # 选择最大的组
            largest_group = max(groups.values(), key=len)
            files = largest_group
        
        # 打乱顺序
        if shuffle:
            if seed >= 0:
                random.seed(seed)
            files_copy = files.copy()
            random.shuffle(files_copy)
            files = files_copy
        
        # 选择批次文件
        batch_files = []
        for i in range(batch_size):
            idx = (start_index + i) % len(files)
            batch_files.append(files[idx])
        
        # 加载文件
        images = []
        texts = []
        file_info = []
        
        # 处理图像+文本模式
        if file_type == "image_text":
            for file_path in batch_files:
                filename = os.path.basename(file_path)
                info = {
                    "文件名": filename,
                    "路径": file_path,
                    "类型": "图像+文本"
                }
                
                try:
                    # 加载图像
                    img = Image.open(file_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 调整大小
                    img = self.resize_image(img, resize_mode, target_size)
                    
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(img_array)
                    info["宽度"] = img.width
                    info["高度"] = img.height
                    
                    # 查找配对文本
                    paired_text = self.find_paired_text(file_path)
                    if paired_text:
                        texts.append(f"[{filename}]:\n{paired_text}")
                        info["配对文本"] = "找到"
                        info["文本长度"] = len(paired_text)
                    else:
                        texts.append(f"[{filename}]: 未找到配对文本")
                        info["配对文本"] = "未找到"
                        
                except Exception as e:
                    print(f"加载图像出错 {file_path}: {e}")
                    if resize_mode != "none":
                        images.append(np.zeros((target_size, target_size, 3), dtype=np.float32))
                    else:
                        images.append(np.zeros((64, 64, 3), dtype=np.float32))
                    texts.append(f"错误: {filename}")
                    info["错误"] = str(e)
                
                file_info.append(info)
                
        # 在混合模式下，分别收集图像和文本
        elif file_type == "mixed":
            image_files = []
            text_files = []
            
            # 分类文件
            for file_path in batch_files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self.supported_image_formats:
                    image_files.append(file_path)
                elif ext in self.supported_text_formats:
                    text_files.append(file_path)
            
            # 处理图像文件
            for file_path in image_files:
                filename = os.path.basename(file_path)
                info = {
                    "文件名": filename,
                    "路径": file_path,
                    "类型": "图像"
                }
                
                try:
                    img = Image.open(file_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 调整大小
                    img = self.resize_image(img, resize_mode, target_size)
                    
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(img_array)
                    info["宽度"] = img.width
                    info["高度"] = img.height
                except Exception as e:
                    print(f"加载图像出错 {file_path}: {e}")
                    # 创建与目标大小匹配的空白图像
                    if resize_mode != "none":
                        images.append(np.zeros((target_size, target_size, 3), dtype=np.float32))
                    else:
                        images.append(np.zeros((64, 64, 3), dtype=np.float32))
                    info["错误"] = str(e)
                
                file_info.append(info)
            
            # 处理文本文件
            text_contents = []
            for file_path in text_files:
                filename = os.path.basename(file_path)
                info = {
                    "文件名": filename,
                    "路径": file_path,
                    "类型": "文本"
                }
                
                content = self.load_text_with_cache(file_path)
                text_contents.append(f"[{filename}]:\n{content}")
                info["文本长度"] = len(content)
                file_info.append(info)
            
            # 合并文本内容
            texts = text_contents if text_contents else ["未找到文本文件"]
            
            # 如果没有图像，创建一个空白图像
            if not images:
                if resize_mode != "none":
                    images.append(np.zeros((target_size, target_size, 3), dtype=np.float32))
                else:
                    images.append(np.zeros((64, 64, 3), dtype=np.float32))
        
        else:
            # 非混合模式，保持原有逻辑
            for file_path in batch_files:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                
                info = {
                    "文件名": filename,
                    "路径": file_path,
                    "类型": "图像" if ext in self.supported_image_formats else "文本"
                }
                
                if ext in self.supported_image_formats:
                    try:
                        img = Image.open(file_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 调整大小
                        img = self.resize_image(img, resize_mode, target_size)
                        
                        img_array = np.array(img).astype(np.float32) / 255.0
                        images.append(img_array)
                        texts.append(filename)
                        info["宽度"] = img.width
                        info["高度"] = img.height
                    except Exception as e:
                        print(f"加载图像出错 {file_path}: {e}")
                        # 创建与目标大小匹配的空白图像
                        if resize_mode != "none":
                            images.append(np.zeros((target_size, target_size, 3), dtype=np.float32))
                        else:
                            images.append(np.zeros((64, 64, 3), dtype=np.float32))
                        texts.append(f"错误: {filename}")
                        info["错误"] = str(e)
                
                elif ext in self.supported_text_formats:
                    # 文本文件，创建空白图像
                    if resize_mode != "none":
                        images.append(np.zeros((target_size, target_size, 3), dtype=np.float32))
                    else:
                        images.append(np.zeros((64, 64, 3), dtype=np.float32))
                    
                    content = self.load_text_with_cache(file_path)
                    texts.append(f"[{filename}]:\n{content}")
                    info["文本长度"] = len(content)
                
                file_info.append(info)
        
        # 转换为tensor
        if images:
            # 确保所有图像具有相同的尺寸
            max_h = max(img.shape[0] for img in images)
            max_w = max(img.shape[1] for img in images)
            
            padded_images = []
            for img in images:
                h, w = img.shape[:2]
                if h < max_h or w < max_w:
                    padded = np.zeros((max_h, max_w, 3), dtype=np.float32)
                    padded[:h, :w] = img
                    padded_images.append(padded)
                else:
                    padded_images.append(img)
            
            images_tensor = torch.from_numpy(np.stack(padded_images))
        else:
            images_tensor = torch.zeros((1, 64, 64, 3))
        
        # 处理文本输出和统计信息
        stats = {
            "总文件数": len(files),
            "批次大小": batch_size,
            "开始索引": start_index
        }
        
        if file_type == "mixed":
            # 混合模式下，只输出文本文件的内容
            combined_text = "\n---\n".join(texts) if isinstance(texts, list) else texts
            # 添加混合模式的统计信息
            stats["图像文件数"] = len([f for f in file_info if f["类型"] == "图像"])
            stats["文本文件数"] = len([f for f in file_info if f["类型"] == "文本"])
            stats["模式"] = "混合"
        elif file_type == "image_text":
            # 图像+文本模式
            combined_text = "\n---\n".join(texts) if texts else ""
            stats["模式"] = "图像+文本"
            stats["配对成功数"] = len([f for f in file_info if f.get("配对文本") == "找到"])
            stats["配对失败数"] = len([f for f in file_info if f.get("配对文本") == "未找到"])
        else:
            # 非混合模式，保持原有逻辑
            combined_text = "\n---\n".join(texts) if texts else ""
            stats["模式"] = file_type
        
        return (images_tensor, combined_text, len(files), file_info, stats)


# 清理缓存的辅助函数
def clear_all_caches():
    """清理所有缓存"""
    global image_cache, text_cache
    image_cache.clear()
    text_cache.clear()
    print("所有缓存已清理")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AdvancedFolderLoader": AdvancedFolderLoader,
    "SmartBatchLoader": SmartBatchLoader,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedFolderLoader": "🐳文件夹加载",
    "SmartBatchLoader": "🐳批量加载",
}