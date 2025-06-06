import os
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import json

class RemoveMetadata:
    """
    删除图像中的所有元数据信息
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE",),
                "文件名前缀": ("STRING", {"default": "ComfyUI_clean"}),
                "删除所有元数据": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "保存工作流": ("BOOLEAN", {"default": False}),
                "自定义元数据": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "remove_metadata"
    OUTPUT_NODE = True
    CATEGORY = "🐳Pond/元数据"

    def remove_metadata(self, 图像, 文件名前缀="ComfyUI_clean", 
                       删除所有元数据=True, 保存工作流=False, 
                       自定义元数据=""):
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        
        # 处理批量图像
        batch_size = 图像.shape[0]
        results = []
        
        for batch_idx in range(batch_size):
            # 将tensor转换为PIL图像
            img_tensor = 图像[batch_idx]
            img_array = 255. * img_tensor.cpu().numpy()
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # 如果是RGB图像
            if img_array.shape[-1] == 3:
                img = Image.fromarray(img_array, mode='RGB')
            # 如果是RGBA图像
            elif img_array.shape[-1] == 4:
                img = Image.fromarray(img_array, mode='RGBA')
            else:
                # 灰度图像
                img = Image.fromarray(img_array.squeeze(), mode='L')
            
            # 准备文件名
            file_name = f"{文件名前缀}_{batch_idx:05d}.png"
            file_path = os.path.join(output_dir, file_name)
            
            # 准备元数据
            metadata = PngInfo()
            
            if not 删除所有元数据:
                # 如果不删除所有元数据，可以添加自定义元数据
                if 保存工作流:
                    # 这里可以添加工作流信息（如果需要）
                    metadata.add_text("workflow", "cleaned")
                
                if 自定义元数据:
                    # 添加自定义元数据
                    try:
                        custom_dict = json.loads(自定义元数据)
                        for key, value in custom_dict.items():
                            metadata.add_text(str(key), str(value))
                    except:
                        # 如果不是JSON格式，直接添加为文本
                        metadata.add_text("custom", 自定义元数据)
                
                # 保存带有选择性元数据的图像
                img.save(file_path, pnginfo=metadata, compress_level=4)
            else:
                # 完全删除所有元数据
                img.save(file_path, compress_level=4)
            
            results.append({
                "filename": file_name,
                "subfolder": "",
                "type": "output"
            })
        
        # 返回原始图像（不修改）
        return (图像,)


class LoadImageWithoutMetadata:
    """
    加载图像时自动删除元数据
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "图像": (sorted(files), {"image_upload": True}),
                "清除元数据": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "🐳Pond/元数据"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "元数据信息")
    FUNCTION = "load_image"

    def load_image(self, 图像, 清除元数据=True):
        image_path = folder_paths.get_annotated_filepath(图像)
        
        # 使用PIL加载图像
        img = Image.open(image_path)
        
        # 提取元数据信息
        metadata_info = ""
        if hasattr(img, 'info'):
            metadata_info = json.dumps(img.info, indent=2, ensure_ascii=False)
        
        # 转换为RGB（如果需要）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 转换为numpy数组
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 转换为torch tensor
        img_tensor = torch.from_numpy(img_array)[None,]
        
        return (img_tensor, metadata_info)


class MetadataInspector:
    """
    检查图像的元数据信息
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "上传图像": (sorted(files), {"image_upload": True}),
            },
        }

    CATEGORY = "🐳Pond/元数据"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("所有元数据", "提示词", "工作流", "图像")
    FUNCTION = "inspect_metadata"

    def inspect_metadata(self, 上传图像):
        # 使用上传的图像
        image_path = folder_paths.get_annotated_filepath(上传图像)
        
        # 使用PIL加载图像
        img = Image.open(image_path)
        
        all_metadata = ""
        prompt = ""
        workflow = ""
        
        if hasattr(img, 'info'):
            # 获取所有元数据
            all_metadata = json.dumps(img.info, indent=2, ensure_ascii=False)
            
            # 尝试提取prompt
            if 'prompt' in img.info:
                prompt = img.info.get('prompt', '')
            
            # 尝试提取workflow
            if 'workflow' in img.info:
                workflow_data = img.info.get('workflow', '')
                if workflow_data:
                    try:
                        # 尝试解析并格式化workflow JSON
                        workflow_json = json.loads(workflow_data)
                        workflow = json.dumps(workflow_json, indent=2, ensure_ascii=False)
                    except:
                        workflow = workflow_data
        
        # 转换图像为tensor以便输出
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        
        return (all_metadata if all_metadata else "无元数据", 
                prompt if prompt else "无提示词", 
                workflow if workflow else "无工作流",
                img_tensor)


class BatchMetadataRemover:
    """
    批量处理文件夹中的图像，删除元数据
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入文件夹": ("STRING", {"default": ""}),
                "输出文件夹": ("STRING", {"default": "cleaned_images"}),
                "文件匹配模式": ("STRING", {"default": "*.png"}),
                "保留原始文件": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("处理状态",)
    FUNCTION = "batch_remove"
    OUTPUT_NODE = True
    CATEGORY = "🐳Pond/元数据"

    def batch_remove(self, 输入文件夹, 输出文件夹, 文件匹配模式="*.png", 保留原始文件=True):
        import glob
        
        if not 输入文件夹:
            return ("错误：未指定输入文件夹",)
        
        if not os.path.exists(输入文件夹):
            return (f"错误：输入文件夹 '{输入文件夹}' 不存在",)
        
        # 创建输出文件夹
        if not os.path.exists(输出文件夹):
            os.makedirs(输出文件夹)
        
        # 获取匹配的文件
        pattern_path = os.path.join(输入文件夹, 文件匹配模式)
        files = glob.glob(pattern_path)
        
        if not files:
            return (f"未找到匹配的文件：{pattern_path}",)
        
        processed = 0
        errors = 0
        
        for file_path in files:
            try:
                # 打开图像
                img = Image.open(file_path)
                
                # 获取文件名
                filename = os.path.basename(file_path)
                
                # 准备输出路径
                if 保留原始文件:
                    output_path = os.path.join(输出文件夹, filename)
                else:
                    output_path = file_path
                
                # 根据原始格式保存
                if file_path.lower().endswith('.png'):
                    img.save(output_path, 'PNG', compress_level=4)
                elif file_path.lower().endswith(('.jpg', '.jpeg')):
                    img.save(output_path, 'JPEG', quality=95)
                else:
                    img.save(output_path)
                
                processed += 1
                
            except Exception as e:
                print(f"处理 {file_path} 时出错：{str(e)}")
                errors += 1
        
        status = f"处理完成：成功 {processed} 个文件，失败 {errors} 个"
        return (status,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "RemoveMetadata": RemoveMetadata,
    "LoadImageWithoutMetadata": LoadImageWithoutMetadata,
    "MetadataInspector": MetadataInspector,
    "BatchMetadataRemover": BatchMetadataRemover,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveMetadata": "🐳删除元数据",
    "LoadImageWithoutMetadata": "🐳加载图像(清除元数据)",
    "MetadataInspector": "🐳查看元数据",
    "BatchMetadataRemover": "🐳批量删除元数据",
}

# 插件信息
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']