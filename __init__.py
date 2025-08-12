# 导入各个节点模块的映射
from .nodes.yoloCrop import NODE_CLASS_MAPPINGS as yoloCrop_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as yoloCrop_DISPLAY_NAME_MAPPINGS
from .nodes.yoloPaste import NODE_CLASS_MAPPINGS as yoloPaste_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as yoloPaste_DISPLAY_NAME_MAPPINGS
from .nodes.BodyPartSelector import NODE_CLASS_MAPPINGS as BodyPartSelector_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BodyPartSelector_DISPLAY_NAME_MAPPINGS
from .nodes.MaskSwitch import NODE_CLASS_MAPPINGS as MaskSwitch_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskSwitch_DISPLAY_NAME_MAPPINGS
from .nodes.InvertImage import NODE_CLASS_MAPPINGS as InvertImage_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as InvertImage_DISPLAY_NAME_MAPPINGS
from .nodes.MaskRegionExpand import NODE_CLASS_MAPPINGS as MaskRegionExpand_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskRegionExpand_DISPLAY_NAME_MAPPINGS
from .nodes.MaskFeatherPercentage import NODE_CLASS_MAPPINGS as MaskFeatherPercentage_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskFeatherPercentage_DISPLAY_NAME_MAPPINGS
from .nodes.TextCleaner import NODE_CLASS_MAPPINGS as TextCleaner_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TextCleaner_DISPLAY_NAME_MAPPINGS
from .nodes.LimbSelector import NODE_CLASS_MAPPINGS as LimbSelector_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LimbSelector_DISPLAY_NAME_MAPPINGS
from .nodes.MemoryManager import NODE_CLASS_MAPPINGS as MemoryManager_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MemoryManager_DISPLAY_NAME_MAPPINGS
from .nodes.RealESRGANUpscaler import NODE_CLASS_MAPPINGS as RealESRGAN_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as RealESRGAN_DISPLAY_NAME_MAPPINGS
from .nodes.MaskSizeAlign import NODE_CLASS_MAPPINGS as MaskSizeAlign_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskSizeAlign_DISPLAY_NAME_MAPPINGS
from .nodes.MaskComposite import NODE_CLASS_MAPPINGS as MaskComposite_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskComposite_DISPLAY_NAME_MAPPINGS
from .nodes.MaskMerge import NODE_CLASS_MAPPINGS as MaskMerge_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskMerge_DISPLAY_NAME_MAPPINGS
from .nodes.MaskRemove import NODE_CLASS_MAPPINGS as MaskRemove_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskRemove_DISPLAY_NAME_MAPPINGS
from .nodes.MaskBoolean import NODE_CLASS_MAPPINGS as MaskBoolean_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MaskBoolean_DISPLAY_NAME_MAPPINGS
from .nodes.MetadataUtils import NODE_CLASS_MAPPINGS as MetadataUtils_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MetadataUtils_DISPLAY_NAME_MAPPINGS
from .nodes.RealisticNoise import NODE_CLASS_MAPPINGS as RealisticNoise_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as RealisticNoise_DISPLAY_NAME_MAPPINGS
from .nodes.desaturate import NODE_CLASS_MAPPINGS as desaturate_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as desaturate_DISPLAY_NAME_MAPPINGS
from .nodes.square_pixel import NODE_CLASS_MAPPINGS as square_pixel_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as square_pixel_DISPLAY_NAME_MAPPINGS
from .nodes.ImageAlignByMask import NODE_CLASS_MAPPINGS as ImageAlignByMask_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ImageAlignByMask_DISPLAY_NAME_MAPPINGS
from .nodes.auto_censor import NODE_CLASS_MAPPINGS as auto_censor_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as auto_censor_DISPLAY_NAME_MAPPINGS
from .nodes.math_tools import NODE_CLASS_MAPPINGS as math_tools_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as math_tools_DISPLAY_NAME_MAPPINGS
from .nodes.VideoFrameExtractor import NODE_CLASS_MAPPINGS as VideoFrameExtractor_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VideoFrameExtractor_DISPLAY_NAME_MAPPINGS
from .nodes.PoseSelector import NODE_CLASS_MAPPINGS as PoseSelector_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PoseSelector_DISPLAY_NAME_MAPPINGS
from .nodes.Clothing_Selector import NODE_CLASS_MAPPINGS as Clothing_Selector_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as Clothing_Selector_DISPLAY_NAME_MAPPINGS
from .nodes.image_filter import NODE_CLASS_MAPPINGS as image_filter_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as image_filter_DISPLAY_NAME_MAPPINGS
from .nodes.HDR import NODE_CLASS_MAPPINGS as HDR_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as HDR_DISPLAY_NAME_MAPPINGS
from .nodes.Batch_Loader import NODE_CLASS_MAPPINGS as Batch_Loader_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as Batch_Loader_DISPLAY_NAME_MAPPINGS
from .nodes.WanVideoReset import NODE_CLASS_MAPPINGS as WanVideoReset_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as WanVideoReset_DISPLAY_NAME_MAPPINGS
from .nodes.mask_color import NODE_CLASS_MAPPINGS as mask_color_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as mask_color_DISPLAY_NAME_MAPPINGS
from .nodes.Wan22_Prompt import NODE_CLASS_MAPPINGS as Wan22_Prompt_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as Wan22_Prompt_DISPLAY_NAME_MAPPINGS
from .nodes.ImagePad import NODE_CLASS_MAPPINGS as ImagePad_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ImagePad_DISPLAY_NAME_MAPPINGS
from .nodes.QwenPrompt import NODE_CLASS_MAPPINGS as QwenPrompt_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as QwenPrompt_DISPLAY_NAME_MAPPINGS
# 合并所有节点类映射
NODE_CLASS_MAPPINGS = {
    **yoloCrop_MAPPINGS,
    **yoloPaste_MAPPINGS,
    **BodyPartSelector_MAPPINGS,
    **MaskSwitch_MAPPINGS,
    **InvertImage_MAPPINGS,
    **MaskRegionExpand_MAPPINGS,
    **MaskFeatherPercentage_MAPPINGS,
    **TextCleaner_MAPPINGS,
    **LimbSelector_MAPPINGS,
    **MemoryManager_MAPPINGS,
    **RealESRGAN_MAPPINGS,
    **MaskSizeAlign_MAPPINGS,
    **MaskComposite_MAPPINGS,
    **MaskMerge_MAPPINGS,
    **MaskRemove_MAPPINGS,
    **MaskBoolean_MAPPINGS,
    **MetadataUtils_MAPPINGS,
    **RealisticNoise_MAPPINGS,
    **desaturate_MAPPINGS,
    **square_pixel_MAPPINGS,
    **ImageAlignByMask_MAPPINGS,
    **auto_censor_MAPPINGS,
    **math_tools_MAPPINGS,
    **VideoFrameExtractor_MAPPINGS,
    **PoseSelector_MAPPINGS,
    **Clothing_Selector_MAPPINGS,
    **image_filter_MAPPINGS,
    **HDR_MAPPINGS,
    **Batch_Loader_MAPPINGS,
    **WanVideoReset_MAPPINGS,
    **mask_color_MAPPINGS,
    **Wan22_Prompt_MAPPINGS,
    **ImagePad_MAPPINGS,
    **QwenPrompt_MAPPINGS
}

# 合并所有节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    **yoloCrop_DISPLAY_NAME_MAPPINGS,
    **yoloPaste_DISPLAY_NAME_MAPPINGS,
    **BodyPartSelector_DISPLAY_NAME_MAPPINGS,
    **MaskSwitch_DISPLAY_NAME_MAPPINGS,
    **InvertImage_DISPLAY_NAME_MAPPINGS,
    **MaskRegionExpand_DISPLAY_NAME_MAPPINGS,
    **MaskFeatherPercentage_DISPLAY_NAME_MAPPINGS,
    **TextCleaner_DISPLAY_NAME_MAPPINGS,
    **LimbSelector_DISPLAY_NAME_MAPPINGS,
    **MemoryManager_DISPLAY_NAME_MAPPINGS,
    **RealESRGAN_DISPLAY_NAME_MAPPINGS,
    **MaskSizeAlign_DISPLAY_NAME_MAPPINGS,
    **MaskComposite_DISPLAY_NAME_MAPPINGS,
    **MaskMerge_DISPLAY_NAME_MAPPINGS,
    **MaskRemove_DISPLAY_NAME_MAPPINGS,
    **MaskBoolean_DISPLAY_NAME_MAPPINGS,
    **MetadataUtils_DISPLAY_NAME_MAPPINGS,
    **RealisticNoise_DISPLAY_NAME_MAPPINGS,
    **desaturate_DISPLAY_NAME_MAPPINGS,
    **square_pixel_DISPLAY_NAME_MAPPINGS,
    **ImageAlignByMask_DISPLAY_NAME_MAPPINGS,
    **auto_censor_DISPLAY_NAME_MAPPINGS,
    **math_tools_DISPLAY_NAME_MAPPINGS,
    **VideoFrameExtractor_DISPLAY_NAME_MAPPINGS,
    **PoseSelector_DISPLAY_NAME_MAPPINGS,
    **Clothing_Selector_DISPLAY_NAME_MAPPINGS,
    **image_filter_DISPLAY_NAME_MAPPINGS,
    **HDR_DISPLAY_NAME_MAPPINGS,
    **Batch_Loader_DISPLAY_NAME_MAPPINGS,
    **WanVideoReset_DISPLAY_NAME_MAPPINGS,
    **mask_color_DISPLAY_NAME_MAPPINGS,
    **Wan22_Prompt_DISPLAY_NAME_MAPPINGS,
    **ImagePad_DISPLAY_NAME_MAPPINGS,
    **QwenPrompt_DISPLAY_NAME_MAPPINGS
}

# 导出
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]



import os
import shutil
import folder_paths

# 导入硬件监控模块
from .nodes.hardware_monitor import hardware_monitor

# 版本信息
__version__ = "1.0.6"

# 获取当前插件目录路径
current_dir = os.path.dirname(os.path.realpath(__file__))

# 定义WEB_DIRECTORY，这是ComfyUI检测自定义前端代码的方式
WEB_DIRECTORY = os.path.join(current_dir, "web")

# 确保web/js目录存在
js_path = os.path.join(WEB_DIRECTORY, "js")
if not os.path.exists(js_path):
    os.makedirs(js_path, exist_ok=True)

# 将JavaScript文件复制到ComfyUI的扩展目录中（作为备用方法）
def copy_js_to_extension_dir():
    try:
        # 获取ComfyUI web/extensions/comfyui 目录
        extensions_dir = os.path.join(folder_paths.base_path, "web", "extensions")
        
        # 确保目标目录存在
        if not os.path.exists(extensions_dir):
            os.makedirs(extensions_dir, exist_ok=True)
        
        # 硬件监控文件路径
        src_hw_file = os.path.join(js_path, "hardware_monitor.js")
        
        # 目标文件路径
        dst_hw_file = os.path.join(extensions_dir, "hardware_monitor.js")
        
        # 复制硬件监控文件
        if os.path.exists(src_hw_file):
            shutil.copy2(src_hw_file, dst_hw_file)
            print(f"Hardware Monitor: JS file copied to {dst_hw_file}")
    except Exception as e:
        print(f"Error copying JS files: {e}")

# 尝试复制JS文件（作为备用手段）
copy_js_to_extension_dir()

# 导出节点映射，使ComfyUI能够加载插件
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 

print("Hardware Monitor Plugin loaded successfully")

# 设置web目录，ComfyUI会自动加载web目录下的文件
#WEB_DIRECTORY = "web" 