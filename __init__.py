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
from .nodes.aged_damaged_effect import NODE_CLASS_MAPPINGS as aged_damaged_effect_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as aged_damaged_effect_DISPLAY_NAME_MAPPINGS
from .nodes.math_tools import NODE_CLASS_MAPPINGS as math_tools_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as math_tools_DISPLAY_NAME_MAPPINGS
from .nodes.VideoFrameExtractor import NODE_CLASS_MAPPINGS as VideoFrameExtractor_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VideoFrameExtractor_DISPLAY_NAME_MAPPINGS
from .nodes.BatchWatermark import NODE_CLASS_MAPPINGS as BatchWatermark_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BatchWatermark_DISPLAY_NAME_MAPPINGS
from .nodes.PoseSelector import NODE_CLASS_MAPPINGS as PoseSelector_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PoseSelector_DISPLAY_NAME_MAPPINGS
from .nodes.Clothing_Selector import NODE_CLASS_MAPPINGS as Clothing_Selector_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as Clothing_Selector_DISPLAY_NAME_MAPPINGS
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
    **aged_damaged_effect_MAPPINGS,
    **math_tools_MAPPINGS,
    **VideoFrameExtractor_MAPPINGS,
    **BatchWatermark_MAPPINGS,
    **PoseSelector_MAPPINGS,
    **Clothing_Selector_MAPPINGS
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
    **aged_damaged_effect_DISPLAY_NAME_MAPPINGS,
    **math_tools_DISPLAY_NAME_MAPPINGS,
    **VideoFrameExtractor_DISPLAY_NAME_MAPPINGS,
    **BatchWatermark_DISPLAY_NAME_MAPPINGS,
    **PoseSelector_DISPLAY_NAME_MAPPINGS,
    **Clothing_Selector_DISPLAY_NAME_MAPPINGS
}

# 导出
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

