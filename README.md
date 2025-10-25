# 🐳 Pond Nodes for ComfyUI

**Version 1.0.7** - Latest Update

[English](#english) | [中文](#chinese)

---

## 🔄 Recent Updates (v1.0.7)

### ✨ New Features
- **📝 Prompt Manager**: Advanced prompt management system with positive/negative prompt separation and weight control
- **🎨 Image Filters**: Comprehensive image filter nodes with brightness, contrast, saturation, sharpness, hue shift, blur, temperature, and gamma adjustments
- **🌈 HDR & Color**: Professional color grading nodes including HDR effects, skin enhancement, artistic effects, and selective color adjustment
- **📂 Batch Loading**: Advanced folder loader and smart batch loader with caching, multi-format support, and flexible file selection modes
- **🖼️ Image Processing**: New image padding, border removal, and crop-paste utilities
- **🔍 YOLO v11 Support**: Dedicated YOLOv11 detection and processing nodes
- **🎭 Mask Tools**: Enhanced mask solidification and color processing capabilities
- **📊 Prompt Templates**: Specialized prompt nodes for different AI models (Wan2.2, Qwen, etc.)

### 🔧 Improvements
- Enhanced file loading with intelligent caching system
- Added support for paired image-text loading
- Improved batch processing with resize and grouping options
- Better Chinese encoding support for text files
- Web UI enhancements for prompt management

### 📦 New Nodes (v1.0.7)
- Prompt Manager (🐳Prompt管理器)
- Image Filter Adjustment (🐳滤镜调节 / 🐳滤镜调节V2)
- Color Grading (🐳色彩平衡)
- HDR Effect (🐳HDR)
- Skin Enhancement (🐳人像美化)
- Artistic Effects (🐳艺术效果)
- Selective Color (🐳色彩范围)
- Folder Loader (🐳文件夹加载)
- Batch Loader (🐳批量加载)
- Image Padding (🐳图像填充)
- Border Remover (🐳边框处理)
- Mask Solidifier (🐳遮罩虚实)
- Crop Paste Back (🐳裁剪粘贴回)
- YOLO v11 Crop/Paste nodes

---

## Chinese

一个为ComfyUI设计的全面自定义节点集合，提供丰富的图像处理和遮罩操作功能，包含计算机视觉、图像处理和工作流优化的高级工具。

### ✨ 功能特点

该插件集合包含多种实用节点，可以帮助你：

- 🔍 **YOLO对象检测与裁剪**：
  - YOLO检测裁剪：智能对象检测和裁剪，支持类别过滤和置信度设置
  - YOLO图像拼接：将检测到的对象智能拼接到其他图像上

- 🎭 **高级遮罩操作**：
  - 遮罩布尔运算：支持交集、并集、差集、异或等操作，带有9种对齐方式
  - 多遮罩运算：支持多个遮罩的连续布尔运算
  - 遮罩合并与合成：提供遮罩合并和多遮罩合并节点
  - 遮罩区域扩展：可在四个方向扩展遮罩区域，支持渐变过渡
  - 遮罩百分比羽化：基于百分比调整羽化半径，可保持锐利边缘
  - 遮罩尺寸对齐：自动对齐不同尺寸的遮罩
  - 遮罩移除：移除图像中的遮罩区域
  - 遮罩切换：在不同遮罩间灵活切换
  - 基于遮罩的图像对齐：使用遮罩进行高级图像对齐

- 👤 **人体部位处理**：
  - 人体部位选择器：轻松选择并处理人体特定部位
  - 肢体选择器：支持20多个人体部位的选择，如眼睛、嘴巴、手臂等
  - 自动马赛克：基于姿态检测的自动马赛克功能，集成MediaPipe

- 🖼️ **图像处理**：
  - RealESRGAN超分辨率：使用ONNX格式的RealESRGAN模型进行高质量图像放大
  - 图像反相：一键反转图像颜色，带有开关控制
  - 去饱和度：基础和高级图像去色功能
  - 像素处理：像素化、像素校正、局部像素化和像素增强
  - 逼真噪点：添加自然外观的噪点，支持多种参数调整

- 📝 **文本处理**：
  - 文本清理器：清理和优化提示文本，支持标签过滤和句子过滤

- 📊 **元数据工具**：
  - 删除元数据：从图像中删除元数据
  - 加载图像(清除元数据)：加载图像的同时清除元数据
  - 查看元数据：检查图像的元数据信息
  - 批量删除元数据：批量处理多张图像的元数据

- 🧠 **内存管理**：
  - 内存管理器：优化节点处理过程中的内存使用，提高处理效率

- 🛠️ **数学工具**：
  - 多数值比较：支持多个数值的比较运算，包括最大值、最小值、中位数、平均值等
  - 宽高比计算器：智能计算图像宽高比，支持多种约束模式和预设比例
  - 数学运算：基础数学运算包括加减乘除、幂运算、三角函数等

- 🎬 **视频处理**：
  - 视频帧提取器：从视频中提取指定帧，支持索引、百分比、时间等提取模式
  - 高级帧提取：支持帧范围提取和批量处理

- 🎭 **姿势与服装**：
  - 姿势选择器：丰富的人体姿势标签选择，包含多种姿势类别和批量生成
  - 服装选择器：全面的服装标签系统，支持多种服装类型和穿搭建议

- 💻 **硬件监控**：
  - 硬件监控器：实时监控CPU、GPU、内存使用情况，优化工作流性能

- 📝 **提示词管理**：
  - 提示词管理器：支持正负面提示词分离管理，权重控制，动态提示词数量调整
  - 专业模板：支持Wan2.2、Qwen等多种AI模型的专业提示词模板

- 🎨 **图像滤镜**：
  - 基础滤镜：亮度、对比度、饱和度、锐度、色调、模糊、色温、伽马调整
  - 高级滤镜：晕影、色差、噪点、胶片颗粒、泛光等电影级效果

- 🌈 **颜色处理**：
  - 色彩平衡：专业高光、中间调、阴影分区调色
  - HDR效果：支持多种色调映射算法，细节增强，局部对比度调整
  - 人像美化：智能磨皮、美白、红润度、去瑕疵、眼睛增强、牙齿美白
  - 艺术效果：油画、水彩、素描、漫画、印象派、点彩画、版画、马赛克等8种艺术风格
  - 选择性颜色：针对特定颜色范围进行精确调整

- 📂 **批量处理**：
  - 文件夹加载器：支持图像、文本、图像+文本配对加载，智能缓存
  - 批量加载器：支持文件分组、打乱、调整大小、多种排序方式
  - 多格式支持：自动识别和处理多种图像和文本格式

### 📂 节点文件与功能对应

本插件包含以下Python模块文件，每个文件实现了特定的功能节点：

| 文件名 | 节点名称 | 功能描述 |
| ------ | ------- | ------- |
| **yoloCrop.py** | 🐳YOLO检测裁剪 | 使用YOLO模型检测图像中的对象并智能裁剪，支持多种检测类别和裁剪模式 |
| **yoloPaste.py** | 🐳YOLO图像拼接 | 将检测到的对象智能拼接到其他图像上，支持位置调整 |
| **MaskBoolean.py** | 🐳遮罩布尔运算, 🐳多遮罩运算 | 遮罩对齐布尔运算和多遮罩连续布尔运算，支持多种布尔操作 |
| **MaskComposite.py** | 🐳高级遮罩合成, 🐳基于遮罩的图像合成 | 复杂的遮罩合成工具，用于创建高级合成效果 |
| **MaskMerge.py** | 🐳多遮罩合并, 🐳遮罩合并 | 提供简单和高级两种遮罩合并功能 |
| **MaskRegionExpand.py** | 🐳遮罩区域扩展 | 在四个方向扩展遮罩区域，支持渐变过渡和边缘平滑 |
| **MaskFeatherPercentage.py** | 🐳遮罩百分比羽化 | 基于图像尺寸百分比调整羽化半径，可保持锐利边缘 |
| **MaskSizeAlign.py** | 🐳遮罩尺寸对齐 | 将不同尺寸的遮罩调整为相同尺寸，支持多种对齐方式 |
| **MaskRemove.py** | 🐳遮罩移除 | 从图像中移除遮罩区域，支持多种填充方式 |
| **MaskSwitch.py** | 🐳遮罩切换 | 在多个遮罩之间进行条件切换 |
| **ImageAlignByMask.py** | 🐳基于遮罩的图像对齐 | 使用遮罩定位进行高级图像对齐 |
| **BodyPartSelector.py** | 🐳人体部位选择器 | 人体主要部位的选择器，用于与ControlNet等配合使用 |
| **LimbSelector.py** | 🐳肢体选择器 | 详细的人体肢体部位选择器，支持20多个人体部位 |
| **auto_censor.py** | 🐳基于OpenPose的自动马赛克 | 使用OpenPose检测的自动马赛克，集成MediaPipe |
| **RealESRGANUpscaler.py** | 🐳RealESRGAN超分辨率 | 使用ONNX格式的RealESRGAN模型进行高质量图像放大 |
| **InvertImage.py** | 🐳图像反相 | 一键反转图像颜色，带有开关控制 |
| **desaturate.py** | 🐳图像去色, 🐳图像去色(V2) | 基础和高级的图像去饱和度处理 |
| **square_pixel.py** | 🐳像素化, 🐳像素校正, 🐳局部像素化, 🐳像素增强 | 多种像素艺术风格处理工具 |
| **RealisticNoise.py** | 🐳逼真噪点 | 添加自然外观的噪点，支持多种参数调整 |
| **TextCleaner.py** | 🐳文本清理器 | 清理和优化提示文本，支持标签过滤和句子过滤 |
| **MetadataUtils.py** | 🐳删除元数据, 🐳加载图像(清除元数据), 🐳查看元数据, 🐳批量删除元数据 | 图像元数据处理工具集 |
| **MemoryManager.py** | 🐳内存管理器 | 优化节点处理过程中的内存使用 |
| **math_tools.py** | 🐳多数值比较, 🐳宽高比计算, 🐳数学运算 | 数学工具集，支持多种数学运算和数值处理 |
| **VideoFrameExtractor.py** | 🐳视频帧提取器, 🐳高级视频帧提取器, 🐳视频帧范围提取器 | 视频帧提取和处理工具 |
| **PoseSelector.py** | 🐳姿势选择器, 🐳简单姿势选择器, 🐳批量姿势生成器 | 人体姿势标签选择和批量生成工具 |
| **Clothing_Selector.py** | 🐳服装选择器, 🐳简单服装选择器, 🐳批量服装生成器, 🐳服装穿搭建议 | 服装标签选择和穿搭建议工具 |
| **hardware_monitor.py** | 硬件监控服务 | 实时监控系统硬件状态，为其他节点提供性能参考 |
| **Prompt_manager.py** | 🐳Prompt管理器 | 支持正负面提示词分离管理，权重控制和动态提示词调整 |
| **image_filter.py** | 🐳滤镜调节, 🐳滤镜调节V2 | 基础和高级图像滤镜效果，支持多种调整参数和艺术效果 |
| **HDR.py** | 🐳色彩平衡, 🐳HDR, 🐳人像美化, 🐳艺术效果, 🐳色彩范围 | 专业颜色处理节点集，包含HDR、人像美化和艺术风格化效果 |
| **Batch_Loader.py** | 🐳文件夹加载, 🐳批量加载 | 高级文件夹和批量文件加载器，支持缓存和多种加载模式 |
| **ImagePad.py** | 🐳图像填充 | 根据参考图像调整大小并填充，支持多种对齐和填充模式 |
| **ImageBorder.py** | 🐳边框处理 | 智能移除图像边框，支持透明度检测和内容裁剪 |
| **maskSolid.py** | 🐳遮罩虚实 | 将遮罩转换为实心（二值化），增强遮罩效果 |
| **CropPaste.py** | 🐳裁剪粘贴回 | 将裁剪后的图像智能粘贴回原图指定位置 |
| **yoloCropV11.py** | 🐳YOLO v11检测裁剪 | 使用YOLO v11模型进行目标检测和裁剪 |
| **yoloPasteV11.py** | 🐳YOLO v11图像拼接 | 使用YOLO v11检测结果进行智能图像拼接 |
| **Wan22_Prompt.py** | 🐳Wan2.2提示词 | Wan2.2模型专用的提示词处理节点 |
| **QwenPrompt.py** | 🐳Qwen提示词 | Qwen模型专用的提示词处理节点 |
| **Prompt.py** | 🐳提示词 | 通用提示词处理和管理节点 |
| **mask_color.py** | 🐳遮罩颜色 | 遮罩颜色处理和转换工具 |
| **WanVideoReset.py** | 🐳视频重置 | 视频序列重置和处理工具 |
| **DigitalJudgment.py** | 🐳数字判断 | 数值比较和逻辑判断工具 |
| **iphone.py** | 🐳iPhone效果 | iPhone相机风格效果处理 |
| **maskBbox.py** | 🐳遮罩边界框 | 从遮罩提取边界框信息 |

### 📋 依赖要求

#### 核心依赖
- ComfyUI最新版本
- Python 3.8+
- torch >= 2.0.0
- Pillow >= 9.0.0
- numpy >= 1.22.0

#### 可选依赖（用于特定功能）
- ultralytics >= 8.0.0 (YOLO功能)
- onnxruntime >= 1.14.0 (RealESRGAN超分辨率和ONNX模型)
- scipy >= 1.8.0 (高级遮罩处理)
- opencv-python >= 4.5.0 (遮罩合成和自动马赛克)
- mediapipe >= 0.9.0 (自动马赛克姿态检测 - 有回退方案)
- realesrgan >= 0.3.0 (RealESRGAN超分辨率 - 有回退方案)
- torchvision >= 0.15.0 (YOLO拼接变换函数)
- requests >= 2.25.0 (内存管理器网络功能)
- psutil >= 5.8.0 (硬件监控功能)
- pynvml >= 8.0.4 (NVIDIA GPU监控 - 硬件监控功能)


1. 克隆或下载此仓库
2. 将文件夹放入ComfyUI的`custom_nodes`目录
3. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 📌 模型设置

- 模型的下载链接:https://pan.baidu.com/s/1xx6KEsdyj9bvV5MlGZcvLQ?pwd=ukr2


#### YOLO模型
- 将YOLO模型文件(.pt)放入`ComfyUI/models/yolo/`目录
- 推荐使用YOLOv8模型，如yolov8n.pt或yolov8s.pt

#### RealESRGAN模型
- 将RealESRGAN ONNX模型放入`ComfyUI/models/upscale_models/`目录
- 默认使用RealESRGAN_x4plus.pth

### ⚠️ 重要说明

- **依赖项**：某些节点需要可选依赖项，按需安装：
  - YOLO节点需要`ultralytics`
  - 自动马赛克需要`mediapipe`（有回退方案）
  - RealESRGAN超分辨率需要`realesrgan`（有回退方案）
  - 高级遮罩操作需要`opencv-python`

- **模型文件**：确保将模型文件放在上述指定的正确目录中。

- **内存使用**：对于大图像或批处理，考虑使用内存管理器节点来优化性能。

- **错误处理**：大多数节点包含优雅的错误处理，如果缺少依赖项或找不到模型，会提供信息性消息。

- **冲突**：本节点与comfyui_HiDream-Sampler有依赖冲突，如果发现终端控制台在进行不断刷屏，请检查你是否安装了这个插件！！！


---

## English

A comprehensive collection of custom nodes for ComfyUI, providing rich image processing and mask operation functionality with advanced tools for computer vision, image manipulation, and workflow optimization.

### ✨ Features

This plugin collection includes various practical nodes to help you with:

- 🔍 **YOLO Object Detection & Cropping**:
  - YOLO Detection Crop: Intelligent object detection and cropping with class filtering and confidence settings
  - YOLO Image Paste: Paste detected objects onto other images with smart positioning

- 🎭 **Advanced Mask Operations**:
  - Mask Boolean Operations: Support intersection, union, difference, XOR operations with 9 alignment modes
  - Multi-Mask Operations: Support continuous boolean operations on multiple masks
  - Mask Merge & Composite: Provide mask merging and multi-mask merging nodes
  - Mask Region Expand: Expand mask regions in four directions with gradient transitions
  - Mask Percentage Feathering: Adjust feathering radius based on percentage, can preserve sharp edges
  - Mask Size Alignment: Automatically align masks of different sizes
  - Mask Remove: Remove mask regions from images with various fill methods
  - Mask Switch: Flexible switching between different masks
  - Image Align by Mask: Advanced image alignment using mask-based positioning

- 👤 **Human Body Part Processing**:
  - Body Part Selector: Easily select and process specific human body parts
  - Limb Selector: Support selection of 20+ human body parts like eyes, mouth, arms, etc.
  - Auto Censor: Automatic censoring using pose detection with MediaPipe integration

- 🖼️ **Image Processing**:
  - RealESRGAN Upscaler: High-quality image upscaling using ONNX format RealESRGAN models
  - Image Invert: One-click color inversion with toggle control
  - Desaturate: Basic and advanced image desaturation functionality
  - Pixel Processing: Pixelization, pixel correction, partial pixelization, and pixel enhancement
  - Realistic Noise: Add natural-looking noise with multiple parameter adjustments

- 📝 **Text Processing**:
  - Text Cleaner: Clean and optimize prompt text with tag filtering and sentence filtering

- 📊 **Metadata Tools**:
  - Remove Metadata: Remove metadata from images
  - Load Image (Clear Metadata): Load images while clearing metadata
  - View Metadata: Check image metadata information
  - Batch Remove Metadata: Batch process metadata from multiple images

- 🧠 **Memory Management**:
  - Memory Manager: Optimize memory usage during node processing for improved efficiency

- 🛠️ **Math Tools**:
  - Multi Number Compare: Support comparison operations on multiple values including max, min, median, average, etc.
  - Aspect Ratio Calculator: Smart aspect ratio calculation with multiple constraint modes and preset ratios
  - Math Operations: Basic mathematical operations including arithmetic, power, trigonometric functions, etc.

- 🎬 **Video Processing**:
  - Video Frame Extractor: Extract specific frames from videos with index, percentage, time-based modes
  - Advanced Frame Extraction: Support frame range extraction and batch processing

- 🎭 **Pose & Clothing**:
  - Pose Selector: Rich human pose tag selection with multiple pose categories and batch generation
  - Clothing Selector: Comprehensive clothing tag system with various clothing types and outfit suggestions

- 💻 **Hardware Monitoring**:
  - Hardware Monitor: Real-time monitoring of CPU, GPU, memory usage for workflow performance optimization

- 📝 **Prompt Management**:
  - Prompt Manager: Advanced prompt management with positive/negative prompt separation, weight control, and dynamic prompt adjustments
  - Professional Templates: Specialized prompt nodes for various AI models (Wan2.2, Qwen, etc.)

- 🎨 **Image Filters**:
  - Basic Filters: Brightness, contrast, saturation, sharpness, hue shift, blur, temperature, and gamma adjustments
  - Advanced Filters: Vignette, chromatic aberration, noise, film grain, bloom effects

- 🌈 **Color Processing**:
  - Color Grading: Professional highlight, midtone, shadow color adjustment
  - HDR Effects: Multiple tone mapping algorithms, detail enhancement, local contrast adjustment
  - Skin Enhancement: Intelligent skin smoothing, whitening, blushing, blemish removal, eye enhancement, teeth whitening
  - Artistic Effects: 8 artistic styles including oil painting, watercolor, sketch, cartoon, impressionism, pointillism, engraving, mosaic
  - Selective Color: Precise adjustment for specific color ranges

- 📂 **Batch Processing**:
  - Folder Loader: Support image, text, and paired image-text loading with intelligent caching
  - Batch Loader: File grouping, shuffle, resize, multiple sorting modes
  - Multi-Format Support: Auto-detect and process various image and text formats

### 📂 Node Files and Function Mapping

This plugin contains the following Python module files, each implementing specific function nodes:

| File Name | Node Names | Function Description |
| --------- | ---------- | -------------------- |
| **yoloCrop.py** | 🐳YOLO Detection Crop | Use YOLO models to detect objects in images and intelligently crop, supporting multiple detection classes and cropping modes |
| **yoloPaste.py** | 🐳YOLO Image Paste | Intelligently paste detected objects onto other images with position adjustment support |
| **MaskBoolean.py** | 🐳Mask Boolean Operations, 🐳Multi-Mask Operations | Mask alignment boolean operations and multi-mask continuous boolean operations with various boolean operations |
| **MaskComposite.py** | 🐳Advanced Mask Composite, 🐳Mask-Based Image Composite | Complex mask compositing tools for creating advanced composite effects |
| **MaskMerge.py** | 🐳Multi-Mask Merge, 🐳Mask Merge | Provide simple and advanced mask merging functionality |
| **MaskRegionExpand.py** | 🐳Mask Region Expand | Expand mask regions in four directions with gradient transitions and edge smoothing |
| **MaskFeatherPercentage.py** | 🐳Mask Percentage Feathering | Adjust feathering radius based on image size percentage, can preserve sharp edges |
| **MaskSizeAlign.py** | 🐳Mask Size Alignment | Adjust masks of different sizes to the same size with multiple alignment modes |
| **MaskRemove.py** | 🐳Mask Remove | Remove mask regions from images with multiple fill methods |
| **MaskSwitch.py** | 🐳Mask Switch | Conditional switching between multiple masks |
| **ImageAlignByMask.py** | 🐳Image Align by Mask | Advanced image alignment using mask positioning |
| **BodyPartSelector.py** | 🐳Body Part Selector | Human body main part selector for use with ControlNet and other tools |
| **LimbSelector.py** | 🐳Limb Selector | Detailed human limb part selector supporting 20+ human body parts |
| **auto_censor.py** | 🐳Auto Censor with OpenPose | Automatic censoring using OpenPose detection with MediaPipe integration |
| **RealESRGANUpscaler.py** | 🐳RealESRGAN Upscaler | High-quality image upscaling using ONNX format RealESRGAN models |
| **InvertImage.py** | 🐳Image Invert | One-click color inversion with toggle control |
| **desaturate.py** | 🐳Image Desaturate, 🐳Image Desaturate (V2) | Basic and advanced image desaturation processing |
| **square_pixel.py** | 🐳Pixelization, 🐳Pixel Correction, 🐳Partial Pixelization, 🐳Pixel Enhancement | Multiple pixel art style processing tools |
| **RealisticNoise.py** | 🐳Realistic Noise | Add natural-looking noise with multiple parameter adjustments |
| **TextCleaner.py** | 🐳Text Cleaner | Clean and optimize prompt text with tag filtering and sentence filtering |
| **MetadataUtils.py** | 🐳Remove Metadata, 🐳Load Image (Clear Metadata), 🐳View Metadata, 🐳Batch Remove Metadata | Image metadata processing toolset |
| **MemoryManager.py** | 🐳Memory Manager | Optimize memory usage during node processing |
| **math_tools.py** | 🐳Multi Number Compare, 🐳Aspect Ratio Calculator, 🐳Math Operations | Mathematical tools supporting various operations and numerical processing |
| **VideoFrameExtractor.py** | 🐳Video Frame Extractor, 🐳Advanced Video Frame Extractor, 🐳Video Frame Range Extractor | Video frame extraction and processing tools |
| **PoseSelector.py** | 🐳Pose Selector, 🐳Simple Pose Selector, 🐳Batch Pose Generator | Human pose tag selection and batch generation tools |
| **Clothing_Selector.py** | 🐳Clothing Selector, 🐳Simple Clothing Selector, 🐳Batch Clothing Generator, 🐳Clothing Outfit Suggestion | Clothing tag selection and outfit suggestion tools |
| **hardware_monitor.py** | Hardware Monitor Service | Real-time system hardware monitoring service for performance reference |
| **Prompt_manager.py** | 🐳Prompt Manager | Advanced prompt management with positive/negative separation, weight control, and dynamic adjustments |
| **image_filter.py** | 🐳Image Filter, 🐳Image Filter V2 | Basic and advanced image filter effects with multiple adjustment parameters |
| **HDR.py** | 🐳Color Grading, 🐳HDR, 🐳Skin Enhancement, 🐳Artistic Effects, 🐳Selective Color | Professional color processing node set including HDR, portrait beautification, and artistic stylization |
| **Batch_Loader.py** | 🐳Folder Loader, 🐳Batch Loader | Advanced folder and batch file loader with caching and multiple loading modes |
| **ImagePad.py** | 🐳Image Padding | Resize and pad images based on reference image with multiple alignment modes |
| **ImageBorder.py** | 🐳Border Remover | Intelligently remove image borders with transparency detection and content cropping |
| **maskSolid.py** | 🐳Mask Solidifier | Convert masks to solid (binarize) for enhanced mask effects |
| **CropPaste.py** | 🐳Crop Paste Back | Intelligently paste cropped images back to original position |
| **yoloCropV11.py** | 🐳YOLO v11 Detection Crop | Object detection and cropping using YOLO v11 models |
| **yoloPasteV11.py** | 🐳YOLO v11 Image Paste | Intelligent image pasting using YOLO v11 detection results |
| **Wan22_Prompt.py** | 🐳Wan2.2 Prompt | Specialized prompt processing for Wan2.2 model |
| **QwenPrompt.py** | 🐳Qwen Prompt | Specialized prompt processing for Qwen model |
| **Prompt.py** | 🐳Prompt | Universal prompt processing and management node |
| **mask_color.py** | 🐳Mask Color | Mask color processing and conversion tools |
| **WanVideoReset.py** | 🐳Video Reset | Video sequence reset and processing tools |
| **DigitalJudgment.py** | 🐳Digital Judgment | Numerical comparison and logic judgment tools |
| **iphone.py** | 🐳iPhone Effect | iPhone camera style effect processing |
| **maskBbox.py** | 🐳Mask Bounding Box | Extract bounding box information from masks |

### 📋 Dependencies

#### Core Requirements
- ComfyUI latest version
- Python 3.8+
- torch >= 2.0.0
- Pillow >= 9.0.0
- numpy >= 1.22.0

#### Optional Dependencies (for specific features)
- ultralytics >= 8.0.0 (for YOLO functionality)
- onnxruntime >= 1.14.0 (for RealESRGAN upscaling and ONNX models)
- scipy >= 1.8.0 (for advanced mask processing)
- opencv-python >= 4.5.0 (for mask compositing and auto censor)
- mediapipe >= 0.9.0 (for auto censor pose detection - fallback available)
- realesrgan >= 0.3.0 (for RealESRGAN upscaling - fallback available)
- torchvision >= 0.15.0 (for YOLO paste transform functions)
- requests >= 2.25.0 (for memory manager network functionality)
- psutil >= 5.8.0 (for hardware monitoring functionality)
- pynvml >= 8.0.4 (for NVIDIA GPU monitoring - hardware monitoring functionality)


1. Clone or download this repository
2. Place the folder in ComfyUI's `custom_nodes` directory
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 📌 Model Setup

#### YOLO Models
- Place YOLO model files (.pt) in `ComfyUI/models/yolo/` directory
- Recommended to use YOLOv8 models such as yolov8n.pt or yolov8s.pt
- Download from: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

#### RealESRGAN Models
- Place RealESRGAN ONNX models in `ComfyUI/models/upscale_models/` directory
- Default uses RealESRGAN_x4plus.pth
- Download from: [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)

### ⚠️ Important Notes

- **Dependencies**: Some nodes require optional dependencies. Install them as needed:
  - YOLO nodes require `ultralytics`
  - Auto Censor requires `mediapipe` (has fallback)
  - RealESRGAN Upscaler requires `realesrgan` (has fallback)
  - Advanced mask operations require `opencv-python`

- **Model Files**: Make sure to place model files in the correct directories as specified above.

- **Memory Usage**: For large images or batch processing, consider using the Memory Manager node to optimize performance.

- **Error Handling**: Most nodes include graceful error handling and will provide informative messages if dependencies are missing or models are not found.

### 🚀 Usage Guide

#### YOLO Nodes

1. **YOLO Detection Crop**:
   - Input image and select YOLO model
   - Set confidence threshold and class filtering
   - Support multiple cropping modes: all objects, single object, by class
   - Adjustable crop region expansion range

2. **YOLO Image Paste**:
   - Intelligently paste detected objects onto other images
   - Support automatic alignment and position adjustment

#### Mask Operation Nodes

1. **Mask Boolean Operations**:
   - Provide two masks and select alignment mode (9 different alignment modes)
   - Support intersection, union, difference A-B, difference B-A, XOR, NOT A, NOT B boolean operations
   - Can add X-axis and Y-axis offsets and threshold settings

2. **Multi-Mask Operations**:
   - Support continuous boolean operations on multiple masks
   - Flexible configuration of operation methods for each step

3. **Mask Merge**:
   - Merge multiple masks with different blend modes
   - Adjust opacity and priority

4. **Mask Region Expand**:
   - Expand mask regions in left, top, right, bottom four directions
   - Support expanding black or white regions
   - Provide edge smoothing and gradient transition options

5. **Mask Percentage Feathering**:
   - Adjust feathering radius based on image size percentage
   - Option to preserve sharp edges

6. **Image Align by Mask**:
   - Advanced image alignment using mask positioning
   - Multiple alignment modes and offset controls

#### Image Processing Nodes

1. **RealESRGAN Upscaler**:
   - Use ONNX format RealESRGAN models for image upscaling
   - Support blend parameter adjustment for smooth transition effects

2. **Image Invert**:
   - One-click color inversion
   - Toggle control for easy workflow use

3. **Desaturate**:
   - Basic desaturation and advanced desaturation options
   - Support brightness preservation and different desaturation algorithms

4. **Pixel Processing**:
   - Pixelization: Convert images to pixel art style
   - Pixel Correction: Fix pixel ratio issues
   - Partial Pixelization: Only pixelize specific parts of the image
   - Pixel Enhancement: Enhance quality of pixel art style images

5. **Realistic Noise**:
   - Add natural-looking noise
   - Support multiple noise types, intensity, and random seed settings

6. **Auto Censor**:
   - Automatic censoring using pose detection
   - Support face, chest, and groin area detection
   - Configurable blur strength and censor area size

#### Text Processing Nodes

1. **Text Cleaner**:
   - Remove tags/prompts or entire sentences containing these words
   - Support multiple separators and Chinese/English punctuation recognition

#### Metadata Tool Nodes

1. **Remove Metadata**:
   - Remove metadata from single images
   - Preserve image quality unchanged

2. **Load Image (Clear Metadata)**:
   - Clear metadata while loading images
   - Avoid sensitive information transfer

3. **View Metadata**:
   - Check metadata information contained in images
   - Display in readable format

4. **Batch Remove Metadata**:
   - Process metadata from multiple images
   - Improve work efficiency

#### Memory Management Nodes

1. **Memory Manager**:
   - Optimize memory usage during node processing
   - Monitor and manage memory allocation for improved performance

#### Math Tools Nodes

1. **Multi Number Compare**:
   - Compare multiple numeric values (up to 10 inputs)
   - Support various comparison modes: max, min, median, average, sum, sorted, range
   - Return primary result, secondary result, and detailed information

2. **Aspect Ratio Calculator**:
   - Calculate optimal dimensions based on aspect ratios
   - Support preset ratios (1:1, 4:3, 16:9, etc.) and custom ratios
   - Multiple constraint modes: keep ratio, max total, min total
   - Automatic rounding to specified multiples (e.g., 8, 16, 32)

3. **Math Operations**:
   - Perform basic arithmetic operations: add, subtract, multiply, divide
   - Advanced operations: power, modulo, logarithm, trigonometric functions
   - Support up to 3 input values for complex calculations

#### Video Processing Nodes

1. **Video Frame Extractor**:
   - Extract specific frames from video sequences
   - Support frame index-based extraction
   - Handle frame boundary checking automatically

2. **Advanced Video Frame Extractor**:
   - Multiple extraction modes: index, percentage, time-based
   - FPS-aware time extraction
   - Return both extracted frame and frame index

3. **Video Frame Range Extractor**:
   - Extract frame ranges with customizable step intervals
   - Efficient batch frame processing
   - Support start frame, end frame, and step configuration

#### Pose & Clothing Selection Nodes

1. **Pose Selector**:
   - Rich collection of human pose tags organized by categories
   - Multiple selection boxes per category for flexible combinations
   - Support both Chinese and English output formats
   - Custom tag input support

2. **Simple Pose Selector**:
   - Number-based selection system for easier use
   - Quick preset combinations for common poses
   - Range selection support (e.g., 1-5, 8, 10)

3. **Batch Pose Generator**:
   - Generate multiple pose combinations automatically
   - Weighted category selection for balanced results
   - Customizable batch count and tags per batch

4. **Clothing Selector Series**:
   - Comprehensive clothing tag system covering all garment types
   - Categories: dresses, tops, bottoms, swimwear, sportswear, underwear, outerwear, special
   - Similar functionality to pose selectors with batch generation
   - Outfit suggestion system with style-based recommendations

#### Hardware Monitoring

1. **Hardware Monitor Service**:
   - Real-time CPU, GPU, and memory monitoring
   - NVIDIA GPU specific monitoring (temperature, utilization, VRAM)
   - WebSocket-based real-time updates to ComfyUI interface
   - Automatic fallback when hardware monitoring libraries unavailable

### 🛠️ Troubleshooting

#### Common Issues

1. **Missing Dependencies**: Install required packages using pip
2. **Model Not Found**: Ensure model files are in correct directories
3. **Memory Issues**: Use Memory Manager node for large batch processing
4. **YOLO Detection Issues**: Check model compatibility and image quality

#### Performance Tips

- Use appropriate model sizes for your hardware
- Enable GPU acceleration when available
- Use Memory Manager for large workflows
- Consider batch processing for multiple images

### 📄 License

This project is provided as-is for ComfyUI users. Please respect the licenses of individual dependencies and model files.

### 🤝 Contributing

Feel free to submit issues and pull requests to improve this node collection. All contributions are welcome!

### 📞 Support

If you encounter any issues or have suggestions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Ensure all dependencies are properly installed
- Verify model files are in correct directories

### 🏷️ Tags

ComfyUI, Custom Nodes, Image Processing, Mask Operations, YOLO, Object Detection, Video Processing, Math Tools, Pose Selection, Clothing Tags, Hardware Monitoring, AI Tools, Prompt Management, Image Filters, HDR, Color Grading, Batch Loading, Artistic Effects, Skin Enhancement

---

*Note: This documentation reflects the current state after adding new features in version 1.0.7. All node names, parameters, and descriptions are available in both Chinese and English for international users.* 