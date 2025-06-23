# 🐳 Pond Nodes for ComfyUI

[English](#english) | [中文](#chinese)

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

### 💾 安装方法

#### 方法1：使用ComfyUI Manager (推荐)

1. 在ComfyUI中安装ComfyUI Manager
2. 在Custom Nodes选项卡中搜索"Pond Nodes"并安装

#### 方法2：手动安装

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

### 💾 Installation

#### Method 1: Using ComfyUI Manager (Recommended)

1. Install ComfyUI Manager in ComfyUI
2. Search for "Pond Nodes" in the Custom Nodes tab and install

#### Method 2: Manual Installation

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

---

*Note: This documentation reflects the current state after translation of all Chinese text to English. All node names, parameters, and descriptions are now in English for international users.* 