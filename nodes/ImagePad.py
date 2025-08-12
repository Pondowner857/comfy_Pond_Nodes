import torch
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None


class ImageResizeAndPadWithReference:
    """
    根据参考图像尺寸对图像进行等比例缩放并填充
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "resize_mode": (["fit", "fill", "stretch", "none"],),
                "padding_color": (["white", "black", "gray", "red", "green", "blue", "transparent"],),
                "position": (["center", "top-left", "top-center", "top-right", 
                            "middle-left", "middle-right", 
                            "bottom-left", "bottom-center", "bottom-right"],),
            },
            "optional": {
                "custom_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "placeholder": "Hex color code (e.g., #FF5733)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize_and_pad"
    CATEGORY = "🐳Pond/image"
    
    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (255, 255, 255)
    
    def get_background_color(self, padding_color, custom_color=None):
        """获取背景颜色"""
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "transparent": (255, 255, 255, 0),
        }
        
        if padding_color == "custom" and custom_color:
            return self.hex_to_rgb(custom_color)
        
        return color_map.get(padding_color, (255, 255, 255))
    
    def calculate_position(self, position, target_width, target_height, img_width, img_height):
        """计算图像放置位置"""
        positions = {
            "center": ((target_width - img_width) // 2, (target_height - img_height) // 2),
            "top-left": (0, 0),
            "top-center": ((target_width - img_width) // 2, 0),
            "top-right": (target_width - img_width, 0),
            "middle-left": (0, (target_height - img_height) // 2),
            "middle-right": (target_width - img_width, (target_height - img_height) // 2),
            "bottom-left": (0, target_height - img_height),
            "bottom-center": ((target_width - img_width) // 2, target_height - img_height),
            "bottom-right": (target_width - img_width, target_height - img_height),
        }
        
        x, y = positions.get(position, positions["center"])
        x = max(0, min(x, target_width - img_width))
        y = max(0, min(y, target_height - img_height))
        
        return x, y
    
    def resize_image(self, img_np, target_width, target_height, mode):
        """根据模式缩放图像"""
        img_height, img_width = img_np.shape[:2]
        
        if mode == "none":
            # 不缩放，如果太大则裁剪
            if img_width > target_width or img_height > target_height:
                crop_width = min(img_width, target_width)
                crop_height = min(img_height, target_height)
                start_x = (img_width - crop_width) // 2
                start_y = (img_height - crop_height) // 2
                img_np = img_np[start_y:start_y + crop_height, start_x:start_x + crop_width]
            return img_np
        
        elif mode == "stretch":
            # 拉伸到目标尺寸
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return np.array(img_pil)
        
        elif mode == "fit":
            # 等比例缩放，确保完全在目标尺寸内
            width_ratio = target_width / img_width
            height_ratio = target_height / img_height
            scale_ratio = min(width_ratio, height_ratio)
            
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return np.array(img_pil)
        
        elif mode == "fill":
            # 等比例缩放，填充满目标尺寸（可能会裁剪）
            width_ratio = target_width / img_width
            height_ratio = target_height / img_height
            scale_ratio = max(width_ratio, height_ratio)
            
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_np = np.array(img_pil)
            
            # 居中裁剪到目标尺寸
            start_x = (new_width - target_width) // 2
            start_y = (new_height - target_height) // 2
            img_np = img_np[start_y:start_y + target_height, start_x:start_x + target_width]
            return img_np
        
        return img_np
    
    def resize_and_pad(self, image, reference, resize_mode, padding_color, position, custom_color=None):
        # 获取参考图像尺寸
        ref_batch, ref_height, ref_width, ref_channels = reference.shape
        target_width = ref_width
        target_height = ref_height
        
        # 获取输入图像信息
        batch_size, img_height, img_width, channels = image.shape
        
        # 获取背景颜色
        bg_color = self.get_background_color(padding_color, custom_color)
        
        results = []
        
        for i in range(batch_size):
            # 将tensor转换为numpy数组
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # 根据模式缩放图像
            img_np = self.resize_image(img_np, target_width, target_height, resize_mode)
            current_height, current_width = img_np.shape[:2]
            
            # 如果是stretch或fill模式，图像已经是目标尺寸，直接返回
            if resize_mode in ["stretch", "fill"]:
                result_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
                results.append(result_tensor)
                continue
            
            # 创建目标尺寸的背景
            if padding_color == "transparent" and channels == 4:
                background = np.ones((target_height, target_width, 4), dtype=np.uint8)
                background[:, :, :3] = 255
                background[:, :, 3] = 0
                
                if img_np.shape[2] == 3:
                    img_rgba = np.ones((current_height, current_width, 4), dtype=np.uint8)
                    img_rgba[:, :, :3] = img_np
                    img_rgba[:, :, 3] = 255
                    img_np = img_rgba
            else:
                if channels == 4:
                    background = np.ones((target_height, target_width, 4), dtype=np.uint8)
                    background[:, :, :3] = bg_color[:3]
                    background[:, :, 3] = 255
                else:
                    background = np.ones((target_height, target_width, 3), dtype=np.uint8)
                    background[:, :] = bg_color[:3]
            
            # 计算放置位置
            paste_x, paste_y = self.calculate_position(
                position, target_width, target_height, current_width, current_height
            )
            
            # 将图像粘贴到背景上
            end_x = min(paste_x + current_width, target_width)
            end_y = min(paste_y + current_height, target_height)
            background[paste_y:end_y, paste_x:end_x] = img_np[:end_y-paste_y, :end_x-paste_x]
            
            # 转换回tensor格式
            result_tensor = torch.from_numpy(background.astype(np.float32) / 255.0)
            results.append(result_tensor)
        
        # 堆叠所有结果
        output = torch.stack(results)
        
        return (output, target_width, target_height)


class ImageResizeAndPadFixed:
    """
    使用固定尺寸对图像进行等比例缩放并填充
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "resize_mode": (["fit", "fill", "stretch", "none"],),
                "padding_color": (["white", "black", "gray", "red", "green", "blue", "custom"],),
                "position": (["center", "top-left", "top-center", "top-right", 
                            "middle-left", "middle-right", 
                            "bottom-left", "bottom-center", "bottom-right"],),
            },
            "optional": {
                "custom_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "placeholder": "Hex color code (e.g., #FF5733)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_and_pad"
    CATEGORY = "🐳Pond/image"
    
    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (255, 255, 255)
    
    def get_background_color(self, padding_color, custom_color=None):
        """获取背景颜色"""
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
        }
        
        if padding_color == "custom" and custom_color:
            return self.hex_to_rgb(custom_color)
        
        return color_map.get(padding_color, (255, 255, 255))
    
    def calculate_position(self, position, target_width, target_height, img_width, img_height):
        """计算图像放置位置"""
        positions = {
            "center": ((target_width - img_width) // 2, (target_height - img_height) // 2),
            "top-left": (0, 0),
            "top-center": ((target_width - img_width) // 2, 0),
            "top-right": (target_width - img_width, 0),
            "middle-left": (0, (target_height - img_height) // 2),
            "middle-right": (target_width - img_width, (target_height - img_height) // 2),
            "bottom-left": (0, target_height - img_height),
            "bottom-center": ((target_width - img_width) // 2, target_height - img_height),
            "bottom-right": (target_width - img_width, target_height - img_height),
        }
        
        x, y = positions.get(position, positions["center"])
        x = max(0, min(x, target_width - img_width))
        y = max(0, min(y, target_height - img_height))
        
        return x, y
    
    def resize_image(self, img_np, target_width, target_height, mode):
        """根据模式缩放图像"""
        img_height, img_width = img_np.shape[:2]
        
        if mode == "none":
            # 不缩放，如果太大则裁剪
            if img_width > target_width or img_height > target_height:
                crop_width = min(img_width, target_width)
                crop_height = min(img_height, target_height)
                start_x = (img_width - crop_width) // 2
                start_y = (img_height - crop_height) // 2
                img_np = img_np[start_y:start_y + crop_height, start_x:start_x + crop_width]
            return img_np
        
        elif mode == "stretch":
            # 拉伸到目标尺寸
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return np.array(img_pil)
        
        elif mode == "fit":
            # 等比例缩放，确保完全在目标尺寸内
            width_ratio = target_width / img_width
            height_ratio = target_height / img_height
            scale_ratio = min(width_ratio, height_ratio)
            
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return np.array(img_pil)
        
        elif mode == "fill":
            # 等比例缩放，填充满目标尺寸（可能会裁剪）
            width_ratio = target_width / img_width
            height_ratio = target_height / img_height
            scale_ratio = max(width_ratio, height_ratio)
            
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_np = np.array(img_pil)
            
            # 居中裁剪到目标尺寸
            start_x = (new_width - target_width) // 2
            start_y = (new_height - target_height) // 2
            img_np = img_np[start_y:start_y + target_height, start_x:start_x + target_width]
            return img_np
        
        return img_np
    
    def resize_and_pad(self, image, target_width, target_height, resize_mode, padding_color, position, custom_color=None):
        batch_size, img_height, img_width, channels = image.shape
        
        # 获取背景颜色
        bg_color = self.get_background_color(padding_color, custom_color)
        
        results = []
        
        for i in range(batch_size):
            # 将tensor转换为numpy数组
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # 根据模式缩放图像
            img_np = self.resize_image(img_np, target_width, target_height, resize_mode)
            current_height, current_width = img_np.shape[:2]
            
            # 如果是stretch或fill模式，图像已经是目标尺寸，直接返回
            if resize_mode in ["stretch", "fill"]:
                result_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
                results.append(result_tensor)
                continue
            
            # 创建目标尺寸的背景
            if channels == 4:
                background = np.ones((target_height, target_width, 4), dtype=np.uint8)
                background[:, :, :3] = bg_color[:3]
                background[:, :, 3] = 255
            else:
                background = np.ones((target_height, target_width, 3), dtype=np.uint8)
                background[:, :] = bg_color[:3]
            
            # 计算放置位置
            paste_x, paste_y = self.calculate_position(
                position, target_width, target_height, current_width, current_height
            )
            
            # 将图像粘贴到背景上
            end_x = min(paste_x + current_width, target_width)
            end_y = min(paste_y + current_height, target_height)
            background[paste_y:end_y, paste_x:end_x] = img_np[:end_y-paste_y, :end_x-paste_x]
            
            # 转换回tensor格式
            result_tensor = torch.from_numpy(background.astype(np.float32) / 255.0)
            results.append(result_tensor)
        
        # 堆叠所有结果
        output = torch.stack(results)
        
        return (output,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageResizeAndPadWithReference": ImageResizeAndPadWithReference,
    "ImageResizeAndPadFixed": ImageResizeAndPadFixed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizeAndPadWithReference": "🐳图像背景填充",
    "ImageResizeAndPadFixed": "🐳图像背景填充V2",
}