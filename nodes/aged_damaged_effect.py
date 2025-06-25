import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import cv2

class AgedDamagedEffect:
    """
    ComfyUI节点：在保持颜色的情况下为图像添加老旧/战损效果
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "effect_type": (["aged", "damaged", "both"],),
                "intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "scratch_amount": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
                "stain_amount": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1
                }),
                "edge_wear": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "noise_level": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "🐳Pond/image"
    
    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        # tensor shape: [batch, height, width, channels]
        i = 255. * tensor.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    
    def pil_to_tensor(self, pil_image):
        """将PIL图像转换为tensor"""
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)
    
    def add_scratches(self, img, num_scratches, intensity):
        """添加划痕效果"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        for _ in range(num_scratches):
            # 随机生成划痕的起点和终点
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-100, 100)
            y2 = y1 + random.randint(-100, 100)
            
            # 划痕的宽度和透明度
            scratch_width = random.randint(1, 3)
            opacity = int(255 * intensity * random.uniform(0.3, 0.7))
            
            # 使用半透明的灰色绘制划痕
            draw.line([(x1, y1), (x2, y2)], 
                     fill=(128, 128, 128, opacity), 
                     width=scratch_width)
        
        return img
    
    def add_stains(self, img, num_stains, intensity):
        """添加污渍效果"""
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = img.size
        
        for _ in range(num_stains):
            # 随机位置和大小
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(10, 50)
            
            # 随机形状的污渍
            points = []
            num_points = random.randint(5, 10)
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                r = radius * random.uniform(0.7, 1.3)
                px = x + int(r * np.cos(angle))
                py = y + int(r * np.sin(angle))
                points.append((px, py))
            
            # 半透明的深色污渍
            opacity = int(255 * intensity * random.uniform(0.1, 0.3))
            draw.polygon(points, fill=(60, 50, 40, opacity))
        
        # 模糊处理使污渍更自然
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
        
        # 合并污渍层
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        return img
    
    def add_edge_wear(self, img, wear_intensity):
        """添加边缘磨损效果"""
        width, height = img.size
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        # 创建边缘磨损遮罩
        wear_width = int(min(width, height) * 0.1 * wear_intensity)
        
        # 使用不规则形状创建磨损边缘
        for i in range(0, width + height, 5):
            if i < width:
                # 上边缘
                variation = random.randint(0, wear_width)
                draw.ellipse([i-variation, -variation, i+variation, variation], fill=200)
                # 下边缘
                variation = random.randint(0, wear_width)
                draw.ellipse([i-variation, height-variation, i+variation, height+variation], fill=200)
            
            if i < height:
                # 左边缘
                variation = random.randint(0, wear_width)
                draw.ellipse([-variation, i-variation, variation, i+variation], fill=200)
                # 右边缘
                variation = random.randint(0, wear_width)
                draw.ellipse([width-variation, i-variation, width+variation, i+variation], fill=200)
        
        # 模糊处理
        mask = mask.filter(ImageFilter.GaussianBlur(radius=wear_width//2))
        
        # 应用遮罩
        img_rgba = img.convert('RGBA')
        img_rgba.putalpha(mask)
        
        return img_rgba
    
    def add_noise(self, img, noise_level):
        """添加噪点效果"""
        img_array = np.array(img)
        
        # 生成噪声
        noise = np.random.normal(0, noise_level * 30, img_array.shape)
        
        # 添加噪声但保持颜色平衡
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def add_damage_cracks(self, img, intensity):
        """添加裂纹效果（战损）"""
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = img.size
        
        num_cracks = int(10 * intensity)
        
        for _ in range(num_cracks):
            # 裂纹起点
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            
            # 生成裂纹路径
            points = [(start_x, start_y)]
            current_x, current_y = start_x, start_y
            
            crack_length = random.randint(30, 150)
            for _ in range(crack_length):
                # 随机方向延伸
                angle = random.uniform(0, 2 * np.pi)
                step = random.randint(2, 5)
                current_x += int(step * np.cos(angle))
                current_y += int(step * np.sin(angle))
                
                # 保持在图像范围内
                current_x = max(0, min(width-1, current_x))
                current_y = max(0, min(height-1, current_y))
                
                points.append((current_x, current_y))
                
                # 偶尔分叉
                if random.random() < 0.2:
                    branch_length = random.randint(10, 30)
                    branch_angle = angle + random.uniform(-np.pi/3, np.pi/3)
                    branch_x, branch_y = current_x, current_y
                    
                    for _ in range(branch_length):
                        branch_x += int(3 * np.cos(branch_angle))
                        branch_y += int(3 * np.sin(branch_angle))
                        draw.line([(current_x, current_y), (branch_x, branch_y)], 
                                fill=(40, 40, 40, int(200 * intensity)), width=1)
            
            # 绘制主裂纹
            for i in range(len(points)-1):
                draw.line([points[i], points[i+1]], 
                         fill=(40, 40, 40, int(200 * intensity)), 
                         width=random.randint(1, 2))
        
        # 合并裂纹层
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        return img
    
    def apply_effect(self, image, effect_type, intensity, scratch_amount, 
                    stain_amount, edge_wear, noise_level):
        """应用老旧/战损效果"""
        # 转换为PIL图像
        pil_image = self.tensor_to_pil(image)
        
        # 确保是RGBA格式
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        # 根据效果类型应用不同的处理
        if effect_type in ["aged", "both"]:
            # 老旧效果
            pil_image = self.add_stains(pil_image, stain_amount, intensity)
            pil_image = self.add_scratches(pil_image, scratch_amount, intensity * 0.7)
            pil_image = self.add_edge_wear(pil_image, edge_wear)
            pil_image = self.add_noise(pil_image, noise_level * intensity)
        
        if effect_type in ["damaged", "both"]:
            # 战损效果
            pil_image = self.add_damage_cracks(pil_image, intensity)
            pil_image = self.add_scratches(pil_image, scratch_amount * 2, intensity)
            if effect_type == "damaged":
                pil_image = self.add_edge_wear(pil_image, edge_wear * 1.5)
        
        # 最终转换回RGB
        if pil_image.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        
        # 转换回tensor
        return (self.pil_to_tensor(pil_image),)

# ComfyUI注册
NODE_CLASS_MAPPINGS = {
    "AgedDamagedEffect": AgedDamagedEffect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgedDamagedEffect": "🐳Aged/Damaged Effect"
}