import torch
import numpy as np
from PIL import Image
import random

class RealisticNoiseNode:
    """
    Convert AI-generated image noise into more realistic camera noise
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "photon_noise": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "thermal_noise": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "read_noise": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "color_noise": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "iso_simulation": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 12800,
                    "step": 100,
                    "display": "slider"
                }),
                "detail_preservation": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32-1,
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_realistic_noise"
    CATEGORY = "🐳Pond/image"
    
    def add_realistic_noise(self, image, noise_strength, photon_noise, thermal_noise, read_noise, color_noise, iso_simulation, detail_preservation, seed=-1):
        # Set random seed
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Get image dimensions
        batch_size, height, width, channels = image.shape
        device = image.device
        
        # Copy image to avoid modifying original data
        noisy_image = image.clone()
        
        # ISO gain simulation
        iso_factor = iso_simulation / 100.0
        overall_strength = noise_strength * iso_factor * 0.1
        
        # 1. Add photon noise (Shot Noise) - Poisson distribution
        if photon_noise > 0:
            # Simulate photon noise, more prominent in dark areas
            luminance = torch.mean(noisy_image, dim=3, keepdim=True)
            # Stronger noise in dark areas
            dark_mask = 1.0 - luminance
            
            # Use Gaussian approximation of Poisson distribution
            photon_noise_tensor = torch.randn_like(noisy_image) * photon_noise * overall_strength
            photon_noise_tensor = photon_noise_tensor * dark_mask * 2.0
            noisy_image = noisy_image + photon_noise_tensor
        
        # 2. Add thermal noise - Fixed pattern noise
        if thermal_noise > 0:
            # Create fixed pattern noise
            thermal_pattern = torch.randn(1, height, width, 1, device=device) * thermal_noise * overall_strength * 0.5
            thermal_pattern = thermal_pattern.expand(batch_size, -1, -1, channels)
            
            # Add slight temporal variation
            temporal_variation = torch.randn_like(noisy_image) * thermal_noise * overall_strength * 0.1
            thermal_noise_tensor = thermal_pattern + temporal_variation
            
            noisy_image = noisy_image + thermal_noise_tensor
        
        # 3. Add read noise - Gaussian distribution
        if read_noise > 0:
            read_noise_tensor = torch.randn_like(noisy_image) * read_noise * overall_strength * 0.8
            noisy_image = noisy_image + read_noise_tensor
        
        # 4. Add color noise
        if color_noise > 0:
            # Add different noise strength for each color channel
            color_noise_tensor = torch.randn_like(noisy_image)
            # R channel noise slightly stronger
            color_noise_tensor[:, :, :, 0] *= color_noise * overall_strength * 1.2
            # G channel standard noise
            color_noise_tensor[:, :, :, 1] *= color_noise * overall_strength * 1.0
            # B channel strongest noise (simulating sensor characteristics)
            color_noise_tensor[:, :, :, 2] *= color_noise * overall_strength * 1.4
            
            noisy_image = noisy_image + color_noise_tensor
        
        # 5. Simulate sensor response nonlinearity
        # Reduce noise in highlight areas (simulating sensor saturation)
        highlights = torch.clamp(luminance - 0.8, 0, 1) * 5.0
        noisy_image = torch.lerp(noisy_image, image, highlights)
        
        # 6. Apply detail preservation
        if detail_preservation > 0:
            # Use edge detection to preserve details
            # Simple edge detection
            dx = torch.abs(image[:, 1:, :, :] - image[:, :-1, :, :])
            dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
            
            # Create edge mask
            edge_mask = torch.zeros_like(image)
            edge_mask[:, 1:, :, :] += dx
            edge_mask[:, :-1, :, :] += dx
            edge_mask[:, :, 1:, :] += dy
            edge_mask[:, :, :-1, :] += dy
            
            edge_mask = torch.clamp(edge_mask * 5.0, 0, 1)
            
            # Blend original image in edge areas to preserve details
            noisy_image = torch.lerp(noisy_image, image, edge_mask * detail_preservation)
        
        # 7. Add slight salt and pepper noise
        if random.random() < 0.3:  # 30% probability to add salt and pepper noise
            salt_pepper_prob = 0.001 * overall_strength
            salt_mask = torch.rand_like(noisy_image[:, :, :, 0]) < salt_pepper_prob
            pepper_mask = torch.rand_like(noisy_image[:, :, :, 0]) < salt_pepper_prob
            
            salt_mask = salt_mask.unsqueeze(-1).expand(-1, -1, -1, channels)
            pepper_mask = pepper_mask.unsqueeze(-1).expand(-1, -1, -1, channels)
            
            noisy_image[salt_mask] = 1.0
            noisy_image[pepper_mask] = 0.0
        
        # 8. Final processing
        # Add slight Gaussian blur to simulate sensor low-pass filter effect
        if overall_strength > 0.5:
            # Simple 3x3 Gaussian blur
            kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=device) / 16.0
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            
            # Apply blur to each channel
            blurred = torch.zeros_like(noisy_image)
            for c in range(channels):
                channel = noisy_image[:, :, :, c:c+1].permute(0, 3, 1, 2)
                blurred_channel = torch.nn.functional.conv2d(channel, kernel, padding=1)
                blurred[:, :, :, c] = blurred_channel.permute(0, 2, 3, 1).squeeze(-1)
            
            # Slightly blend blur effect
            blur_strength = min(0.3, overall_strength * 0.2)
            noisy_image = torch.lerp(noisy_image, blurred, blur_strength)
        
        # Clip to valid range
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        
        return (noisy_image,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # If seed is -1 (random), mark as changed every time
        if kwargs.get('seed', -1) == -1:
            return float("NaN")
        return None


# For node registration
NODE_CLASS_MAPPINGS = {
    "RealisticNoiseNode": RealisticNoiseNode
}

# Node display name in UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "RealisticNoiseNode": "🐳Realistic Noise"
}