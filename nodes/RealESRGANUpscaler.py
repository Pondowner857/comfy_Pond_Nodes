import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import os
import folder_paths

# ComfyUI models directory
models_dir = folder_paths.models_dir

class RealESRGANUpscaler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "🐳Pond/image"

    def upscale(self, image, blend):
        # Convert image to numpy
        img_tensor = image
        img = 255. * img_tensor.cpu().numpy()[0]
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        # Specify model path (fixed)
        model_path = os.path.join(models_dir, "real_esrgan_x4_fp16.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # GPU priority, fallback to CPU if not available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Preprocessing
        lr_img = np.array(img).astype(np.float32) / 255.0
        lr_img = np.transpose(lr_img, (2, 0, 1))
        lr_img = np.expand_dims(lr_img, axis=0)

        # Inference
        sr_img = sess.run(
            [output_name],
            {input_name: lr_img}
        )[0]

        # Post-processing
        sr_img = np.squeeze(sr_img, axis=0)
        sr_img = np.clip(sr_img * 255, 0, 255)
        sr_img = np.transpose(sr_img, (1, 2, 0))

        if blend < 1.0:
            orig_resized = img.resize((sr_img.shape[1], sr_img.shape[0]), Image.BICUBIC)
            orig_array = np.array(orig_resized).astype(np.float32)
            sr_img = (1 - blend) * orig_array + blend * sr_img
            sr_img = np.clip(sr_img, 0, 255)

        result = torch.from_numpy(sr_img.astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "RealESRGANUpscaler": RealESRGANUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealESRGANUpscaler": "🐳RealESRGAN Super Resolution"
}