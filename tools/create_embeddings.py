import os

import jax
import matplotlib.pyplot as plt
import numpy as np
from medclip.configuration_hybrid_clip import HybridCLIPConfig
from medclip.modeling_hybrid_clip import FlaxHybridCLIP
from PIL import Image
from torchvision.transforms import (ConvertImageDtype, Normalize, Resize,
                                    ToTensor)
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, CLIPProcessor


def main():
    model = FlaxHybridCLIP.from_pretrained("flax-community/medclip-roco")
    vision_model_name = "openai/clip-vit-base-patch32"
    img_dir = "/Users/kaumad/Documents/coding/hf-flax/demo/medclip-roco/images"

    processor = CLIPProcessor.from_pretrained(vision_model_name)

    img_list = os.listdir(img_dir)
    embeddings = []

    for idx, img_path in enumerate(img_list):
        if idx % 10 == 0:
            print(f"{idx} images processed")
        img = Image.open(os.path.join(img_dir, img_path)).convert('RGB')
        inputs = processor(images=img, return_tensors="jax", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].transpose(0, 2, 3, 1)
        img_vec = model.get_image_features(**inputs)
        img_vec = np.array(img_vec).reshape(-1).tolist()
        embeddings.append(img_vec)
        
if __name__=='__main__':
    main()
