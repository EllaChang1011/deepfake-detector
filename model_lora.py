import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from peft import get_peft_model
from utils.lora_config import get_clip_lora_config

class CLIPWithLoRA(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda", use_lora=False):
        super().__init__()
        self.device = device
        self.use_lora = use_lora

        self.clip = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        if use_lora:
            config = get_clip_lora_config()
            self.clip = get_peft_model(self.clip, config)

        self.classifier = nn.Linear(self.clip.config.projection_dim, 1).to(device)

    def forward(self, images):
        features = self.clip.get_image_features(pixel_values=images)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
