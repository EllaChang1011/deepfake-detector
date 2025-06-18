import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPBase(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", num_classes=1, device="cuda"):
        super().__init__()
        self.device = device
        self.clip = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)

    def forward(self, pixel_values):
        features = self.clip.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def preprocess(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)
