import os
from PIL import Image
import matplotlib.pyplot as plt
from random import sample
import torch
from torch.utils.data import DataLoader

from dataset import RealFakeFrameDataset
from model_lora import CLIPWithLoRA  # 或換成 CLIPBase
from utils.transforms import get_default_transform
from utils.split import split_real_fake_dirs

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
splits = split_real_fake_dirs("datasets/Real_youtube", "datasets/FaceSwap", "datasets/NeuralTextures")
transform = get_default_transform()
test_set = RealFakeFrameDataset(splits["test"][0], splits["test"][1], transform)
test_loader = DataLoader(test_set, batch_size=16)

# Load model
model = CLIPWithLoRA().to(DEVICE)  # 改成 CLIPBase() 如果要看 base
model.load_state_dict(torch.load("clip_lora.pth", map_location=DEVICE))
model.eval()

# Prediction loop
y_pred, y_true = [], []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().tolist()
        y_pred += probs
        y_true += labels.tolist()

# Find mistakes
errors = [(i, p) for i, (p, t) in enumerate(zip(y_pred, y_true)) if int(p > 0.5) != t]
chosen = sample(errors, min(3, len(errors)))

# Create folder to save
save_dir = "figures/mistakes"
os.makedirs(save_dir, exist_ok=True)

# Save each mistaken image
for idx, score in chosen:
    path, label = test_set.samples[idx]
    img = Image.open(path).convert("RGB")

    plt.imshow(img)
    plt.title(f"Pred: {'fake' if score > 0.5 else 'real'} | GT: {'fake' if label == 1 else 'real'}\nScore: {score:.2f}")
    plt.axis("off")

    fname = os.path.basename(path).replace(".png", f"_score{score:.2f}.png")
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path)
    plt.close()

print(f"Saved {len(chosen)} mistaken predictions to '{save_dir}'")

