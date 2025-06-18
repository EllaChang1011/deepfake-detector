import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_lora import CLIPWithLoRA
from dataset import RealFakeFrameDataset
from utils.transforms import get_default_transform
from utils.split import split_real_fake_dirs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPWithLoRA().to(DEVICE)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

splits = split_real_fake_dirs("datasets/Real_youtube", "datasets/FaceSwap", "datasets/NeuralTextures")
transform = get_default_transform()

train_set = RealFakeFrameDataset(splits["train"][0], splits["train"][1], transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels, video_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        pixel_values = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(pixel_values)
        loss = criterion(logits, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "clip_lora.pth")
