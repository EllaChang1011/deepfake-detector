import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

from model_lora import CLIPWithLoRA
from dataset import RealFakeFrameDataset
from utils.split import split_real_fake_dirs
from utils.transforms import get_default_transform
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_default_transform()
splits = split_real_fake_dirs("datasets/Real_youtube", "datasets/FaceSwap", "datasets/NeuralTextures")
test_set = RealFakeFrameDataset(splits["test"][0], splits["test"][1], transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

os.makedirs("results", exist_ok=True)

def evaluate_and_save(model_path, use_lora, frame_out, video_out):
    model = CLIPWithLoRA(use_lora=use_lora).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_videos = [], [], []

    with torch.no_grad():
        for images, labels, video_ids in tqdm(test_loader, desc=f"Evaluating {os.path.basename(model_path)}"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            all_videos.extend(video_ids)

    df = pd.DataFrame({
        "video": all_videos,
        "label": all_labels,
        "score": all_preds
    })

    df.to_csv(frame_out, index=False)

    video_df = df.groupby("video").mean()
    video_df["label"] = df.groupby("video")["label"].first()
    video_df.to_csv(video_out)

    auc = roc_auc_score(video_df["label"], video_df["score"])
    acc = accuracy_score(video_df["label"], video_df["score"] > 0.5)
    f1 = f1_score(video_df["label"], video_df["score"] > 0.5)
    print(f"{os.path.basename(model_path)} - AUC: {auc:.3f}, ACC: {acc:.3f}, F1: {f1:.3f}")

# Base model
evaluate_and_save(
    model_path="clip_base.pth",
    use_lora=False,
    frame_out="results/scores_base_frame.csv",
    video_out="results/scores_base_video.csv"
)

# LoRA model
evaluate_and_save(
    model_path="clip_lora.pth",
    use_lora=True,
    frame_out="results/scores_lora_frame.csv",
    video_out="results/scores_lora_video.csv"
)
