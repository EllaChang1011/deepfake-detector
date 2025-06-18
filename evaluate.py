import torch
from torch.utils.data import DataLoader
from dataset import RealFakeFrameDataset
from utils.transforms import get_default_transform
from utils.split import split_real_fake_dirs
from utils.metrics import compute_metrics
from model_baseline import CLIPBase
from model_lora import CLIPWithLoRA 

from collections import defaultdict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_type, checkpoint_path):
    if model_type == "base":
        model = CLIPBase().to(DEVICE)
    elif model_type == "lora":
        model = CLIPWithLoRA().to(DEVICE)
    else:
        raise ValueError("model_type must be 'base' or 'lora'")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def evaluate(model, model_name="model"):
    splits = split_real_fake_dirs("datasets/Real_youtube", "datasets/FaceSwap", "datasets/NeuralTextures")
    transform = get_default_transform()
    test_set = RealFakeFrameDataset(splits["test"][0], splits["test"][1], transform)
    test_loader = DataLoader(test_set, batch_size=16)

    all_labels, all_probs, all_vids = [], [], []

    with torch.no_grad():
        for images, labels, vids in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_labels += labels.tolist()
            all_probs += probs
            all_vids += vids

    frame_result = compute_metrics(all_labels, all_probs)
    print(f"[{model_name} - Frame] AUC: {frame_result['auc']:.3f}, ACC: {frame_result['acc']:.3f}, F1: {frame_result['f1']:.3f}")

    # video-level
    vid_probs = defaultdict(list)
    vid_labels = {}
    for prob, label, vid in zip(all_probs, all_labels, all_vids):
        vid_probs[vid].append(prob)
        vid_labels[vid] = label

    video_true, video_avg_prob = [], []
    for vid in vid_probs:
        avg_prob = sum(vid_probs[vid]) / len(vid_probs[vid])
        video_avg_prob.append(avg_prob)
        video_true.append(vid_labels[vid])

    video_result = compute_metrics(video_true, video_avg_prob)
    print(f"[{model_name} - Video] AUC: {video_result['auc']:.3f}, ACC: {video_result['acc']:.3f}, F1: {video_result['f1']:.3f}")

    # ROC plot
    fpr_f, tpr_f, _ = roc_curve(all_labels, all_probs)
    auc_f = auc(fpr_f, tpr_f)

    fpr_v, tpr_v, _ = roc_curve(video_true, video_avg_prob)
    auc_v = auc(fpr_v, tpr_v)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(fpr_f, tpr_f, color="orange", label=f"AUC = {auc_f:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{model_name} Frame-level ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fpr_v, tpr_v, color="orange", label=f"AUC = {auc_v:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"{model_name} Video-level ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/roc_curve_{model_name}.png")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="base",
        choices=["base", "lora"],
        help="Which model to evaluate: base or lora"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model weights (e.g., clip_base.pth)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional name for saving ROC plot (default uses model_type)"
    )

    args = parser.parse_args()
    model = load_model(args.model_type, args.ckpt)
    output_name = args.output_name or args.model_type
    evaluate(model, model_name=output_name)
