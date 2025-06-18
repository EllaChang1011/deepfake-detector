import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class RealFakeFrameDataset(Dataset):
    def __init__(self, real_subdirs, fake_subdirs, transform=None):
        self.real_images = []
        self.fake_images = []
        self.transform = transform

        for d in real_subdirs or []:
            self.real_images += glob.glob(os.path.join(d, "*.png"))
        for d in fake_subdirs or []:
            self.fake_images += glob.glob(os.path.join(d, "*.png"))

        self.samples = [(p, 0) for p in self.real_images] + [(p, 1) for p in self.fake_images]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        video_id = os.path.basename(os.path.dirname(path))
        return img, label, video_id

