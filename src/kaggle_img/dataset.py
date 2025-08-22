# src/kaggle_img/dataset.py
import os, glob, random
from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset, random_split


class SpectrogramFolder(Dataset):
    def __init__(self, root, transform=None):
        # expects structure root/class_x/*.png
        self.root = root
        self.paths = []
        self.labels = []
        self.transform = transform
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for c in classes:
            for p in glob.glob(os.path.join(root, c, "**", "*.png"), recursive=True):
                self.paths.append(p)
                self.labels.append(self.class_to_idx[c])

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]
