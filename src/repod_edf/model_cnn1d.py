# src/repod_edf/model_cnn1d.py
import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    def __init__(self, in_ch, in_len, n_classes=2, channels=(32,64,128), kernels=(7,5,3), dropout=0.25):
        super().__init__()
        layers = []
        c_in = in_ch
        for c, k in zip(channels, kernels):
            layers += [nn.Conv1d(c_in, c, kernel_size=k, padding=k//2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.feat = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, in_len)
            out_len = self.feat(dummy).shape[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (B, C, L)
        z = self.feat(x)
        return self.classifier(z)
