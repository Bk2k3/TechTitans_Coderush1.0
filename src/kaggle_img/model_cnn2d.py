# src/kaggle_img/model_cnn2d.py
import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str, n_classes=2, dropout=0.2, pretrained=True):
    name = name.lower()
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_classes))
        return m
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, n_classes)
        return m
    elif name in ['vit_b_16','vit_b_32']:
        w = getattr(models, f"ViT_B_16_Weights" if name=='vit_b_16' else "ViT_B_32_Weights")
        m = getattr(models, name)(weights=w.DEFAULT if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
        return m
    else:
        raise ValueError(f"Unknown backbone {name}")
