# src/repod_edf/model_transformer.py
import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, n_tokens, feat_dim, d_model=128, nhead=4, nlayers=2, dim_ff=256, dropout=0.2, n_classes=2):
        super().__init__()
        self.input = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.pos = PositionalEncoding(d_model)

    def forward(self, x):
        # x: (B, T, F)
        z = self.input(x)
        z = self.pos(z)
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.cls(z)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T]
