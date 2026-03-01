"""
BeautyNet 模型定义
- 图像分支: EfficientNet-B0 (ImageNet 预训练)
- Landmark 分支: MLP (478*3 → 256 → 128)
- 融合 → 分布预测头 (5 bins, 对应 1-5 分)
"""

import torch
import torch.nn as nn
import timm


class LandmarkBranch(nn.Module):
    def __init__(self, n_landmarks=478, in_dim=3, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_landmarks * in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class BeautyNet(nn.Module):
    def __init__(self, n_bins=5, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        img_feat_dim = self.backbone.num_features

        self.landmark_branch = LandmarkBranch(n_landmarks=478, in_dim=3, hidden=256, out_dim=128)

        fused_dim = img_feat_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.dist_head = nn.Linear(512, n_bins)
        self.reg_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, img, landmarks):
        img_feat = self.backbone(img)
        lm_feat = self.landmark_branch(landmarks)
        fused = self.fusion(torch.cat([img_feat, lm_feat], dim=1))
        dist = torch.softmax(self.dist_head(fused), dim=1)
        score = self.reg_head(fused)
        return dist, score
