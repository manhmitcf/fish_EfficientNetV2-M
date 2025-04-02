
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_V2_M_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=8):  # Số lớp đầu ra tùy chỉnh
        super(FishClassifier, self).__init__()
        # Sử dụng EfficientNetV2-M pretrained trên ImageNet
        self.efficientnet = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)

        # Freeze các layers ban đầu (tùy chọn)
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Thay thế Fully Connected Layer
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout để giảm overfitting
            nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)  # Tùy chỉnh số lớp đầu ra
        )

    def forward(self, x):
        return self.efficientnet(x)