"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()

        # Vgg11 architecture
        self.backbone = VGG11Encoder(num_classes=37, in_channels=in_channels)

        # Removing classifier head from the backbone
        self.backbone.classifier = nn.Identity()

        # Locationzation head
        self.regressor = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p =dropout_p),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            CustomDropout(p =dropout_p),
            nn.Linear(512,4)
        )

        self.img_size: int = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # Getting Features from VGGNET11
        _ , features= self.backbone(x, return_features= True)
        bottleneck = features['block5']
        bottleneck = torch.flatten(bottleneck,1)

        # Regress bounding box
        x = self.regressor(bottleneck)

        # Scaling to pixel size  
        x = torch.sigmoid(x)* self.img_size

        return x