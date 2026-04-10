"""Classification components
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_bn=True):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes (int): Number of output classes.
            in_channels (int): Number of input channels.
            dropout_p (float): Dropout probability for the classifier head.
        """
        super().__init__()
        self.vgg11 = VGG11Encoder(
            num_classes = num_classes,
            in_channels = in_channels,
            dropout_p = dropout_p,
            use_bn = use_bn
            
        )

    def forward(self, x: torch.Tensor):
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        return self.vgg11(x, return_features=False)