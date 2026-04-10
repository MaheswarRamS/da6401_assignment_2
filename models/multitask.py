"""Unified multi-task model
"""

from typing import Dict

import torch
import torch.nn as nn
import os
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import DecoderBlock
from .segmentation import VGG11UNet
from .classification import VGG11Classifier
from.localization import VGG11Localizer
import gdown

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,dropout_p: float = 0.5, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/segmentation.pth"):
        """
            Unified Multitask model

            Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        gdown.download(id="1MP3lCBkNCUz68Elk9TONFkn-UUVYfwCf", output=classifier_path, quiet=False)
        gdown.download(id="1o7-AN2HGuA1li_vZ3qigMkiiRCYaihjB", output=localizer_path, quiet=False)
        gdown.download(id="1iMx8MslugB5OwtDMjayWlMIxzwA9rCTe", output=unet_path, quiet=False)   


        # Shared Backbone
        self.classifier = VGG11Classifier(num_classes = num_breeds, in_channels=in_channels, dropout_p=dropout_p)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'), strict=False)
        self.localizer = VGG11Localizer(in_channels=in_channels, dropout_p=dropout_p)
        self.localizer.load_state_dict(torch.load(localizer_path, map_location='cpu'), strict=False)
        self.segmentation = VGG11UNet(num_classes=seg_classes, in_channels=in_channels,dropout_p=dropout_p)
        self.segmentation.load_state_dict(torch.load(unet_path, map_location='cpu'), strict=False)

        self.img_size: int = 224

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Forward pass for multi-task model.
                Args:
                x: Input tensor of shape [B, in_channels, H, W].
                Returns:
                    A dict with keys:
                    - 'classification': [B, num_breeds] logits tensor.
                    - 'localization': [B, 4] bounding box tensor.
                    - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
             """
            # Get features from backbone
            cls_out = self.classifier(x)
            # Localization
            loc_out = self.localizer(x)

            seg_out = self.segmentation(x)

            return {
                  'classification': cls_out,
                  'localization': loc_out,
                  'segmentation': seg_out

            }
    

    