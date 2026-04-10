"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11 architecture
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_bn: bool =True) -> None:
        """Initializing the VGG11Encoder model."""
        super().__init__()
        self.use_bn = use_bn
        # Block 1: 64 filters (conv1_1)
        layers = [nn.Conv2d(in_channels, 64, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(64))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.conv1_1 = nn.Sequential(*layers) 
    
        # Block 2: 128 filters
        layers = [nn.Conv2d( 64, 128, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(128))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.conv2_1 = nn.Sequential(*layers) 
    
        #Block 3: 256 filters (2 3x3 filters)
        layers = [nn.Conv2d(128,256, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(256))
        layers.extend([nn.ReLU(inplace=True)])
        self.conv3_1 = nn.Sequential(*layers)

        layers = [nn.Conv2d(256,256, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(256))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.conv3_2 = nn.Sequential(*layers)

        # Block 4: 512 filters (2 3x3 filters)
        layers = [nn.Conv2d(256,512, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(512))
        layers.extend([nn.ReLU(inplace=True)])
        self.conv4_1 = nn.Sequential(*layers)
        
        layers = [nn.Conv2d(512,512, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(512))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.conv4_2 = nn.Sequential(*layers)

        # Block 5: 512 filters same as block 4

        layers = [nn.Conv2d(512,512, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(512))
        layers.extend([nn.ReLU(inplace=True)])
        self.conv5_1 = nn.Sequential(*layers)

        layers = [nn.Conv2d(512,512, kernel_size = 3, padding = 1)]
        if use_bn:
            layers.append( nn.BatchNorm2d(512))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])
        self.conv5_2 = nn.Sequential(*layers)
        
        # Classifier Head: 2 x Fc layers(4096) and one FC(1000)

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes)
        )
    
    def forward(
        self, x: torch.Tensor, return_features: bool = False
        ):

        """+

        Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
       
        """
        features: Dict[str, torch.Tensor] = {}
        x1 = self.conv1_1(x)
        features['block1'] = x1 
        x2 = self.conv2_1(x1)
        features['block2'] = x2
        x3 = self.conv3_1(x2)
        x4 = self.conv3_2(x3)
        features['block3'] = x4 
        x5 = self.conv4_1(x4)
        x6 = self.conv4_2(x5)
        features['block4'] = x6 
        x7 = self.conv5_1(x6)
        x8 = self.conv5_2(x7)
        features['block5'] = x8
        
        # Flatten the output --> input to FC Layer
        x9 = torch.flatten(x8, 1)
        
        # Classifier Head output
        logits = self.classifier(x9)
        
        if return_features:
            # return both feature and logits
            return logits, features
        else:
            # Returns only logits
            return logits
        
      