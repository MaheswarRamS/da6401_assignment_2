"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initializing the CustomDropout layer.

        Args:
            p - Dropout probability.
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].


        """
        if not self.training or self.p==0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        
            # Creating a binary mask: 1=Keep, 0=Drop
        mask = (torch.rand_like(x)>self.p).float()
            # Applying Inverted Dropout
        return x * mask / (1-self.p + 1e-7)
        
    
    
