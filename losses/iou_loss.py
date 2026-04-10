"""Custom IoU loss 
"""
from typing import Literal
import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: Literal ['none', 'mean','sum']= 'mean') -> None:
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps

        # Validate Reduction parameter
        if reduction not in ['none', 'mean','sum']:
            raise ValueError(f'reduction must be "none", "mean","sum", got{reduction}')
        self.reduction = reduction

    def _to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert [cx,cy,w,h] to [x1,y1,x2,y2]
        
        Args:
        boxes(torch.Tensor):Boxes [B,4]
        
        Returns:
               torch.Tensor: Corner format [B,4]
        """
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = cx -w/2
        y1 = cy -h/2
        x2 = cx + w/2
        y2 = cy + h/2

        return torch.stack([x1,y1,x2,y2], dim=1)

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        
        # converting  to corner format
        pred_c = self._to_corners(pred_boxes)
        target_c = self._to_corners(target_boxes)

        # Calculating the intersection coordinates
        inter_x1 = torch.max(pred_c[:,0], target_c[:,0])
        inter_y1 = torch.max(pred_c[:,1], target_c[:,1])
        inter_x2 = torch.min(pred_c[:,2], target_c[:,2])
        inter_y2 = torch.min(pred_c[:,3], target_c[:,3])

        # Calcualte the intersection area
        inter_w = torch.clamp(inter_x2-inter_x1, min=0)
        inter_h = torch.clamp(inter_y2-inter_y1, min=0)
        intersection = inter_w * inter_h

        # Calculate Union area
        pred_area = (pred_c[:,2] - pred_c[:,0]) * (pred_c[:,3] - pred_c[:,1])
        target_area = (target_c[:,2] - target_c[:,0]) * (target_c[:,3] - target_c[:,1])
        union = pred_area + target_area - intersection

        # Calculate IOU
        iou = intersection / (union +self.eps)

        loss = 1 - iou

        # Applying Reduction

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

