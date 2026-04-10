
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class DecoderBlock(nn.Module):
   """
     Decoder Block with transposed convolution upsampling
   """
   def __init__(self, in_ch: int, skip_ch: int, out_ch:int, dropout_p: float =0.5) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch,in_ch, kernel_size=2, stride=2)

        # Convolutional Processing
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch+ skip_ch, out_ch, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p)
        )
   def forward(self, x:torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
       x = self.upsample(x)
       x = torch.cat([x, skip], dim=1)
       return self.conv(x)
    

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5) -> None:

        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        # VGG11Encoder
        self.encoder = VGG11Encoder(num_classes=37, in_channels=in_channels)
        self.encoder.classifier = nn.Identity()

        # Decoder block
        self.dec4 = DecoderBlock(512,512,512,dropout_p)
        self.dec3 = DecoderBlock(512,256,256,dropout_p)
        self.dec2 = DecoderBlock(256,128,128,dropout_p)
        self.dec1 = DecoderBlock(128,64,64,dropout_p)
        
        # Final upsampling to original Size
        self.final_upsample = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)

        # Output Convolution 
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder 
        _, features = self.encoder(x, return_features = True)

        # Decoder with skip connections
        x = features['block5']
        x1  = self.dec4(x, features['block4'])
        x2  = self.dec3(x1, features['block3'])
        x3  = self.dec2(x2, features['block2'])
        x4  = self.dec1(x3, features['block1'])
    
        # Final upsampling
        x = self.final_upsample(x4)

        # output
        x = self.output(x)

        return x
    