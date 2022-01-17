import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv,self).__init__()
        self.conv = double_conv(in_channels,out_channels)

    def forward(self,x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels,out_channels)
        )

    def forward(self,x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.conv = double_conv(in_channels,out_channels)

    def forward(self, x1,x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, (diff_x//2, diff_x - diff_x//2, diff_y//2, diff_y - diff_y//2))

        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet_model(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Unet_model, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128,256)
        self.down3 = down(256,512)
        self.down4 = down(512,512)
        self.up3 = up(1024,256)
        self.up2 = up(512,128)
        self.up1 = up(256,64)
        self.up0 = up(128,64)
        self.outc = outconv(64,n_classes)

    def forward(self,x1):
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.up3(x5,x4)
        x5 = self.up2(x5,x3)
        x5 = self.up1(x5,x2)
        x5 = self.up0(x5,x1)
        x5 = self.outc(x5)
        
        return torch.sigmoid(x5)

if __name__ == "__main__":
    
    dummy_img = torch.rand(4, 3, 256, 256)
    net = Unet_model(6)
    # print(net)
    outputs = net(dummy_img)
    print(outputs[0].shape)