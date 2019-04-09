import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel= (3, 3), padding= 0):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(SingleConv(in_chans, out_chans, kernel, padding), SingleConv(out_chans, out_chans, kernel, padding))

    def forward(self, inputs):
        y = self.layers(inputs)
        return y

class SingleConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel, padding, activation= 'relu'):
        super(SingleConv, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        if activation == 'relu':
            self.layer = nn.Sequential(nn.Conv2d(in_channels= in_chans, out_channels= out_chans, kernel_size= kernel, padding= padding), 
                nn.BatchNorm2d(out_chans), nn.ReLU(inplace= True))
        elif activation == 'sigmoid':
            self.layer = nn.Sequential(nn.Conv2d(in_channels= in_chans, out_channels= out_chans, kernel_size= kernel, padding= padding), 
                nn.BatchNorm2d(out_chans), nn.Sigmoid())
        elif activation == 'none':
            self.layer = nn.Sequential(nn.Conv2d(in_channels= in_chans, out_channels= out_chans, kernel_size= kernel, padding= padding), 
                nn.BatchNorm2d(out_chans))

    def forward(self, inputs):
        y = self.layer(inputs)
        return y

class MaxPool(nn.Module):
    def __init__(self, kernel=(2, 2), stride= 2):
        super(MaxPool, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size= kernel, stride= stride)

    def forward(self, inputs):
        y = self.layer(inputs)
        return y

class TransposeConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel= (2, 2), stride= 2):
        super(TransposeConv, self).__init__()
        self.layers = nn.Sequential(nn.ConvTranspose2d(in_channels= in_chans, out_channels= out_chans, 
                kernel_size= kernel, stride= stride), nn.BatchNorm2d(out_chans))

    def forward(self, inputs):
        y = self.layers(inputs)
        return y

class PreActivConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel, padding):
        super(PreActivConv, self).__init__()
        self.layers = nn.Sequential(nn.BatchNorm2d(in_chans), nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= in_chans, out_channels= out_chans, kernel_size= kernel, padding= padding))

    def forward(self, inputs):
        y = self.layers(inputs)
        return y

class ResidualBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel= (3, 3), padding= 1, activation= 'relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = SingleConv(in_chans, out_chans, kernel= kernel, padding= padding, activation= activation)
        self.conv2 = SingleConv(out_chans, out_chans, kernel= kernel, padding= padding, activation= 'none')
        self.relu = nn.ReLU(inplace= True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        x = torch.cat((x, inputs), 1)
        x = self.relu(x)

        return x

class FullPreResBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel= (3, 3), padding= 1):
        super(FullPreResBlock, self).__init__()
        self.conv1 = PreActivConv(in_chans, out_chans, kernel= kernel, padding= padding)
        self.conv2 = PreActivConv(out_chans, out_chans, kernel= kernel, padding= padding)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        x = torch.cat((x, inputs), 1)

        return x



