import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel= (3, 3)):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(SingleConv(in_chans, out_chans, kernel), SingleConv(out_chans, out_chans, kernel))

    def forward(self, inputs):
        y = self.layers(inputs)
        return y

class SingleConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel, activation= 'relu'):
        super(SingleConv, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        if activation == 'relu':
            self.layer = nn.Sequential(nn.Conv2d(in_channels= in_chans, out_channels= out_chans, kernel_size= kernel), 
                nn.BatchNorm2d(out_chans), nn.ReLU(inplace= True))
        if activation == 'sigmoid':
            self.layer = nn.Sequential(nn.Conv2d(in_channels= in_chans, out_channels= out_chans, kernel_size= kernel), 
                nn.BatchNorm2d(out_chans), nn.Sigmoid())

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
                kernel_size= kernel, stride= stride), nn.BatchNorm2d(out_chans), nn.ReLU(inplace= True))

    def forward(self, inputs):
        y = self.layers(inputs)
        return y