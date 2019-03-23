from unet_sections import *

class UNet(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_chans, 64)
        self.pool1 = MaxPool()
        self.down2 = DoubleConv(64, 128)
        self.pool2 = MaxPool()
        self.down3 = DoubleConv(128, 256)
        self.pool3 = MaxPool()
        self.down4 = DoubleConv(256, 512)
        self.pool4 = MaxPool()
        self.down5 = DoubleConv(512, 1024)
        self.up1 = TransposeConv(1024, 512)
        self.down6 = DoubleConv(1024, 512)
        self.up2 = TransposeConv(512, 256)
        self.down7 = DoubleConv(512, 256)
        self.up3 = TransposeConv(256, 128)
        self.down8 = DoubleConv(256, 128)
        self.up4 = TransposeConv(128, 64)
        self.down9 = DoubleConv(128, 64)
        self.out = SingleConv(64, out_chans, kernel= (1, 1))

        self.down_layers = [self.down1, self.pool1, 
        self.down2, self.pool2, 
        self.down3, self.pool3, 
        self.down4, self.pool4, 
        self.down5]

        self.up_layers = [self.up1, 
        self.down6, self.up2, 
        self.down7, self.up3, 
        self.down8, self.up4, 
        self.down9, self.out]

    def forward(self, x):
        filters = []

        for i, down_layer in enumerate(self.down_layers):
            if isinstance(down_layer, DoubleConv) and i != len(self.down_layers) - 1:
                d_filters = down_layer(x)
                filters.append(d_filters)
                x = d_filters.clone()
            else:
                x = down_layer(x)

        i = len(filters) - 1
        for up_layer in self.up_layers:
            if isinstance(up_layer, DoubleConv):
                d_filters = filters[i]
                x_size = x.size()
                h, w = x_size[2], x_size[3]
                d_filters = d_filters[:, :, 0:h, 0:w]
                cat = torch.cat((x, d_filters), 1)
                x = up_layer(cat)
                i -= 1
            else:
                x = up_layer(x)
        return x





