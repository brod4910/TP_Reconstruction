from .unet_sections import *

import torch

class GeneralUnet(nn.Module):
    def __init__(self, down_layers, up_layers):
        super(GeneralUnet, self).__init__()
        self.up_layers = up_layers
        self.down_layers = down_layers
        self.params = torch.nn.ModuleList([*down_layers, *up_layers])

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

