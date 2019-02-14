import torch.nn as nn
import math
from print_layer import PrintLayer

NGF = 128

class Generator(nn.Module):
    def __init__(self, image_size, image_channels, z_size):
        super(Generator, self).__init__()
        
        layers = []
        in_channels = 0
        out_channels = int(NGF * image_size/8)
        mid_layer_count = int(math.log(image_size/16, 2)+1)

        layers.append(nn.ConvTranspose2d(z_size, out_channels, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))
        #layers.append(PrintLayer())

        for i in range(mid_layer_count):
            in_channels = out_channels
            out_channels = int(NGF * image_size/(8 * (2 ** (i+1))))

            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            #layers.append(PrintLayer())

        layers.append(nn.ConvTranspose2d(NGF, image_channels, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        #layers.append(PrintLayer())

        self.network = nn.Sequential(*layers)

    def forward(self, input):
        return self.network(input)