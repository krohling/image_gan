import torch.nn as nn
import math
from print_layer import PrintLayer

NDF = 64

class Discriminator(nn.Module):
    def __init__(self, image_size, image_channels):
        super(Discriminator, self).__init__()

        layers = []
        in_channels = 0
        out_channels = NDF
        mid_layer_count = int(math.log(image_size/16, 2)+1)

        layers.append(nn.Conv2d(image_channels, out_channels, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        #layers.append(PrintLayer())

        for i in range(mid_layer_count):
            in_channels = out_channels
            out_channels = int(NDF * (2 ** (i+1)))

            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            #layers.append(PrintLayer())

        layers.append(nn.Conv2d(out_channels, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())
        #layers.append(PrintLayer())

        self.network = nn.Sequential(*layers)

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)