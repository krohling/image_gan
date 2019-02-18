import torch
import torch.nn as nn
import math
from print_layer import PrintLayer

NDF = 64
RAND_SCALE = 0.01

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
    
    def init_weights(self):
        self.apply(self.__init_weights)
    
    def __init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def randomize_weights(self):
        self.apply(self.__randomize_weights)

    def __randomize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            d_weights = self.__rand_tensor(m.weight.data)
            m.weight.data.add_(d_weights)
        elif classname.find('BatchNorm') != -1:
            d_weights = self.__rand_tensor(m.weight.data)
            m.weight.data.add_(d_weights)
            d_bias = self.__rand_tensor(m.weight.data)
            m.bias.data.add_(d_bias)
    
    def __rand_tensor(self, t_like):
        r_tensor = torch.rand_like(t_like)
        r_tensor.sub_(0.5)
        r_tensor.mul_(RAND_SCALE)
        return r_tensor
