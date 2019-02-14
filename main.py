from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.utils as vutils
import torchvision.datasets as dset
from torch.utils.data import DataLoader

from dataset import ImageDataset
from generator import Generator
from discriminator import Discriminator

EPOCHS = 10
LEARNING_RATE = 0.0002
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
BATCH_SIZE = 1
Z_SIZE = 100
BETA1 = 0.5
OUTPUT_PATH = './output'
IMAGES_PATH = './images'

REAL_LABEL = 1
FAKE_LABEL = 0

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE),
        torchvision.transforms.CenterCrop(IMAGE_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

dataset = ImageDataset(IMAGES_PATH, transforms, '*.jpg')
#dataset = dset.LSUN(root='../lsun', classes=['bedroom_train'], transform=transforms)
data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(BATCH_SIZE, Z_SIZE, 1, 1, device=device)
netG = Generator(IMAGE_SIZE, IMAGE_CHANNELS, Z_SIZE).to(device)
netD = Discriminator(IMAGE_SIZE, IMAGE_CHANNELS).to(device)
criterion = nn.BCELoss()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG.apply(weights_init)
netD.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

print(device)
print(netD)
print(netG)

for epoch in range(EPOCHS):
    print('Starting epoch: %d' % (epoch))
    for i, real_data in enumerate(data_loader):
        real_data = real_data.to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), REAL_LABEL, device=device)

        output = netD(real_data)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, Z_SIZE, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(FAKE_LABEL)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(REAL_LABEL)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, EPOCHS, i, len(data_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        if i % 5 == 0:
            vutils.save_image(real_data, '%s/real_samples.png' % OUTPUT_PATH, normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (OUTPUT_PATH, epoch), normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (OUTPUT_PATH, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (OUTPUT_PATH, epoch))
