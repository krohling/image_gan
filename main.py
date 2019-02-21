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
from torch.utils.data import DataLoader, ConcatDataset
from torchnet.dataset import SplitDataset

from image_dataset import ImageDataset
from generator_dataset import GeneratorDataset
from generator import Generator
from discriminator import Discriminator

EPOCHS = 100000
LEARNING_RATE = 0.0002
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
Z_SIZE = 100
BETA1 = 0.5
OUTPUT_PATH = './output'
IMAGES_PATH = './images'

G_RAND_THRESHOLD = 20.0
D_RAND_THRESHOLD = 0.01
D_ACC_TRAIN_THRESHOLD = 0.90
OVERFIT_THRESHOLD = 0.15
OVERFIT_RAND_RATE = 0.20

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(BATCH_SIZE, Z_SIZE, 1, 1, device=device)
netG = Generator(IMAGE_SIZE, IMAGE_CHANNELS, Z_SIZE).to(device)
netD = Discriminator(IMAGE_SIZE, IMAGE_CHANNELS).to(device)
netG.init_weights()
netD.init_weights()
criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

print('Loading dataset...')
real_dataset = ImageDataset(IMAGES_PATH, transforms, REAL_LABEL, device, '*.*')
real_dataset = SplitDataset(real_dataset, {'train': 0.8, 'validation': 0.2})
real_dataset.select('train')
real_data_loader = DataLoader(real_dataset, shuffle=True, batch_size=BATCH_SIZE)
generator_dataset = GeneratorDataset(netG, len(real_dataset), Z_SIZE, FAKE_LABEL, device)
generator_dataset.generate()
gen_data_loader = DataLoader(generator_dataset, shuffle=True, batch_size=BATCH_SIZE)
concat_dataset = ConcatDataset([real_dataset, generator_dataset])
concat_data_loader = DataLoader(concat_dataset, shuffle=True, batch_size=BATCH_SIZE)
#real_dataset = dset.LSUN(root='../lsun', classes=['bedroom_train'], transform=transforms)
print('Done loading dataset.')

print(device)
print(netD)
print(netG)

def train(model, inputs, targets):
    model.train()
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        return loss


def calc_accuracy(outputs, targets):
    accuracy_count = 0
    preds = torch.round(outputs)
    for i in range(len(preds)):
        if preds[i] == targets[i]:
            accuracy_count += 1
    
    return accuracy_count

def validate(model, data_loader):
    model.eval()
    input_count = 0
    accuracy_count = 0
    
    for _, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            accuracy_count += calc_accuracy(outputs, targets)
            input_count += len(inputs)

    return accuracy_count / input_count



for epoch in range(EPOCHS):
    print('Starting epoch: %d' % (epoch))

    ############################
    # (1) Train Discriminator
    ###########################
    netD.zero_grad()
    netD.train()
    real_dataset.select('train')
    generator_dataset.generate()
    errD = 0
    for _, (inputs, targets) in enumerate(concat_data_loader):
        inputs = inputs.to(device)
        targets = torch.squeeze(targets).to(device)
        errD += train(netD, inputs, targets)
        optimizerD.step()


    ############################
    # (2) Train Generator
    ###########################
    netG.zero_grad()
    netG.train()
    errG = 0
    errG_total = 0
    for _, (fake, _) in enumerate(gen_data_loader):
        outputs = netD(fake)
        targets = torch.full((len(fake),), REAL_LABEL, device=device)
        errG = criterion(outputs, targets)
        errG.backward()
        optimizerG.step()
        errG_total += errG


    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, EPOCHS, errD, errG))

    if errD < D_RAND_THRESHOLD and errG > G_RAND_THRESHOLD:
        print("Randomizing Descriminator")
        netD.randomize_weights()

    if epoch % 100 == 0:
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (OUTPUT_PATH, epoch), normalize=True)
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (OUTPUT_PATH, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (OUTPUT_PATH, epoch))

        real_dataset.select('validation')
        D_val_accuracy = validate(netD, real_data_loader)

        real_dataset.select('train')
        D_train_accuracy = validate(netD, real_data_loader)

        D_fake_accuracy = validate(netD, gen_data_loader)

        print('***********************')
        print('Val Accuracy: %.4f Train Accuracy: %.4f Fake Accuracy: %.4f' % (D_val_accuracy, D_train_accuracy, D_fake_accuracy))
        print('***********************')
