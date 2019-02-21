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
from torchnet.dataset import SplitDataset, ShuffleDataset

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
real_dataset = ImageDataset(IMAGES_PATH, transforms, REAL_LABEL, '*.*')
real_dataset = SplitDataset(ShuffleDataset(real_dataset), {'train': 0.8, 'validation': 0.2})
real_dataset.select('train')
real_data_loader = DataLoader(real_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)
generator_dataset = GeneratorDataset(netG, len(real_dataset), Z_SIZE, FAKE_LABEL, device)
generator_dataset.generate()
gen_data_loader = DataLoader(generator_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)
concat_dataset = ConcatDataset([real_dataset, generator_dataset])
concat_data_loader = DataLoader(concat_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)
#real_dataset = dset.LSUN(root='../lsun', classes=['bedroom_train'], transform=transforms)
print('Done loading dataset.')

print(device)
print(netD)
print(netG)

def calc_accuracy(outputs, targets):
    accuracy_count = 0
    preds = torch.round(outputs)
    for i in range(len(preds)):
        if preds[i] == targets[i]:
            accuracy_count += 1
    
    return accuracy_count

def train(model, inputs, targets):
    model.train()
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        return loss

def validate(model, inputs, targets):
    model.eval()
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        return calc_accuracy(outputs, targets)

for epoch in range(EPOCHS):
    print('Starting epoch: %d' % (epoch))

    ############################
    # (1) Calculate Discriminator accuracy on Real Validation dataset
    ###########################
    netD.eval()
    D_val_accuracy_count = 0
    D_val_accuracy = 0.0
    real_dataset.select('validation')
    for _, (inputs, targets) in enumerate(real_data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        D_val_accuracy_count += validate(netD, inputs, targets)
    D_val_accuracy = D_val_accuracy_count / len(real_dataset)
    # print('D_val_accuracy: %.4f - [%d/%d]' % (D_val_accuracy, D_val_accuracy_count, len(real_dataset)))

    ############################
    # (2) Calculate Discriminator accuracy on Real Training dataset
    ###########################
    netD.eval()
    D_train_accuracy_count = 0
    D_train_accuracy = 0.0
    real_dataset.select('train')
    for _, (inputs, targets) in enumerate(real_data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        D_train_accuracy_count += validate(netD, inputs, targets)
    D_train_accuracy = D_train_accuracy_count / len(real_dataset)
    # print('D_train_accuracy: %.4f - [%d/%d]' % (D_train_accuracy, D_train_accuracy_count, len(real_dataset)))

    ############################
    # (3) Train Discriminator
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
    # (4) Train Generator
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

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Val Accuracy: %.4f Train Accuracy: %.4f' % (epoch, EPOCHS, errD, errG, D_val_accuracy, D_train_accuracy))

    if errD < D_RAND_THRESHOLD and errG > G_RAND_THRESHOLD:
        print("Randomizing Descriminator")
        netD.randomize_weights()

    if epoch % 1 == 0:
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (OUTPUT_PATH, epoch), normalize=True)

        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (OUTPUT_PATH, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (OUTPUT_PATH, epoch))
