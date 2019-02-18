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
from torchnet.dataset import SplitDataset, ShuffleDataset

from image_dataset import ImageDataset
from generator import Generator
from discriminator import Discriminator

EPOCHS = 10000
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
OVERFIT_THRESHOLD = 0.20
OVERFIT_RAND_RATE = 0.15

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

print('Loading dataset...')
real_dataset = ImageDataset(IMAGES_PATH, transforms, '*.*')
#real_dataset = dset.LSUN(root='../lsun', classes=['bedroom_train'], transform=transforms)
real_dataset = SplitDataset(ShuffleDataset(real_dataset), {'train': 0.8, 'validation': 0.2})
real_dataset.select('train')
real_data_loader = DataLoader(real_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)

# noise_dataset = NoiseDataset(len(real_data_loader), Z_SIZE)
# noise_data_loader = DataLoader(noise_dataset, batch_size = BATCH_SIZE)
print('Done loading dataset.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(BATCH_SIZE, Z_SIZE, 1, 1, device=device)
netG = Generator(IMAGE_SIZE, IMAGE_CHANNELS, Z_SIZE).to(device)
netD = Discriminator(IMAGE_SIZE, IMAGE_CHANNELS).to(device)
netG.init_weights()
netD.init_weights()
criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

print(device)
print(netD)
print(netG)



def calc_loss(model, inputs, label, rand_rate=None):
    batch_size = inputs.size(0)
    targets = torch.full((batch_size,), label, device=device)
    if rand_rate is not None:
        count = rand_rate * batch_size
        indices = torch.rand(count) * batch_size

        new_label = REAL_LABEL
        if label == REAL_LABEL:
            new_label = FAKE_LABEL
        
        for idx in indices:
            targets[idx] = new_label

    outputs = model(inputs)
    return criterion(outputs, targets), outputs, targets

def calc_accuracy(outputs, targets):
    accuracy_count = 0
    preds = torch.round(outputs)
    for i in range(len(preds)):
        if preds[i] == targets[i]:
            accuracy_count += 1
    
    return accuracy_count

def train(model, inputs, label, rand_rate=None):
    model.train()
    #model.zero_grad()
    with torch.set_grad_enabled(True):
        loss, _, _ = calc_loss(model, inputs, label, rand_rate)
        loss.backward()
        return loss

def validate(model, inputs, label):
    model.eval()
    with torch.set_grad_enabled(False):
        loss, outputs, targets = calc_loss(model, inputs, label)
        return calc_accuracy(outputs, targets)

for epoch in range(EPOCHS):
    print('Starting epoch: %d' % (epoch))
    errD_total = 0
    errG_total = 0
    err_count = 0
    D_val_accuracy_count = 0
    D_val_accuracy = 0.0
    D_train_accuracy_count = 0
    D_train_accuracy = 0.0

    netD.eval()
    real_dataset.select('validation')
    for _, real_data in enumerate(real_data_loader):
        #real_data = real_data[0].to(device)
        real_data = real_data.to(device)
        D_val_accuracy_count += validate(netD, real_data, REAL_LABEL)
    D_val_accuracy = D_val_accuracy_count / len(real_dataset)
    print('D_val_accuracy: %.4f - [%d/%d]' % (D_val_accuracy, D_val_accuracy_count, len(real_dataset)))

    real_dataset.select('train')
    for _, real_data in enumerate(real_data_loader):
        #real_data = real_data[0].to(device)
        real_data = real_data.to(device)
        D_train_accuracy_count += validate(netD, real_data, REAL_LABEL)
    D_train_accuracy = D_train_accuracy_count / len(real_dataset)
    print('D_train_accuracy: %.4f - [%d/%d]' % (D_train_accuracy, D_train_accuracy_count, len(real_dataset)))

    real_train_rand_rate = None
    if D_train_accuracy-D_val_accuracy > OVERFIT_THRESHOLD:
        print("Randomizing Descriminator: Overfitting")
        netD.randomize_weights()
        real_train_rand_rate = OVERFIT_RAND_RATE

    netD.zero_grad()
    netD.train()
    real_dataset.select('train')
    for i, real_data in enumerate(real_data_loader):
        #real_data = real_data[0].to(device)
        real_data = real_data.to(device)
        netD.zero_grad()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        if D_val_accuracy < D_ACC_TRAIN_THRESHOLD:
            errD_real = train(netD, real_data, REAL_LABEL, real_train_rand_rate)
        else:
            with torch.set_grad_enabled(False):
                errD_real, _, _ = calc_loss(netD, real_data, REAL_LABEL)

        # train with fake
        noise = torch.randn(len(real_data), Z_SIZE, 1, 1, device=device)
        fake = netG(noise)
        errD_fake = train(netD, fake.detach(), FAKE_LABEL)
        errD = errD_real + errD_fake
        #print('errD_real: %.4f errD_fake: %.4f errD: %.4f' % (errD_real, errD_fake, errD))
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label = torch.full((len(real_data),), REAL_LABEL, device=device)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        err_count += 1
        errD_total += errD
        errG_total += errG

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
              % (epoch, EPOCHS, i, len(real_data_loader), errD.item(), errG.item()))
    
    errD_avg = errD_total.item()/err_count
    errG_avg = errG_total.item()/err_count
    print('errD_avg: %.4f errG_avg: %.4f' % (errD_avg, errG_avg))

    if errD_avg < D_RAND_THRESHOLD and errG_avg > G_RAND_THRESHOLD:
        print("Randomizing Descriminator")
        netD.randomize_weights()

    if epoch % 100 == 0:
        vutils.save_image(real_data, '%s/real_samples.png' % OUTPUT_PATH, normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (OUTPUT_PATH, epoch), normalize=True)

        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (OUTPUT_PATH, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (OUTPUT_PATH, epoch))
