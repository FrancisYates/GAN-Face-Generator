import datetime
from time import strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import  TensorDataset
import torch.nn.functional as F
from PIL import Image
import os
from torch import Tensor
import matplotlib.pyplot as plt
import time
import random


def createDatasetDGAN(transform):
    # load image paths from csvs
    train = 'data/train_ex/'
    validate = 'data/validate_ex/'
    imgNamesT = np.loadtxt(train + 'labels.csv', delimiter=',', dtype=str, usecols=(0))
    dir = [train] * len(imgNamesT)
    imgNamesT = np.core.defchararray.add(dir, imgNamesT)
    imgNamesV = np.loadtxt(validate + 'labels.csv', delimiter=',', dtype=str, usecols=(0))
    dir = [validate] * len(imgNamesV)
    imgNamesV = np.core.defchararray.add(dir, imgNamesV)

    imgNames = np.concatenate((imgNamesT, imgNamesV))

    # load light vectors from csv
    lblT = np.loadtxt(train + 'labels.csv', delimiter=',', usecols=(1, 2, 3))
    lblV = np.loadtxt(validate + 'labels.csv', delimiter=',', usecols=(1, 2, 3))
    lbl = np.concatenate((lblT, lblV))

    # prevent creating partial batchs
    numBatches = len(imgNames) // batch_size
    print("creating " + str(numBatches) + " batches")

    # create Tensor with all images
    for i in range(0, numBatches, 1):
        start_time = time.time()
        print("batch " + str(i + 1) + "/" + str(numBatches))
        temp = transform(Image.open(imgNames[i*batch_size]))

        for j in range(1, batch_size, 1):
            img = transform(Image.open(imgNames[i * batch_size+ j]))
            temp = torch.cat((temp, img), 0)

        if(i == 0):
            master_temp = temp
        else:
            master_temp = torch.cat((master_temp, temp), 0)

        delta_time = time.time() - start_time
        print("time taken " + str(delta_time))

    # prevent creating partial batchs
    lbl = lbl[:numBatches * batch_size]
    print(master_temp.size(0))
    print(Tensor(lbl).size(0))
    return TensorDataset(Tensor(master_temp).unsqueeze(1), Tensor(lbl))


class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(3,6),
            nn.Linear(6,20),
            )
        self.mixer = nn.Sequential(
            nn.Linear(20+nz,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.main = nn.Sequential(
            # input is Z of size B x nz x 1 x 1, we put this directly into a transposed convolution
            nn.ConvTranspose2d( 50, ngf*4, (12,10), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size: B x (ngf*4) x 12 x 10
            nn.ConvTranspose2d( ngf*4, ngf*2, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Size: B x (ngf*4) x 24 x 20
            nn.ConvTranspose2d(ngf*2, ngf, 2, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size: B x (ngf*2) x 48 x 40
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            # Size: B x (ngf*2) x 96 x 80
            nn.Tanh()
        )

    def forward(self, z, label):
        #classembedding = self.MLP(label) # Size B x 50
        #mixedembedding = self.mixer(torch.cat((z.squeeze(),classembedding),1))
        #conditional_input = mixedembedding.unsqueeze(-1).unsqueeze(-1) # Size B x 128 x 1 x 1

        #To make GAN a CGAN replace unconditional_input below with conditional_input above
        unconditional_input = z.squeeze()
        input = unconditional_input.unsqueeze(-1).unsqueeze(-1) # Size B x 128 x 1 x 1
        x = self.main(input)
        img = F.interpolate(x, (192, 160))
        return img


class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(3,10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10,192*160)
        )
        self.main = nn.Sequential(
            # state size. 1 x 192 x 160
            nn.Conv2d(2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 96 x 80
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 48 x 40
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 24 x 20
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 5
            nn.Conv2d(ndf*8, 1, (12,10), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, label):
        classembedding = self.MLP(label) # Size B x 32^2
        classembedding = classembedding.view(classembedding.shape[0],1,192,160)
        input = torch.cat((image,classembedding),1) # Size B x 2 x 32 x 32
        return self.main(input)


def getRandLabels(numLabels):
  lblT = np.loadtxt('data/train/' +'labels.csv',delimiter=',',usecols=(1,2,3))
  lblV = np.loadtxt('data/validate/' +'labels.csv',delimiter=',',usecols=(1,2,3))
  lbl = np.concatenate((lblT,lblV))
  lightDirections = np.unique(lbl,axis = 0)
  ranLabels = []
  rndIndex = np.random.randint(0,63,numLabels)
  for index in rndIndex:
    ranLabels.append(lightDirections[index])

  ranLabels = np.stack(ranLabels, axis = 0)
  returnLabels = torch.from_numpy(ranLabels).float()
  return returnLabels


def prnt(file_name):
    netCG.eval()
    figure = plt.figure(figsize=(16, 16), dpi=80)
    cols, rows = 3, 3

    count = 0
    for i in range(cols):
        z = torch.randn(4, nz, 1, 1, device=device)
        labels = getRandLabels(4).to(device)
        images = netCG(z, labels)
        for i in range(rows):
            figure.add_subplot(10, 8, count * 8 + i + 1)
            plt.axis("off")
            plt.imshow(images[i, :].cpu().detach().squeeze(), cmap="gray")
        count += 1
        
    directory = file_name + '.png'
    plt.savefig(directory, bbox_inches='tight')

    img = Image.open(directory)
    size = [img.size[0] * 8, img.size[1] * 8]
    img.resize(size)
    img.save(directory)
    plt.close(figure)
    plt.clf()
    netCG.train()


def train(folder):
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerCD = optim.Adam(netCD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerCG = optim.Adam(netCG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (images,labels) in enumerate(trainDataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netCD.zero_grad()
            # Format batch
            real_images, labels = images.to(device), labels.to(device)

            #label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            label = np.random.default_rng().uniform(0.9, 1.1, batch_size)
            label = torch.from_numpy(label)
            label = label.float()
            label = label.to(device)

            # Forward pass real batch through D
            output = netCD(real_images,labels).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            z = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate random labels
            random_labels = getRandLabels(batch_size).to(device)
            #noisyLabel =  (0.05*torch.randn(1))
            noisy_label = random.gauss(0, 0.1)
            noisy_label = max(0, noisy_label)
            #noisyLabel = noisyLabel.to(device)
            # Generate fake image batch with G
            fake = netCG(z,random_labels)
            #label.fill_(fake_label)
            label = np.random.default_rng().uniform(low=0.0, high=0.1, size=batch_size)
            label = torch.from_numpy(label)
            label = label.float()
            label = label.to(device)

            # Classify all fake batch with D
            output = netCD(fake.detach(),random_labels).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerCD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netCG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netCD(fake,labels).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerCG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, num_epochs, i+1, len(trainDataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                x = folder + '%d-%d--%d-%d' % (epoch+1, num_epochs, i+1, len(trainDataloader))
                prnt(x)
                torch.save(netCG.state_dict(), x + 'CG.pkl')
                torch.save(netCD.state_dict(), x +'CD.pkl')


if __name__ == '__main__':
    folder = 'GeneratedImages\\' + strftime("%m-%d-%H-%M") + '\\'
    os.mkdir(folder)
    batch_size = 64
    nz = 50  # Size of z latent vector (i.e. size of generator input)
    ngf = 64  # Size of feature maps in generator
    ndf = 48  # Size of feature maps in discriminator
    num_epochs = 25  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Grayscale(),
                                    transforms.Normalize(0.5, 0.5),
                                    transforms.Resize((192, 160))])

    trainData = createDatasetDGAN(transform)
    trainDataloader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True, num_workers=1)

    netCG = ConditionalGenerator()
    netCG = netCG.to(device)

    netCD = ConditionalDiscriminator()
    netCD = netCD.to(device)

    train(folder)