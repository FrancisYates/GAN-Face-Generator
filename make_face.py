import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # Size: B x (ngf*2) x 96 x 80
            #nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            # Size: B x (ngf*2) x 192 x 160
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


def Generate_faces():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    netCG.eval()
    lbl = np.loadtxt('data/train/' +'labels.csv',delimiter=',',usecols=(1,2,3))
    #lblV = np.loadtxt('validate/' +'labels.csv',delimiter=',',usecols=(1,2,3))
    #lbl = np.concatenate((lblT,lblV))
    lightDirections = np.unique(lbl,axis = 0)
    rndIndex = np.random.randint(0,63,4)

    #f, axarr = plt.subplots(4,4)
    #f.
    cols=4
    rows = 4 
    figure = plt.figure(figsize=(10, 8))

    for i in range(0,4,1):
    #rndIndex = np.random.randint(0,63,4)
        ranLabels = [lightDirections[rndIndex[i]]]*4
        ranLabels = np.stack(ranLabels, axis = 0)
        labels = torch.from_numpy(ranLabels).to(device).float()
        z = torch.randn(4,nz,1,1,device=device)
        images = netCG(z,labels)
        for j in range(0,4,1):
            figure.add_subplot(rows, cols, ((i*4)+j)+1)
            plt.axis("off")
            plt.imshow(images[i,:].cpu().detach().squeeze(), cmap="gray")

    
    directory = 'img.png'
    plt.savefig(directory, bbox_inches='tight')


def Interpolate_light():
    device = torch.device("cpu")
    netCG.eval()
    lbl = np.loadtxt('data/train/' +'labels.csv',delimiter=',',usecols=(1,2,3))
    lightDirections = np.unique(lbl,axis = 0)
    rndIndex = np.random.randint(0,63,1)

    label = [lightDirections[rndIndex[0]]]*7
    label = np.stack(label, axis = 0)
    labels = torch.from_numpy(label).to(device).float()

    # Solution

    nsamples = 7
    z1 = torch.randn(1, nz, 1, 1, device=device)
    z2 = torch.randn(1, nz, 1, 1, device=device)
    z = torch.zeros(nsamples,nz,1,1,device=device)
    for i in range(nsamples):
        w1 = i/(nsamples-1)
        w2 = 1-w1
        z[i,:,:,:] = w1*z1 + w2*z2
        images = netCG(z,labels)

    figure = plt.figure(figsize=(12, 4))
    for i in range(nsamples):
        figure.add_subplot(1, nsamples, i+1)
        plt.axis("off")
        plt.imshow(images[i,:].squeeze().cpu().detach(), cmap="gray")
    plt.show()


def Interpolate_faces():
    device = torch.device("cpu")
    netCG.eval()
    lbl = np.loadtxt('data/train/' +'labels.csv',delimiter=',',usecols=(1,2,3))
    lightDirections = np.unique(lbl,axis = 0)

    rndIndex = np.random.randint(0,63,2)

    l1 = [lightDirections[rndIndex[0]]]
    l1 = torch.from_numpy(np.reshape(l1, (-1, 1))).to(device).float()
    l2 = [lightDirections[rndIndex[1]]]
    l2 = torch.from_numpy(np.reshape(l2, (-1, 1))).to(device).float()

    nsamples = 7
    l = []
    z = torch.randn(1,nz,1,1,device=device)
    for i in range(nsamples):
        w1 = i/(nsamples-1)
        w2 = 1-w1
        x = w1*l1
        l.append(w1*l1 + w2*l2)
        if i == 0 :
            fixedZ = z
        else:
            fixedZ = torch.cat((fixedZ,z))

    l = np.stack(l, axis = 0)
    l = l.squeeze()
    labels = torch.from_numpy(l).to(device).float()

    images = netCG(fixedZ,labels)

    print('Interpolation between 2 light source directions with fixed z value')
    figure = plt.figure(figsize=(12, 4))
    for i in range(nsamples):
        figure.add_subplot(1, nsamples, i+1)
        plt.axis("off")
        plt.imshow(images[i,:].squeeze().cpu().detach(), cmap="gray")
    plt.show()

    print('Light source direction 1: ' + str(labels[0]))
    print('Light source direction 2: ' + str(labels[nsamples-1]))
    



if __name__ == '__main__':
    nz = 50  # Size of z latent vector (i.e. size of generator input)
    ngf = 64  # Size of feature maps in generator
    ndf = 48  # Size of feature maps in discriminator

    #Load weights from our trained model
    device = torch.device("cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netCG = ConditionalGenerator()
    netCD = ConditionalDiscriminator()

    netCG.load_state_dict(torch.load('CG.pkl',map_location=torch.device(device)))
    netCD.load_state_dict(torch.load('CD.pkl',map_location=torch.device(device)))

    Generate_faces()
    Interpolate_light()