import torch, math, copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings
from deepvi.arch import Generator, Discriminator
warnings.filterwarnings("ignore")
global use_cuda
use_cuda = torch.cuda.is_available()

generator = Generator(100).cuda() #100 latent dimensions
gopt = torch.optim.Adam(generator.parameters(), lr=5e-4, betas=(0.5, 0.999))
discriminator = Discriminator().cuda()
dopt = torch.optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999))
criterion = torch.nn.BCEWithLogitsLoss()

batch_size = 128
fake_label = 0.
real_label = 1.

for i in range(50000):

    #Train the discriminator
    label = torch.full((batch_size,), real_label, dtype=torch.float).cuda()

    #Train with real batch
    discriminator.zero_grad()

    #Classify real batch, compute loss
    real_data = sample(batch_size).cuda()
    output = discriminator(real_data).view(-1)
    derror_real = criterion(output, label)
    derror_real.backward()

    #Train with fake batch
    fake = generator(batch_size)
    label.fill_(fake_label)

    # Classify fake batch, compute loss
    output = discriminator(fake.detach()).view(-1)
    derror_fake = criterion(output, label)
    derror_fake.backward()
    
    derror = derror_real + derror_fake
    dopt.step()

    #Train the generator 
    generator.zero_grad()
    label.fill_(real_label)

    #Classify fake batch
    output = discriminator(fake).view(-1)
    gerror = criterion(output, label)
    gerror.backward()

    # Update G
    gopt.step()
    
    if i % 1000 == 0:
        data = generator(5000)
        plot_density(data.cpu().data)
