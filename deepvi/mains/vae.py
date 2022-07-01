import torch, math, copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings
from deepvi.arch import VAE
warnings.filterwarnings("ignore")
global use_cuda
use_cuda = torch.cuda.is_available()
        
def loss(x, out, mu, logvar, beta):

    diff = x - out
    latent_dim = len(logvar)

    #Compute reconstruction loss
    mse = nn.MSELoss()
    recons_loss = 0.5*(latent_dim*np.log(2*np.pi) + mse(x, out))

    #Compute KL loss
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    #Compute total loss
    loss = recons_loss + beta * kld_loss

    return recons_loss, kld_loss, loss
    
vae = VAE(100).cuda()
opt = torch.optim.Adam(vae.parameters(), lr=5e-4)
beta = 0.01
for i in range(20000):
    s = sample(128).cuda()
    mu, logvar, out = vae(s)
    rl, kl, l = loss(s, out, mu, logvar, beta)
    opt.zero_grad()
    l.backward()
    opt.step()
    if i % 1000 == 0:
        data = vae.generate(5000)
        plot_density(data.cpu().data)
