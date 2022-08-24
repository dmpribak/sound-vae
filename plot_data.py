from turtle import color
from load_data import load_data
from model import Encoder, Decoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


latent_dim = 128

ffts, names = load_data("X:/Datasets/mfdoom/mfdoom/*")
ffts = torch.unsqueeze(ffts, 1)

encoder = Encoder(latent_dim).cuda()
decoder = Decoder(latent_dim).cuda()

save = torch.load("model.pt")
encoder.load_state_dict(save["encoder_states"])
decoder.load_state_dict(save["decoder_states"])


categories = ["\\808s", "\\Claps", "\\ClosedHats", "\\Snares", "\\Kicks", "\\Crashes", "\\Vox" ]
colors = ["r", "g", "b", "y", "c", "m", "k"]

samplesx = []
samplesy = []
colormap = []
sampsx = []
sampsy = []

num_samples = 1
for i in range(ffts.size()[0]):
    sampsx.clear()
    sampsy.clear()
    mu, sigma = encoder(ffts[i:i+1])
    mu = torch.squeeze(mu)
    sigma = torch.squeeze(sigma)

    found = True
    for j in range(num_samples):
        sample = torch.normal(mu, torch.exp(sigma))
        x = mu[0].detach().cpu().numpy()
        y = mu[48].detach().cpu().numpy()
        sampsx.append(x)
        sampsy.append(y)

    for j in range(6):
        if names[i].find(categories[j]) != -1:
            samplesx += sampsx
            samplesy += sampsy
            colormap += num_samples * [colors[j]]

plt.scatter(samplesx, samplesy, c=colormap, s=2)
print(len(colormap))
print(len(samplesx))

plt.show()