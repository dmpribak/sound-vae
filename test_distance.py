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

def dist(a,b):
    mu1, sigma1 = encoder(ffts[a:a+1])
    mu2, sigma2 = encoder(ffts[b:b+1])
    mu1 = torch.squeeze(mu1)
    mu2 = torch.squeeze(mu2)

    return(torch.linalg.norm(mu1 - mu2))

print(dist(0,5), names[0], names[5])
print(dist(0,200), names[0], names[120])

dist_dict = {}
for i in range(len(names)):
    d = dist(0, i)
    dist_dict[i] = d.detach().cpu().numpy()

sorted = dict(sorted(dist_dict.items(), key=lambda item: item[1]))

for i in range(100):
    item = list(sorted.items())[i]
    print(item, names[item[0]])


"""
for i in range(16):
    plt.subplot(4, 4, i+1)
    index = list(sorted.items())[i][0]
    plt.plot(torch.squeeze(ffts[index]).detach().cpu().numpy())
    plt.title(names[index])

plt.show()

print("DIFFERENT")

for i in range(16):
    plt.subplot(4, 4, i+1)
    index = list(sorted.items())[-i][0]
    plt.plot(torch.squeeze(ffts[index]).detach().cpu().numpy())
    plt.title(names[index])
"""
plt.show()