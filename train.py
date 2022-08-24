from joblib import PrintTime
from load_data import load_data
from model import Encoder, Decoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np



latent_dim = 128
mse = nn.MSELoss()

def loss(x, x_pred, mu, sigma):
    recon_error = mse(x_pred, x)
    rel_entropy = torch.mean(.5*(torch.sum(torch.exp(sigma) + mu**2 - sigma - 1, dim=1)))
    return recon_error + .000001 * rel_entropy

losses = []
val_losses = []
Load = True

def train():
    if Load:
        save = torch.load("model.pt")
        encoder.load_state_dict(save["encoder_states"])
        decoder.load_state_dict(save["decoder_states"])

    adam = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=.0001)
    if Load:
        adam.load_state_dict(save["optimizer_states"])
    batch_size = 20
    for epoch in range(0,20):
        for chunk in range(0,1):
            #ffts = load_data(2000,chunk)

            for batch in range(0, 2000, batch_size):
                
                adam.zero_grad()
                #epsilon = torch.normal(0, 1, size=(batch_size,latent_dim)).cuda()
                
                mu, sigma = encoder(ffts[batch:(batch+batch_size)])
                epsilon = torch.randn_like(sigma, requires_grad=True).cuda()

                z = mu + torch.exp(.5*sigma) * epsilon
                pred = decoder(z)

                los = loss(ffts[batch:(batch+batch_size)], pred, mu, sigma)
                los.backward()
                losses.append(los.item())
                adam.step()
                
                
                epsilon_val = torch.normal(0, 1, size=(83,latent_dim)).cuda()
                mu_val, sigma_val = encoder(val[:])
                z_val = mu_val + torch.exp(sigma_val) * epsilon_val
                pred_val = decoder(z_val)
                val_los = loss(val[:], pred_val, mu_val, sigma_val)
                val_losses.append(val_los.item())
                

                
            print(losses[-1])
            
        if epoch % 10 == 0:
            torch.save({
                "encoder_states": encoder.state_dict(),
                "decoder_states": decoder.state_dict(),
                "optimizer_states": adam.state_dict(),
                },
                "model.pt")
            print("Saved!")
    torch.save({
        "encoder_states": encoder.state_dict(),
        "decoder_states": decoder.state_dict(),
        "optimizer_states": adam.state_dict(),
        },
        "model.pt")



ffts, names = load_data("X:/Datasets/mfdoom/mfdoom/*", randomize=False, refreshCsv=False)
#ffts = ffts
#ffts = ffts/torch.max(ffts)

ffts = torch.unsqueeze(ffts, 1)

val = ffts[0:83]
ffts = ffts[83:]


encoder = Encoder(latent_dim).cuda()
decoder = Decoder(latent_dim).cuda()



train()

plt.plot(losses, color="r")
plt.plot(val_losses, color="b")
plt.show()

mu, sigma = encoder(ffts[0:1])
print(mu)
print(torch.exp(sigma))

epsilon = torch.normal(0, 1, size=(1,latent_dim)).cuda()
z = mu + torch.exp(sigma) * epsilon

pred = torch.squeeze(decoder(z))

#plt.plot(pred.detach().cpu().numpy())
plt.subplot(1, 2, 1)
plt.pcolormesh(torch.squeeze(ffts[0]).detach().cpu().numpy().T, vmin=0, vmax=1, shading='gouraud')
plt.title("Truth")
plt.subplot(1, 2, 2)
prediction = pred.detach().cpu().numpy().T
plt.pcolormesh(prediction, vmin=0, vmax=1, shading='gouraud')
plt.title("Reconstruction")
#plt.plot(torch.squeeze(ffts[0]).detach().cpu().numpy())
plt.show()

