from turtle import color
from load_data import load_data
from model import Encoder, Decoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from scipy.signal import stft, istft
from scipy.io import wavfile
import numpy as np

latent_dim = 128

ffts, names = load_data("X:/Datasets/mfdoom/mfdoom/*")
ffts = torch.unsqueeze(ffts, 1)

encoder = Encoder(latent_dim).cuda()
decoder = Decoder(latent_dim).cuda()

save = torch.load("model.pt")
encoder.load_state_dict(save["encoder_states"])
decoder.load_state_dict(save["decoder_states"])


mu, sigma = encoder(ffts[0:1])

epsilon = torch.normal(0, 1, size=(1,latent_dim)).cuda()
z = mu + torch.exp(sigma) * epsilon

pred = torch.squeeze(decoder(z))

prediction = torch.squeeze(pred).detach().cpu().numpy().T
plt.subplot(1, 2, 1)
plt.pcolormesh(torch.squeeze(ffts[0]).detach().cpu().numpy().T, vmin=0, vmax=1, shading='gouraud')
plt.title("Truth")

plt.subplot(1, 2, 2)
plt.pcolormesh(prediction, vmin=0, vmax=1, shading='gouraud')
plt.title("Reconstruction")
plt.show()


SAMPLERATE, sound = wavfile.read("X:/Datasets/mfdoom/mfdoom/808s/A#_07_Bass_01_SP.wav")
sound = np.array(sound.T[0])
wavfile.write("original.wav", 44100, sound)
freq, times, Zxx = stft(sound, fs=44100, nperseg=256)

Zxx = np.angle(Zxx)
Zxx = Zxx.T
Zxx = Zxx[:1024]

Zxx = np.concatenate((Zxx, np.zeros((1024-Zxx.shape[0], 129))), axis=0)
Zxx = Zxx.T

prediction = np.concatenate((prediction, np.zeros((1,1024))))

prediction = prediction*np.exp(1j*Zxx)

t, x = istft(prediction, fs=44100, nperseg=256)
x *= 10000

plt.subplot(1, 2, 1)
plt.plot(sound)
plt.title("Original")

plt.subplot(1,2,2)
plt.plot(t,x)
plt.title("reconstruction")
plt.show()

x = np.rint(x).astype(np.int16)

print(x)

wavfile.write("reconstruction.wav", 44100, x)