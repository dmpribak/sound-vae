import torch.nn as nn
from torch.nn.modules import conv
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        hidden_dims = [2048, 1024, 512, 256, 128]

        self.conv1 = nn.Conv1d(1, 4, 3, 2, 1)
        self.conv2 = nn.Conv1d(4, 8, 3, 2, 1)
        self.conv3 = nn.Conv1d(8, 16, 3, 2, 1)
        self.conv4 = nn.Conv1d(16, 32, 3, 2, 1)
        self.conv5 = nn.Conv1d(32, 64, 3, 2, 1)
            
        self.linear_mu = nn.Linear(64*64, latent_dim)
        self.linear_sigma = nn.Linear(hidden_dims[-1], latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        conv1 = self.conv1(X)
        conv1 = self.relu(conv1)
        
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        
        conv4 = self.conv4(conv3)
        conv4 = self.relu(conv4) #32x128
        
        conv5 = self.conv5(conv4)
        conv5 = self.relu(conv5) #64x64
        
        flat = torch.flatten(conv5, 1)
        
        mu = self.linear_mu(flat)
        sigma = self.linear_mu(flat)
        
        return mu, sigma
        
        
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        hidden_dims = [128, 256, 512, 1028, 2048]
        
        self.conv1 = nn.ConvTranspose1d(64, 32, 3, 2, 1, 1)
        self.conv2 = nn.ConvTranspose1d(32, 16, 3, 2, 1, 1)
        self.conv3 = nn.ConvTranspose1d(16, 8, 3, 2, 1, 1)
        self.conv4 = nn.ConvTranspose1d(8, 4, 3, 2, 1, 1)
        self.conv5 = nn.ConvTranspose1d(4, 2, 3, 2, 1, 1)
        self.conv6 = nn.Conv1d(2, 1, 3, 1, 1)
        
        self.input_layer = nn.Linear(latent_dim, 64*64)
        self.final_layer = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        input_layer = self.input_layer(X)
        input_layer = input_layer.view(-1, 64, 64)

        conv1 = self.conv1(input_layer)
        conv1 = self.relu(conv1)
        
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        
        conv4 = self.conv4(conv3)
        conv4 = self.relu(conv4) 
        
        conv5 = self.conv5(conv4)
        conv5 = self.relu(conv5) 

        conv6 = self.conv6(conv5)
        conv6 = self.sigmoid(conv6)
        
        return conv6
