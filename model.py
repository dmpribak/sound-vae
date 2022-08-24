import torch.nn as nn
from torch.nn.modules import conv
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        hidden_dims = [2048, 1024, 512, 256, 128]

        self.conv1 = nn.Conv2d(1, 128, (1,128), 1, 0)
        self.conv2 = nn.Conv2d(128, 256, (3,1), 2, (1,0))
        self.conv3 = nn.Conv2d(256, 512, (3,1), 2, (1,0))
        self.conv4 = nn.Conv2d(128, 256, (3,1), 2, (1,0))
        self.conv5 = nn.Conv2d(256, 512, (3,1), 2, (1,0))
            
        self.linear_mu = nn.Linear(512*256, latent_dim)
        self.linear_sigma = nn.Linear(512*256*1, latent_dim)
        self.relu = nn.Tanh()
    
    def forward(self, X):
        conv1 = self.conv1(X)
        conv1 = self.relu(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        
        #conv4 = self.conv4(conv3)
        #conv4 = self.relu(conv4) #32x128

        
        #conv5 = self.conv5(conv4)
        #conv5 = self.relu(conv5) #512x64x1

        flat = torch.flatten(conv3, 1)
        
        mu = self.linear_mu(flat)
        sigma = self.linear_sigma(flat)
        
        return mu, sigma
        
        
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(512, 256, (3,1), 2, (1,0), (1,0))
        self.conv2 = nn.ConvTranspose2d(256, 128, (3,1), 2, (1,0), (1,0))
        self.conv3 = nn.ConvTranspose2d(512, 256, (3,1), 2, (1,0), (1,0))
        self.conv4 = nn.ConvTranspose2d(256, 128, (3,1), 2, (1,0), (1,0))
        self.conv5 = nn.ConvTranspose2d(128, 1, (1,128), 1, (0,0), (0,0))
        
        self.input_layer = nn.Linear(latent_dim, 512*256*1)

        self.relu = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        input_layer = self.input_layer(X)
        input_layer = input_layer.view(-1, 512, 256, 1)

        #conv1 = self.conv1(input_layer)
        #conv1 = self.relu(conv1)
        
        #conv2 = self.conv2(conv1)
        #conv2 = self.relu(conv2)
        
        conv3 = self.conv3(input_layer)
        conv3 = self.relu(conv3)
        
        conv4 = self.conv4(conv3)
        conv4 = self.relu(conv4) 

        conv5 = self.conv5(conv4)
        conv5 = self.sigmoid(conv5) 

        return conv5
