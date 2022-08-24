import torch.nn as nn
from torch.nn.modules import conv
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        hidden_dims = [2048, 1024, 512, 256, 128]

        self.conv1 = nn.Conv1d(1, 4, 3, 2, 1)
        self.conv2 = nn.Conv1d(4, 8, 3, 2, 1)
        self.conv3 = nn.Linear(512, 256)
        self.conv4 = nn.Linear(256, 128)
            
        self.linear_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.linear_sigma = nn.Linear(hidden_dims[-1], latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        linear1 = self.linear1(X)
        linear1 = self.relu(linear1)
        
        linear2 = self.linear2(linear1)
        linear2 = self.relu(linear2)
        
        linear3 = self.linear3(linear2)
        linear3 = self.relu(linear3)
        
        linear4 = self.linear4(linear3)
        linear4 = self.relu(linear4)
            
        mu = self.linear_mu(linear4)
        sigma = self.linear_mu(linear4)
        
        return mu, sigma
        
        
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        hidden_dims = [128, 256, 512, 1028, 2048]
        
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1028)
        self.linear4 = nn.Linear(1028, 2048)
        
        self.input_layer = nn.Linear(latent_dim, hidden_dims[0])
        self.final_layer = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        input_layer = self.input_layer(X)

        linear1 = self.linear1(input_layer)
        linear1 = self.relu(linear1)
        
        linear2 = self.linear2(linear1)
        linear2 = self.relu(linear2)
        
        linear3 = self.linear3(linear2)
        linear3 = self.relu(linear3)
        
        linear4 = self.linear4(linear3)
        linear4 = self.sigmoid(linear4)
        
        #output = self.final_layer(linear4)
        #output = self.sigmoid(output)
        
        return linear4
