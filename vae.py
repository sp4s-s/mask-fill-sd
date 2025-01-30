import torch
import torch.nn as nn
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, h_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim*2, h_dim*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim*4, latent_channels*2, kernel_size=3, padding=1),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, h_dim*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim*4, h_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim*2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar





