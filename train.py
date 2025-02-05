import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IMG_SIZE = 512
LATENT_DIM = 256
BETA = 4  # KL divergence weight
GAMMA = 0.1  # Entropy weight

# Data transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


dataset = dset.ImageFolder(root="coco", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  

# Define ResNet Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# Define VAE with ResNet Blocks
class ResNetVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ResBlock(3, 64, 2),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            ResBlock(512, 1024, 2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            ResBlock(1024, 512),
            nn.Upsample(scale_factor=2),
            ResBlock(512, 256),
            nn.Upsample(scale_factor=2),
            ResBlock(256, 128),
            nn.Upsample(scale_factor=2),
            ResBlock(128, 64),
            nn.Upsample(scale_factor=2),
            ResBlock(64, 32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x).view(x.shape[0], -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z).view(z.shape[0], 1024, 1, 1)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

#Entropy based loss function
def entropy_loss(z):
    p = F.softmax(z, dim=-1)
    return -torch.sum(p * torch.log(p + 1e-8), dim=-1).mean()

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    entropy = entropy_loss(mu)
    return recon_loss + BETA * kl_div + GAMMA * entropy

#Training loop
# Initialize model, optimizer
vae = ResNetVAE(LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)

# Training Loop
EPOCHS = 20
vae.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon_imgs, mu, logvar = vae(imgs)
        loss = loss_function(recon_imgs, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

# Save the trained model
torch.save(vae.state_dict(), "model.pth")


