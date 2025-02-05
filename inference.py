import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
LATENT_DIM = 256

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNetVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetVAE, self).__init__()
        self.encoder = nn.Sequential(
            ResBlock(3, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 512, stride=2),
            ResBlock(512, 1024, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 1024 * 16 * 16)
        self.decoder = nn.Sequential(
            ResBlock(1024, 512, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(512, 256, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(256, 128, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(128, 64, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock(64, 32, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def encode(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        x = self.decoder_fc(z).view(z.size(0), 1024, 16, 16)
        return self.decoder(x)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py 'path_to_test_image'")
        sys.exit(1)
    test_image_path = sys.argv[1]
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    original_img = Image.open(test_image_path).convert("RGB")
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Load the model and weights (adjust weight filename if needed)
    model = ResNetVAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load("model_epoch_20.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        recon_img, _, _ = model(img_tensor)
    
    original_np = img_tensor.squeeze(0).cpu().permute(1,2,0).numpy()
    recon_np = img_tensor.squeeze(0).cpu().permute(1,2,0).numpy()
    
    # Plot original vs reconstructed image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(recon_np)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
