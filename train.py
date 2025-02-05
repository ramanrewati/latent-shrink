import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 512
LATENT_DIM = 256
BETA = 4
GAMMA = 0.1
BATCH_SIZE = 16
EPOCHS = 20
TRAIN_SUBSET_SIZE = 10000  #

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load dataset and perform train/test split (90/10)
dataset = dset.ImageFolder(root="coco", transform=transform)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Trim test dataset to 900 images (if available)
if len(test_dataset) > 900:
    test_dataset = Subset(test_dataset, list(range(900)))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Select a fixed test image (first image in the trimmed test set)
sample_img, _ = test_dataset[0]
sample_img = sample_img.unsqueeze(0).to(device)  # [1, 3, 512, 512]

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
            ResBlock(3, 64, stride=2),      # 512 -> 256
            ResBlock(64, 128, stride=2),     # 256 -> 128
            ResBlock(128, 256, stride=2),    # 128 -> 64
            ResBlock(256, 512, stride=2),    # 64 -> 32
            ResBlock(512, 1024, stride=2),   # 32 -> 16
            nn.AdaptiveAvgPool2d(1)          # 16 -> 1
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
        std = torch.exp(0.5 * logvar) t
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        x = self.decoder_fc(z).view(z.size(0), 1024, 16, 16)
        return self.decoder(x)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def entropy_loss(z):
    p = F.softmax(z, dim=-1)
    return -torch.sum(p * torch.log(p + 1e-8), dim=-1).mean()

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + BETA * kl_div + GAMMA * entropy_loss(mu)

model = ResNetVAE(LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

performance_log = {}
# To store reconstruction of the sample image per epoch
reconstructions = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    indices = torch.randperm(len(train_dataset))[:TRAIN_SUBSET_SIZE].tolist()
    epoch_train_subset = Subset(train_dataset, indices)
    epoch_train_loader = DataLoader(epoch_train_subset, batch_size=BATCH_SIZE, shuffle=True)
    
    progress_bar = tqdm(epoch_train_loader, desc=f"Epoch {epoch+1} Training", total=len(epoch_train_loader))
    processed_images = 0
    for imgs, _ in progress_bar:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon_imgs, mu, logvar = model(imgs)
        loss = loss_function(recon_imgs, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        processed_images += imgs.size(0)
        progress_bar.set_postfix({"Processed": f"{processed_images}/{TRAIN_SUBSET_SIZE}", "Batch Loss": f"{loss.item():.2f}"})
    
    train_loss_avg = train_loss / TRAIN_SUBSET_SIZE

    model.eval()
    test_loss = 0
    progress_bar_test = tqdm(test_loader, desc=f"Epoch {epoch+1} Testing", total=len(test_loader))
    with torch.no_grad():
        for imgs, _ in progress_bar_test:
            imgs = imgs.to(device)
            recon_imgs, mu, logvar = model(imgs)
            loss = loss_function(recon_imgs, imgs, mu, logvar)
            test_loss += loss.item()
    test_loss_avg = test_loss / len(test_dataset)
    performance_log[epoch+1] = {'train_loss': train_loss_avg, 'test_loss': test_loss_avg}
    print(f"Epoch {epoch+1}: Train Loss: {train_loss_avg:.4f} | Test Loss: {test_loss_avg:.4f}")
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    
    # Get reconstruction for the fixed sample image
    with torch.no_grad():
        recon_sample, _, _ = model(sample_img)
    # Convert tensor to numpy array for plotting
    recon_np = recon_sample.squeeze(0).cpu().permute(1, 2, 0).numpy()
    reconstructions.append(recon_np)

# Save performance log to text file
with open("performance_log.txt", "w") as f:
    for epoch, metrics in performance_log.items():
        f.write(f"Epoch {epoch}: Train Loss: {metrics['train_loss']:.4f} | Test Loss: {metrics['test_loss']:.4f}\n")

epochs = list(performance_log.keys())
train_losses = [performance_log[e]['train_loss'] for e in epochs]
test_losses = [performance_log[e]['test_loss'] for e in epochs]

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Loss Curves")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

# Plot original vs. reconstruction over epochs for the fixed sample image
# Convert the original image to numpy for plotting
original_np = sample_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

num_epochs = len(reconstructions)
fig, axes = plt.subplots(num_epochs, 2, figsize=(6, num_epochs*2))
for i in range(num_epochs):
    if i == 0:
        axes[i,0].imshow(original_np)
        axes[i,0].set_title("Original")
        axes[i,0].axis("off")
    else:
        axes[i,0].axis("off")
    axes[i,1].imshow(reconstructions[i])
    axes[i,1].set_title(f"Epoch {i+1}")
    axes[i,1].axis("off")
plt.tight_layout()
plt.savefig("reconstruction_progress.png")
plt.show()
