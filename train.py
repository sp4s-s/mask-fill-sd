import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from vae import VAE
from unet import UNet
from diffusion import Diffusion

class InpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_size=256):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)
    
def prepare_data(data_dir, batch_size=32, val_split=0.2):
    full_dataset = InpaintingDataset(data_dir)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    #  dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def generate_mask(batch_size, img_size=256, device='cuda'):
    masks = torch.ones((batch_size, 1, img_size, img_size), device=device)
    for i in range(batch_size):
        w, h = img_size, img_size
        mask_width = int(w * np.random.uniform(0.1, 0.5))
        mask_height = int(h * np.random.uniform(0.1, 0.5))
        x = np.random.randint(0, w - mask_width)
        y = np.random.randint(0, h - mask_height)
        masks[i, :, y:y+mask_height, x:x+mask_width] = 0
    return masks

def train_model(data_dir, epochs=100, batch_size=32, lr=1e-4, save_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = prepare_data(data_dir, batch_size)
    vae = VAE().to(device)
    unet = UNet().to(device)
    vae_path = os.path.join(save_dir, 'vae.pth')
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path))
        print("Loaded pretrained VAE")
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    diffusion = Diffusion()
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
        unet.train()
        train_loss = 0
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            masks = generate_mask(images.size(0), device=device)
            with torch.no_grad():
                z_mean, z_logvar = vae.encode(images)
                z = vae.reparameterize(z_mean, z_logvar)
            # Diffusion process
            t = diffusion.sample_timesteps(z.size(0), device)
            z_noisy, noise = diffusion.add_noise(z, t, device)
            z_masked = z * (1 - masks) + z_noisy * masks
            # UNet forward
            unet_input = torch.cat([z_masked, masks], dim=1)
            predicted_noise = unet(unet_input, t)
            # Loss calculation
            loss = F.mse_loss(predicted_noise * masks, noise * masks)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        # Validation phase
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                masks = generate_mask(images.size(0), device=device)
                z_mean, z_logvar = vae.encode(images)
                z = vae.reparameterize(z_mean, z_logvar)
                t = diffusion.sample_timesteps(z.size(0), device)
                z_noisy, noise = diffusion.add_noise(z, t, device)
                z_masked = z * (1 - masks) + z_noisy * masks
                unet_input = torch.cat([z_masked, masks], dim=1)
                predicted_noise = unet(unet_input, t)
                val_loss += F.mse_loss(predicted_noise * masks, noise * masks).item()
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch {epoch} Summary:')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unet.state_dict(), os.path.join(save_dir, 'unet_best.pth'))
            print('Saved best model!')
        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(unet.state_dict(), os.path.join(save_dir, f'unet_epoch_{epoch}.pth'))
    print("Training completed!")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train inpainting diffusion model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to directory containing training images')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir
    )
