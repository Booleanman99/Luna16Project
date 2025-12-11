# train_unet.py
# 🚀 Train U-Net directly from batch_*.npz files (merge_temp)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm
from BatchLazyDataset import BatchLazyDataset
from unet_model import UNet

def train(model, loader, criterion, optimizer, device):
    print("🧪 Beginning training loop...")
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Training batches")):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def main():
    print("🚀 Starting U-Net training pipeline (batch-based)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔌 Using device: {device}")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # Dataset
    print("📂 Loading dataset from merge_temp/...")
    dataset = BatchLazyDataset("../dataset/volumes_modified/merge_temp", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    print("🧠 Initializing U-Net model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Model save directory
    os.makedirs("../models", exist_ok=True)

    # Training
    epochs = 10
    for epoch in range(epochs):
        print(f"\n📚 Epoch {epoch+1}/{epochs}")
        loss = train(model, dataloader, criterion, optimizer, device)
        print(f"✅ Epoch {epoch+1} Complete - Avg Loss: {loss:.4f}")

        checkpoint_path = f"../models/unet_batch_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved model checkpoint: {checkpoint_path}")

    print("🎉 Training complete! All checkpoints saved at ../models/")

if __name__ == "__main__":
    main()
