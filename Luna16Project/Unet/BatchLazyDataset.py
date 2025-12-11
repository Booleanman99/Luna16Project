import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

class BatchLazyDataset(Dataset):
    def __init__(self, batch_dir, transform=None):
        print("📁 Initializing BatchLazyDataset...")
        self.batch_dir = batch_dir
        self.transform = transform

        self.slice_pointers = []  # (batch_idx, slice_idx)
        self.valid_batch_files = []

        batch_files = sorted([f for f in os.listdir(batch_dir) if f.startswith("batch_") and f.endswith(".npz")])

        for batch_file in batch_files:
            batch_path = os.path.join(batch_dir, batch_file)
            try:
                batch_data = np.load(batch_path)
                num_slices = batch_data['images'].shape[0]
                for i in range(num_slices):
                    self.slice_pointers.append((len(self.valid_batch_files), i))
                self.valid_batch_files.append(batch_file)
            except Exception as e:
                print(f"⚠️ Skipping corrupted batch: {batch_file} due to {e}")

        # ✅ Limit to first 5000 slices for faster training
        self.slice_pointers = self.slice_pointers[:5000]

    def __len__(self):
        return len(self.slice_pointers)

    def __getitem__(self, idx):
        batch_idx, slice_idx = self.slice_pointers[idx]
        batch_file = self.valid_batch_files[batch_idx]
        batch_path = os.path.join(self.batch_dir, batch_file)

        try:
            batch_data = np.load(batch_path)

            img = batch_data['images'][slice_idx]
            mask = batch_data['masks'][slice_idx]

            # 🛠 Squeeze to remove extra dimensions (1,512,512) -> (512,512)
            img = np.squeeze(img)
            mask = np.squeeze(mask)

            # 🛠 Normalize if needed (you can skip if already 0–1 range)
            img = (img * 255).astype(np.uint8)
            mask = (mask * 255).astype(np.uint8)

            img = Image.fromarray(img).convert("L")
            mask = Image.fromarray(mask).convert("L")

            if self.transform:
                img = self.transform(img)
                mask = self.transform(mask)

            return img, mask

        except Exception as e:
            print(f"⚠️ Error loading slice idx {idx} in batch {batch_file}: {e}")
            dummy_img = torch.zeros((1, 256, 256), dtype=torch.float32)
            dummy_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
            return dummy_img, dummy_mask
