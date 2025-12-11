import os
import numpy as np
from tqdm import tqdm

# Paths
input_dir = "../dataset/volumes_modified/"
save_dir = "../dataset/volumes_modified/"
resume_dir = "../dataset/volumes_modified/merge_temp/"
os.makedirs(resume_dir, exist_ok=True)

# Collect all scan files
image_files = sorted([f for f in os.listdir(input_dir) if f.startswith("trainImages_")])
mask_files = sorted([f for f in os.listdir(input_dir) if f.startswith("trainMasks_")])

print(f"📦 Found {len(image_files)} image files and {len(mask_files)} mask files to merge.")

assert len(image_files) == len(mask_files), "Mismatch between images and masks!"

small_batch_slices = 1000  # Maximum slices before saving partial batch
slice_counter = 0
batch_counter = 0

partial_images = []
partial_masks = []

# Detect already merged batches
existing_batches = sorted([int(f.replace("batch_", "").replace(".npz", "")) for f in os.listdir(resume_dir) if f.startswith("batch_")])
if existing_batches:
    last_batch_done = max(existing_batches)
    batch_counter = last_batch_done + 1
    print(f"🔁 Resume detected. Skipping already merged batches up to batch {last_batch_done}.")
else:
    print("🚀 Starting fresh merge.")

# Start merging
for idx, (img_f, mask_f) in enumerate(tqdm(zip(image_files, mask_files), total=len(image_files), desc="Merging slices")):
    img_arr = np.load(os.path.join(input_dir, img_f))
    mask_arr = np.load(os.path.join(input_dir, mask_f))

    slices = img_arr.shape[0]

    partial_images.append(img_arr)
    partial_masks.append(mask_arr)
    slice_counter += slices

    # Save partial batch if slice limit reached
    if slice_counter >= small_batch_slices:
        images_np = np.concatenate(partial_images, axis=0)
        masks_np = np.concatenate(partial_masks, axis=0)

        np.savez_compressed(os.path.join(resume_dir, f"batch_{batch_counter}.npz"), images=images_np, masks=masks_np)
        
        batch_counter += 1
        partial_images = []
        partial_masks = []
        slice_counter = 0

# Save remaining slices
if partial_images:
    images_np = np.concatenate(partial_images, axis=0)
    masks_np = np.concatenate(partial_masks, axis=0)

    np.savez_compressed(os.path.join(resume_dir, f"batch_{batch_counter}.npz"), images=images_np, masks=masks_np)

print("✅ All slices processed. Loading batches for final merge...")

# Load all saved batches
merged_images = []
merged_masks = []

batch_files = sorted([f for f in os.listdir(resume_dir) if f.startswith("batch_")])

for batch_file in tqdm(batch_files, desc="Loading saved batches"):
    batch_data = np.load(os.path.join(resume_dir, batch_file))
    merged_images.append(batch_data["images"])
    merged_masks.append(batch_data["masks"])

# Final big merge
all_images = np.concatenate(merged_images, axis=0)
all_masks = np.concatenate(merged_masks, axis=0)

print(f"✅ Total merged: {all_images.shape[0]} slices.")

# Shuffle and split
rand_idx = np.random.permutation(all_images.shape[0])
all_images = all_images[rand_idx]
all_masks = all_masks[rand_idx]

split_idx = int(0.2 * all_images.shape[0])

np.save(os.path.join(save_dir, "trainImages.npy"), all_images[split_idx:])
np.save(os.path.join(save_dir, "trainMasks.npy"), all_masks[split_idx:])
np.save(os.path.join(save_dir, "testImages.npy"), all_images[:split_idx])
np.save(os.path.join(save_dir, "testMasks.npy"), all_masks[:split_idx])

print("🎯 Merge complete! Final datasets ready:")
print(f"Train: {all_images[split_idx:].shape}, Test: {all_images[:split_idx].shape}")
