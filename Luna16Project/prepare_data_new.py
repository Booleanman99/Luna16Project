import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
train_img_path = "dataset/volumes_modified/trainImages.npy"
train_mask_path = "dataset/volumes_modified/trainMasks.npy"
test_img_path = "dataset/volumes_modified/testImages.npy"
test_mask_path = "dataset/volumes_modified/testMasks.npy"

# Output folders
os.makedirs("dataset/prepared_data/train/images", exist_ok=True)
os.makedirs("dataset/prepared_data/train/masks", exist_ok=True)
os.makedirs("dataset/prepared_data/test/images", exist_ok=True)
os.makedirs("dataset/prepared_data/test/masks", exist_ok=True)

# Load training data
train_img = np.load(train_img_path)
train_mask = np.load(train_mask_path)

if len(train_img) == 0 or len(train_mask) == 0:
    print("\n❌ No training data found. Exiting.")
    exit(1)

print(f"\n✅ Train set: {train_img.shape[0]} images")

for i, (img, mask) in enumerate(tqdm(zip(train_img, train_mask), total=len(train_img), desc="Generating training PNGs")):
    img = img.reshape(512, 512)
    mask = mask.reshape(512, 512)

    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mask[mask > 0] = 255  # Save as binary (0 or 255)

    cv2.imwrite(f"dataset/prepared_data/train/images/{i}.png", img.astype(np.uint8))
    cv2.imwrite(f"dataset/prepared_data/train/masks/{i}.png", mask.astype(np.uint8))

# Load test data
test_img = np.load(test_img_path)
test_mask = np.load(test_mask_path)

if len(test_img) == 0 or len(test_mask) == 0:
    print("\n❌ No test data found. Exiting.")
    exit(1)

print(f"\n✅ Test set: {test_img.shape[0]} images")

for i, (img, mask) in enumerate(tqdm(zip(test_img, test_mask), total=len(test_img), desc="Generating test PNGs")):
    img = img.reshape(512, 512)
    mask = mask.reshape(512, 512)

    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mask[mask > 0] = 255

    cv2.imwrite(f"dataset/prepared_data/test/images/{i}.png", img.astype(np.uint8))
    cv2.imwrite(f"dataset/prepared_data/test/masks/{i}.png", mask.astype(np.uint8))
