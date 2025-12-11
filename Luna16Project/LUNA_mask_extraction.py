import pandas as pd
import numpy as np
import os
import glob
import SimpleITK as sitk
from tqdm import tqdm

def make_mask(center, diameter, z, width, height, spacing, origin):
    mask = np.zeros([height, width], dtype=np.uint8)
    center = np.array(center)
    origin = np.array(origin[:2])
    spacing = np.array(spacing[:2])
    v_center = (center[:2] - origin) / spacing
    v_diameter = diameter / spacing[0]
    for x in range(width):
        for y in range(height):
            p = np.array([y, x])
            if np.linalg.norm(p - v_center) <= v_diameter:
                mask[x, y] = 1
    return mask

# ✅ Correct paths
annotations_file = "dataset/annotations.csv"
subset_path = "dataset/subsets/subset6"
output_path = "dataset/volumes_modified/"
os.makedirs(output_path, exist_ok=True)

# Load annotations
df = pd.read_csv(annotations_file)

# Grab list of .mhd files
file_list = glob.glob(os.path.join(subset_path, "*.mhd"))

fcount = 0
for img_file in tqdm(file_list):
    seriesuid = os.path.basename(img_file).replace(".mhd", "")
    mini_df = df[df["seriesuid"] == seriesuid]
    if mini_df.shape[0] > 0:
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        num_z, height, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())

        for _, row in mini_df.iterrows():
            center = np.array([row["coordX"], row["coordY"], row["coordZ"]])
            diameter = row["diameter_mm"]
            v_center = np.rint((center - origin) / spacing).astype(np.int16)

            imgs = np.ndarray([3, 512, 512], dtype=np.float32)
            masks = np.ndarray([3, 512, 512], dtype=np.uint8)

            for i, i_z in enumerate(np.arange(v_center[2] - 1, v_center[2] + 2).clip(0, num_z - 1)):
                mask = make_mask(center, diameter, origin[2] + spacing[2] * i_z,
                                 width, height, spacing, origin)

                img = img_array[i_z].astype(np.float32)
                img = (img - np.mean(img)) / np.std(img)

                imgs[i] = img
                masks[i] = mask

            np.save(os.path.join(output_path, f"images_{fcount:04d}.npy"), imgs)
            np.save(os.path.join(output_path, f"masks_{fcount:04d}.npy"), masks)
            fcount += 1
