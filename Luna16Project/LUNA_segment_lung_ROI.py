import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage import morphology, measure
from skimage.transform import resize
from sklearn.cluster import KMeans

# Paths
input_root = "./dataset/subsets/"
output_root = "./dataset/volumes_modified/"
os.makedirs(output_root, exist_ok=True)

# Function to process one scan
def process_scan(img_array):
    imgs_to_process = img_array.astype(np.float32)
    lung_masks = np.zeros_like(imgs_to_process)

    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std

        middle = img[100:400, 100:400]
        mean_middle = np.mean(middle)
        img[img == np.max(img)] = mean_middle
        img[img == np.min(img)] = mean_middle

        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [-1, 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)

        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))

        labels = measure.label(dilation)
        regions = measure.regionprops(labels)
        good_labels = [prop.label for prop in regions
                       if prop.bbox[2] - prop.bbox[0] < 475 and
                          prop.bbox[3] - prop.bbox[1] < 475 and
                          prop.bbox[0] > 40 and prop.bbox[2] < 472]

        mask = np.zeros([512, 512], dtype=np.int8)
        for N in good_labels:
            mask += np.where(labels == N, 1, 0)

        mask = morphology.dilation(mask, np.ones([10, 10]))
        lung_masks[i] = mask

    return imgs_to_process, lung_masks

# Start processing all subsets
for subset_idx in range(10):
    subset_name = f"subset{subset_idx}"
    subset_path = os.path.join(input_root, subset_name)
    files = [f for f in os.listdir(subset_path) if f.endswith(".mhd")]
    files = sorted(files)  # sort to keep order

    print(f"📂 Processing {subset_name}: {len(files)} scans found")

    for f_idx, f in enumerate(tqdm(files, desc=f"Subset{subset_idx}")):
        save_prefix = f"{subset_name}_{f.replace('.mhd', '')}"

        # ✅ Check if already saved
        image_save_path = os.path.join(output_root, f"trainImages_{save_prefix}.npy")
        mask_save_path = os.path.join(output_root, f"trainMasks_{save_prefix}.npy")

        if os.path.exists(image_save_path) and os.path.exists(mask_save_path):
            print(f"⏩ Skipping {save_prefix} (already saved)")
            continue

        # Process normally if not saved
        img_path = os.path.join(subset_path, f)
        itk_img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(itk_img)

        imgs_to_process, lung_masks = process_scan(img_array)

        scan_images = []
        scan_masks = []

        for i in range(len(imgs_to_process)):
            mask = lung_masks[i]
            img = imgs_to_process[i] * mask

            if np.sum(mask) == 0:
                continue

            new_mean = np.mean(img[mask > 0])
            new_std = np.std(img[mask > 0])
            img[img == np.min(img)] = new_mean - 1.2 * new_std
            img = (img - new_mean) / new_std

            labels = measure.label(mask)
            regions = measure.regionprops(labels)

            min_row, max_row, min_col, max_col = 512, 0, 512, 0
            for prop in regions:
                B = prop.bbox
                min_row = min(min_row, B[0])
                min_col = min(min_col, B[1])
                max_row = max(max_row, B[2])
                max_col = max(max_col, B[3])

            width, height = max_col - min_col, max_row - min_row
            if width <= 5 or height <= 5:
                continue

            if width > height:
                max_row = min_row + width
            else:
                max_col = min_col + height

            img_crop = img[min_row:max_row, min_col:max_col]
            mask_crop = mask[min_row:max_row, min_col:max_col]

            img_crop -= np.mean(img_crop)
            norm_range = np.max(img_crop) - np.min(img_crop)
            if norm_range > 0:
                img_crop /= norm_range

            scan_images.append(resize(img_crop, [512, 512]))
            scan_masks.append(resize(mask_crop, [512, 512]))

        # Save after processing each scan
        if scan_images:
            scan_images_np = np.array(scan_images)[:, None, :, :].astype(np.float32)
            scan_masks_np = np.array(scan_masks)[:, None, :, :].astype(np.float32)

            np.save(image_save_path, scan_images_np)
            np.save(mask_save_path, scan_masks_np)

        # Free memory
        del imgs_to_process
        del lung_masks
        del scan_images
        del scan_masks
        del itk_img
        del img_array

    print(f"✅ Completed and saved all scans in {subset_name}")

print("🎉 ALL subsets processed. All scans individually saved. Ready to merge!")
