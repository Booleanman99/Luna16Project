import os
import subprocess

def step(msg):
    print(f"\n{'='*60}\n🛠️  {msg}\n{'='*60}\n")

def run_script(script_name, subset_name):
    subprocess.run(["python", script_name, subset_name], check=True)

def main():
    SUBSET = "subset6"  # Change this when working with other subsets

    folders = [
        f"dataset/subsets/{SUBSET}",
        "dataset/volumes_modified",
        "dataset/prepared_data/train/images",
        "dataset/prepared_data/train/masks",
        "dataset/prepared_data/test/images",
        "dataset/prepared_data/test/masks",
        "logs"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    if not os.listdir(f"dataset/subsets/{SUBSET}"):
        step(f"⚠️ No .mhd files in dataset/subsets/{SUBSET}")
        return

    if not os.path.exists("dataset/annotations.csv"):
        step("⚠️ Missing annotations.csv")
        return

    step("Extracting masks...")
    run_script("LUNA_mask_extraction.py", SUBSET)

    step("Segmenting lungs...")
    run_script("LUNA_segment_lung_ROI.py", SUBSET)

    step("Generating PNGs...")
    run_script("prepare_data_new.py", SUBSET)

    step("Training Mask R-CNN...")
    step("Training Mask R-CNN...")
    subprocess.run([
        "python", "Luna.py", "train",
        "--dataset=dataset/prepared_data",
        "--weights=imagenet",
        "--logs=logs/",
        "--subset=train"
    ], check=True)


    step("🎉 Training done!")

if __name__ == "__main__":
    main()
