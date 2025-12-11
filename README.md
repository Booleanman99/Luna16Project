# Luna16 Project: Automated Lung Nodule Detection using Deep Learning

A deep learning pipeline for automated lung nodule detection and segmentation on CT scans using the LUNA16 (LUng Nodule Analysis 2016) dataset. Developed for BME-515 Medical Imaging course.

## Summary

This project implements an end-to-end computer-aided detection (CAD) system for pulmonary nodule detection, addressing the critical need for automated lung cancer screening. The system uses a two-stage deep learning approach combining instance segmentation with classification.

### Clinical Significance

Lung cancer is the leading cause of cancer-related death worldwide. Early detection through CT screening significantly improves survival rates, but analyzing millions of scans is a substantial burden for radiologists. This project aims to assist in that screening process.

### Technical Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| **Stage 1: Segmentation** | Mask R-CNN with ResNet50 backbone | Instance segmentation of nodule regions |
| **Stage 2: Detection** | Region Proposal Network (RPN) | Bounding box detection and classification |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet50 |
| Image Size | 512 x 512 |
| Number of Classes | 2 (Background, Nodule) |
| RPN Anchor Scales | (8, 16, 32, 64, 128) |
| Detection Max Instances | 400 |
| Steps per Epoch | 710 |
| Validation Steps | 177 |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Mean Average Precision (mAP) | 0.2487 |
| Mean Dice Coefficient | 0.9859 |
| Mean ROC-AUC | 0.6722 |
| Test Images | 77 |

## Report Location

The complete project report and documentation can be found here:
- **[`Luna16Project Report.pdf`](./Luna16Project%20Report.pdf)** - Full technical report
- **[`Inferences.pdf`](./Inferences.pdf)** - Model inference results and visualizations
- **[`Luna_Inference.pdf`](./Luna_Inference.pdf)** - Detailed inference analysis
- **[`How To Run.docx`](./How%20To%20Run.docx)** - Step-by-step execution instructions

## Repository Structure

```
Luna16Project/
├── Luna16Project/              # Source code directory
│   ├── Luna.py                 # Main training script with hyperparameters
│   ├── Luna_Inference.ipynb    # Inference notebook
│   ├── Unet/                   # U-Net segmentation module
│   ├── src/                    # Mask R-CNN source code
│   ├── models/                 # Trained model weights (.pth files)
│   ├── Evaluation/             # Evaluation scripts and metrics
│   └── dataset/                # Data preparation scripts
├── dataset/                    # Training/test data
├── Luna16Project Report.pdf    # Technical documentation
├── Inferences.html/pdf         # Model output visualizations
└── How To Run.docx             # Execution instructions
```

## Dataset

The project uses the [LUNA16 dataset](https://luna16.grand-challenge.org/Data/), a publicly available benchmark for lung nodule detection derived from the LIDC/IDRI database.

### Data Preparation
```bash
mkdir dataset/volumes dataset/volumes/images dataset/volumes/masks
mkdir dataset/volumes_modified dataset/volumes_modified/images dataset/volumes_modified/masks
cp download.sh dataset/volumes/
./dataset/volumes/download.sh
./dataset/volumes/extract.sh
```

## Installation

```bash
pip install src/Mask_RCNN/requirements.txt
python src/Mask_RCNN/setup.py install
pip install -r requirements.txt
```

### Trained Weights

Pre-trained weights are available on [Google Drive](https://drive.google.com/drive/folders/1h8nu07VJ_AxVdplNk8sQdjw1SssJOuJx?usp=sharing). After downloading:
```bash
mkdir logs/
# Copy Luna.zip to logs/ and extract
```

## Usage

### Training
```bash
python Luna.py train --dataset=dataset/prepared_data --weights=imagenet --logs logs/
```

### Inference
Run the `Luna_Inference.ipynb` notebook for model inference and visualization.

## Methods

1. **Data Preprocessing**: CT scan normalization, lung ROI extraction, and mask generation
2. **Segmentation**: Mask R-CNN with Feature Pyramid Network (FPN) for multi-scale detection
3. **Validation**: K-fold cross-validation with mAP, Dice coefficient, and ROC-AUC metrics
4. **Visualization**: Bounding box and mask overlay on detected nodules

## Technologies

- Python 3.x
- TensorFlow / Keras
- PyTorch (U-Net models)
- Mask R-CNN
- NumPy, Matplotlib, scikit-learn

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Based on the LUNA16 Grand Challenge framework. Original Mask R-CNN implementation by Matterport.
