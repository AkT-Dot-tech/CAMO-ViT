# CAMO-ViT
### A Vision Transformer Framework for Camouflaged Object Segmentation in Low-Visibility Aerial Imagery

## Overview

Identifying objects in aerial imagery can be challenging when their visual appearance closely resembles the surrounding environment. Similarities in color, texture, and structure can make camouflaged objects difficult to distinguish from the background.

CAMO-ViT explores this problem through pixel-level image segmentation. The project focuses on identifying camouflaged objects in aerial RGB images and generating segmentation masks that represent their boundaries.

The current implementation uses a pretrained SegFormer-B0 model adapted for binary segmentation through transfer learning. This implementation serves as the baseline for further experimentation and research-driven improvements.

## Understanding the Problem

Object detection answers the question: *Where is the object?*

Image segmentation takes this a step further: *Which exact pixels belong to the object?*

For camouflaged objects, this distinction is particularly important. An object may be present in an image but difficult to separate from its surroundings.

CAMO-ViT focuses on this pixel-level distinction by predicting a foreground mask that separates the target object from the background.

## Approach

The current baseline uses SegFormer-B0, a transformer-based semantic segmentation architecture.

Rather than training the entire model from scratch, the project uses transfer learning. A pretrained model is adapted to the specific task of binary segmentation by modifying its segmentation head to predict two classes: foreground and background.

The workflow consists of the following stages:

1. Preparing visible RGB images and their corresponding ground-truth masks.
2. Preprocessing the images and masks for model training.
3. Fine-tuning the pretrained SegFormer-B0 model.
4. Evaluating the trained model on a separate test set.
5. Generating segmentation predictions and visualizing the results.

## Dataset

The project uses the VIAC dataset, which contains visible RGB images, infrared images, and corresponding ground-truth masks.

The current implementation utilizes only the visible RGB images and their corresponding ground-truth segmentation masks. Infrared imagery is not used in this baseline.

The dataset is organized into the following splits:

- **Training:** 1,020 images
- **Validation:** 180 images
- **Testing:** 300 images
- **Total:** 1,500 images

The original training pool was divided into training and validation subsets. The test set was kept separate for final evaluation.

Ground-truth masks provide the correct pixel-level labels during training, allowing the model to learn the distinction between foreground objects and their surroundings.

## Model Configuration

The baseline model was trained using the following configuration:

- **Architecture:** SegFormer-B0
- **Pretrained Model:** `nvidia/segformer-b0-finetuned-ade-512-512`
- **Framework:** PyTorch
- **Learning Approach:** Transfer Learning
- **Task:** Binary Image Segmentation
- **Input Resolution:** 384 × 384 pixels
- **Number of Classes:** 2 (Foreground and Background)
- **Loss Function:** Combined Binary Cross-Entropy and Dice Loss
- **Optimizer:** AdamW
- **Learning Rate:** 0.0001
- **Batch Size:** 1
- **Maximum Epochs:** 30
- **Best Checkpoint:** Epoch 18

The pretrained segmentation head was adapted for binary segmentation. The model was trained to generate pixel-level predictions for the foreground and background regions.

## Evaluation Results

The baseline model was evaluated on 300 test images using standard segmentation metrics.

- **Dice Score:** 89.80%
- **Intersection over Union (IoU):** 81.48%
- **Precision:** 93.07%
- **Recall:** 86.74%
- **Mean Absolute Error (MAE):** 0.0301

These results establish the baseline performance of the current implementation. Further comparative experiments are required to determine its performance relative to other existing segmentation methods.

## Application

A Streamlit-based application has been developed to make the model accessible for image-level testing and demonstration.

The application supports:

- Uploading an input image.
- Generating segmentation predictions.
- Viewing the predicted mask.
- Inspecting the segmentation overlay on the original image.
- Visualizing prediction heatmaps.
- Downloading prediction results.

The application provides a functional interface for examining model outputs without directly interacting with the training scripts.

## Technology Stack

The project uses the following technologies:

- **Python:** Core programming language
- **PyTorch:** Model training and inference
- **Hugging Face Transformers:** Pretrained model integration
- **SegFormer-B0:** Baseline segmentation architecture
- **VIAC Dataset:** Training and evaluation data
- **Streamlit:** Interactive application

## Repository Structure

```text
CAMO-ViT/
├── app/                    # Streamlit application
├── data_and_model/          # Data processing, training and evaluation
├── checkpoints/             # Trained model checkpoints
├── input_images/            # Images for inference
├── outputs/                 # Evaluation and prediction outputs
├── dataset_inspection.png   # Dataset inspection visualization
├── requirements.txt         # Project dependencies
├── .gitignore
└── README.md
```

The raw dataset, local virtual environment, and Python cache files are excluded from the repository.

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/AkT-Dot.tech/CAMO-ViT.git
cd CAMO-ViT
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app/your_app_filename.py
```

Replace `your_app_filename.py` with the actual Streamlit application filename.

The VIAC dataset is not included in the repository. The trained model checkpoint and required dependencies must be available for inference.

## Current Limitations and Future Work

The current implementation establishes a functional baseline for camouflaged object segmentation. However, challenging cases remain, particularly when foreground and background features are highly similar or object boundaries are difficult to define.

The next phase of the project will investigate potential improvements to the model architecture and training methodology.

The planned research work includes:

- Investigating architectural modifications.
- Exploring improved feature extraction and boundary refinement.
- Experimenting with alternative training strategies and loss functions.
- Comparing the improved model against the current baseline and existing methods.
- Conducting ablation studies to understand the contribution of individual modifications.
- Evaluating robustness and generalization.

The final research direction will be determined through a review of existing literature and identification of a meaningful research gap.

## Project Status

### Completed

- Dataset verification and preprocessing
- Baseline model implementation
- Model training using transfer learning
- Test-set evaluation
- Prediction visualization
- Streamlit application development
- GitHub repository setup

### In Progress

- Identification and development of a novel research contribution

### Planned

- Comparative experiments
- Ablation studies
- Robustness and generalization evaluation
- Research paper preparation

## Author

B.Tech Computer Science and Engineering  
AI/ML Specialization

**Project Repository: (https://github.com/AkT-Dot.tech/CAMO-ViT)
