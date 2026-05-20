# Skin Lesion Classification in Thermal Images using Deep Learning and Classical ML

Academic thesis (TCC) project investigating the classification of skin lesions in infrared/thermal images using convolutional neural networks (CNNs) and classical machine learning methods (SVM, Random Forest).

## Project Structure

```
skin-lesion-thermal-cnn/
├── data/
│   ├── raw/            # Original thermal images
│   ├── processed/      # Preprocessed images
│   └── splits/         # Train/validation/test splits
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training_and_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py       # Dataset loading and augmentation
│   ├── preprocessing.py # Image preprocessing pipeline
│   ├── model_cnn.py     # CNN model definitions
│   ├── model_classical.py # SVM and Random Forest classifiers
│   └── evaluate.py      # Evaluation metrics and visualization
├── results/             # Saved metrics, plots, and reports
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

<!-- Describe the thermal image dataset here: source, number of classes, image dimensions, acquisition protocol. -->

## Methods

### CNN

<!-- Describe the CNN architecture(s) used, e.g. custom CNN, ResNet transfer learning, fine-tuning strategy. -->

### SVM

<!-- Describe the SVM configuration: kernel, feature extraction method (HOG, hand-crafted, CNN embeddings). -->

### Random Forest

<!-- Describe the Random Forest setup: number of estimators, feature extraction method. -->

## Results

<!-- Summarize classification metrics (accuracy, F1, AUC-ROC) and comparison between methods. -->
