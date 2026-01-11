# ELEC0134 AMLS Assignment - BreastMNIST Classification

## Project Overview

For this assignment, we benchmark the performance of **two machine learning models** on a **single medical image dataset from MedMNIST (BreastMNIST)**. The goal is to understand how **model capacity**, **data augmentation**, and **training budget** influence performance across both **classical** and **deep learning** approaches.

### What this repository does

- **Dataset**: Loads the official MedMNIST-style `BreastMNIST.npz` from `Datasets/` and uses the provided **train/val/test** splits.
- **Model A (Classical ML)**: Extracts features (**raw pixels / PCA / HOG**) and trains an **SVM (RBF)** using **GridSearchCV**, then reports metrics.
- **Model B (Deep Learning)**: Trains a small **ResNet-style CNN** in PyTorch with early stopping and optional augmentation (as configured).
- **Evaluation**: Reports **Accuracy, Precision, Recall, F1-score** (and ROC-AUC where probabilities are available), plus confusion matrices and saved plots when `--out_dir` is provided.

### Assignment factors you can vary (experiments)

- **Training budget**: `--train_fraction`, `--val_fraction`
- **Representation / preprocessing (Model A)**: `raw` vs `pca` vs `hog`
- **Augmentation**:
  - **Model A**: compares **without vs with** augmentation using the augmentation settings in `model_A/config.yaml`
  - **Model B**: controlled via `model_B/config.yaml` (and optional CLI ablation flag if enabled in your code)

## Project Structure

```
AMLS_25_26_SN24205421/
├── model_A/              # Classical ML models (Category A)
│   ├── __init__.py
│   ├── dataset.py        # Dataset loading utilities
│   ├── preprocessing.py  # Image preprocessing
│   ├── features.py       # Feature extraction (raw, PCA, HOG)
│   ├── classifier.py     # SVM classifier
│   ├── trainer.py        # Training utilities
│   ├── evaluator.py      # Evaluation utilities
│   └── augmentation.py   # Data augmentation
│   └── config.yaml       # Configuration file for Model A
├── model_B/              # Deep learning models (Category B)
│   ├── __init__.py
│   ├── dataset.py        # PyTorch DataLoader utilities
│   ├── model.py          # ResNet-style network architecture
│   ├── trainer.py        # Training with early stopping
│   ├── evaluator.py      # Evaluation utilities
│   ├── augmentation.py   # Data augmentation (PyTorch)
│   └── config.yaml       # Configuration file for Model B
├── Datasets/             # Dataset directory 
│   └── .gitkeep          # An empty placeholder so Git tracks the empty folder (DO NOT commit dataset files)
├── main.py               # Main entry point (includes utility functions)
├── results/              # Output directory (auto-created; safe to delete/re-generate)
│   ├── comparison_summary.md
│   ├── model_comparison_metrics.png
│   ├── model_comparison_confusion_matrices.png
│   └── model_comparison_roc_curves.png
└── README.md             # This file
```

## Setup

### Requirements

Install the required packages (these match the imports used in this repo):

```bash
pip install numpy scipy scikit-learn scikit-image matplotlib seaborn torch torchvision pyyaml tqdm pillow
```

Notes:
- `scikit-image` is required for **HOG** features in Model A.
- `seaborn` is used for the evaluation plots (confusion matrices, metric bars, etc.).
- `pillow` is required by the PyTorch image pipeline (`PIL.Image`).

### Dataset

For local runs, place the BreastMNIST dataset file (`BreastMNIST.npz`) directly in `Datasets/`.

For submission/marking, **do not commit any dataset files**. The marker will provide `BreastMNIST.npz` in `Datasets/` during evaluation.

**Important:** Do not save any processed or intermediate data in the `Datasets/` folder. Only the original dataset files should be present.

If you accidentally committed a dataset file previously, remove it from Git history/tracking (while keeping your local copy) using `git rm --cached` and commit that change.

## Usage

### Basic Usage

Run the project with a single command from the project root (runs both models by default):

```bash
python main.py
```

If you are on Windows and `python` is not available as a command, use:

```bash
py main.py
```

Or run a specific model:

```bash
python main.py --model A --mode train
python main.py --model B --mode train
```

### Command-Line Arguments

```
--model {A,B,both}     Model to train (A for classical ML, B for deep learning, both for both models, default: both)
--mode {train,test}    Mode: train or test (default: train)
--config PATH          Path to YAML configuration file (optional, defaults to model-specific configs)
--config_A PATH        Path to YAML configuration file for Model A (default: model_A/config.yaml)
--config_B PATH        Path to YAML configuration file for Model B (default: model_B/config.yaml)
--data_dir PATH        Path to directory containing BreastMNIST.npz file (default: Datasets)
--out_dir PATH         Output directory for results and checkpoints (default: results)
--train_fraction FLOAT Fraction of training set to use (0 < fraction <= 1.0, default: 1.0)
--val_fraction FLOAT   Fraction of validation set to use (default: 1.0)
--feature_mode {raw,pca,hog}  Feature mode for Model A (default: raw)
--compare_all_features Compare all feature modes (raw, PCA, HOG) for Model A
--show_examples        Visualize sample images from the dataset
--model_b_ablation {on,off}   Model B only: run augmentation ablation OFF vs ON (on) or do a single run (off)
```

### Model A (Classical ML - SVM)

#### Train with default settings:
```bash
python main.py --model A --mode train
```

#### Train with specific feature mode:
```bash
python main.py --model A --mode train --feature_mode pca
```

#### Compare all feature modes (raw, PCA, HOG):
```bash
python main.py --model A --mode train --compare_all_features --out_dir results/model_A
```

#### Train with custom configuration:
```bash
python main.py --model A --mode train --config model_A/config.yaml --out_dir results/model_A
```

Or use the default config (automatically uses `model_A/config.yaml` if it exists):
```bash
python main.py --model A --mode train --out_dir results/model_A
```

#### Train with reduced training budget:
```bash
python main.py --model A --mode train --train_fraction 0.5 --val_fraction 0.5
```

#### Visualize sample images:
```bash
python main.py --model A --mode train --show_examples
```

### Model B (Deep Learning - ResNet-style)

#### Train with default settings:
```bash
python main.py --model B --mode train
```

#### Train with custom configuration:
```bash
python main.py --model B --mode train --config model_B/config.yaml --out_dir results/model_B
```

Or use the default config (automatically uses `model_B/config.yaml` if it exists):
```bash
python main.py --model B --mode train --out_dir results/model_B
```

#### Train without augmentation:
Edit `model_B/config.yaml` and set `augmentation.enabled: false`, then:
```bash
python main.py --model B --mode train --config model_B/config.yaml
```

#### Train with reduced training budget:
```bash
python main.py --model B --mode train --train_fraction 0.25 --val_fraction 0.25
```

#### Faster CPU run (skip augmentation ablation):
```bash
python main.py --model B --mode train --model_b_ablation off
```

## Configuration Files

Configuration files are in YAML format and allow you to customize:

### Model A Configuration (`model_A/config.yaml`)

- **model**: Model hyperparameters (model_type, kernel, C, gamma)
- **features**: Feature extraction settings (mode, pca_components, standardize)
- **data**: Data loading settings (train_fraction, val_fraction, normalize)

### Model B Configuration (`model_B/config.yaml`)

- **model**: Network architecture (base_channels, num_blocks, dropout_rate)
- **training**: Training hyperparameters (num_epochs, batch_size, learning_rate, etc.)
  - `run_ablation`: If true, trains OFF vs ON augmentation (2× training time). If false, trains once.
  - `num_workers`: DataLoader workers (0 recommended on Windows).
  - `torch_num_threads` / `torch_num_interop_threads`: Optional CPU thread controls for reproducibility/perf.
- **data**: Data loading settings
- **augmentation**: Augmentation parameters (rotation_range, translation_range, etc.)

## Features

### Dataset Loading
- Loads BreastMNIST data from `.npz` files
- Supports train/val/test splits
- Optional normalization (z-score using training set statistics)
- Subsampling support for training budget experiments

### Preprocessing
- Intensity scaling: [0, 255] → [0, 1]
- Z-score normalization using training set statistics
- Image reshaping to fixed size (28, 28)

### Feature Extraction (Model A)
- **Raw**: Flattened pixel values (784 features)
- **PCA**: Principal Component Analysis with configurable components
- **HOG**: Histogram of Oriented Gradients features

### Data Augmentation
- Random rotations (±10 degrees)
- Random translations/shifts
- Horizontal flips
- Gaussian noise
- Brightness and contrast adjustments
- Applied only to training set

### Model A (Classical ML)
- SVM classifier with linear, RBF, or polynomial kernels
- Logistic Regression option
- Configurable regularization (C parameter)
- Support for raw and processed features

### Model B (Deep Learning)
- Small ResNet-style architecture
- Configurable model capacity (base_channels parameter)
- Early stopping to prevent overfitting
- Dropout for regularization
- Adaptive learning rate scheduling

### Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrices
- Results saved to JSON/CSV files
- Feature mode comparison for Model A

## Output Files

Results are saved in the `--out_dir` directory (if specified):

- **model_A_results.json**: Model A evaluation results
- **model_B_results.json**: Model B evaluation results
- **feature_comparison_results.csv**: Feature mode comparison (when using `--compare_all_features`)
- **sample_images.png**: Sample images visualization
- **class_distribution.png**: Class distribution plot
- **comparison_summary.md**: Auto-generated Model A vs Model B test-metric summary (written by `main.py`)
- **model_comparison_metrics.png**: Model A vs Model B test metrics bar chart
- **model_comparison_confusion_matrices.png**: Model A vs Model B test confusion matrices
- **model_comparison_roc_curves.png**: Model A vs Model B test ROC curves (when probabilities are available)

### Model B augmentation ablation outputs

When running Model B, the code performs an augmentation ablation (**OFF vs ON**) and writes:

- **augmentation_comparison_detailed.json**: Detailed OFF/ON results (train/val/test metrics, preds, probs, history)
- **augmentation_comparison_results.csv**: Summary table of OFF vs ON metrics by split
- **augmentation_comparison.png**: Test metric bar chart (OFF vs ON)
- **augmentation_confusion_matrices.png**: Test confusion matrices (OFF vs ON)
- **augmentation_roc_curves.png**: Test ROC curves (OFF vs ON)
- **history_without_aug.csv / history_with_aug.csv**: Per-epoch train/val loss+acc
- **history_without_aug.json / history_with_aug.json**: Same as JSON
- **training_curves_without_aug.png / training_curves_with_aug.png**: Learning curves plots
- **checkpoints/without_aug/best_model.pth** and **checkpoints/with_aug/best_model.pth**: Best checkpoints for each run

If ablation is disabled (via `training.run_ablation: false` or `--model_b_ablation off`), Model B runs once and still writes:

- **model_B_results.json**: Single-run schema (`run_type: single_run`)
- **confusion_matrices.png** and **roc_curves.png**: Test plots for the single run

## Reproducibility

The code uses fixed random seeds (default: 42) for reproducibility. This ensures that:
- Dataset subsampling is deterministic
- Model initialization is consistent
- Training results are reproducible

## Expected Runtime and Resources

- **Model A (SVM)**: 
  - Raw features: ~1-5 minutes on CPU
  - PCA features: ~1-5 minutes on CPU
  - HOG features: ~5-15 minutes on CPU
  
- **Model B (Deep Learning)**:
  - Training: ~10-30 minutes on CPU, ~2-5 minutes on GPU
  - GPU is optional but recommended for faster training

## Notes

- The entire test set is always used for final evaluation (as required by the assignment brief)
- Training and validation sets can be subsampled for training budget experiments
- No intermediate or processed data is saved in the `Datasets/` folder
- Pre-trained weights (if any) are stored in `model_A/` or `model_B/` folders

## Example Workflow for Report

1. **Baseline Experiment** (small capacity, no augmentation, small training budget):
   ```bash
   # Model A: Small C, raw features, 25% training data
   python main.py --model A --mode train --train_fraction 0.25 --feature_mode raw --out_dir results/baseline_A
   
   # Model B: Small base_channels, no augmentation, 25% training data
   # (Edit config to disable augmentation and set base_channels=8)
   python main.py --model B --mode train --train_fraction 0.25 --config model_B/config.yaml --out_dir results/baseline_B
   ```

2. **Increased Capacity**:
   ```bash
   # Model A: Larger C, PCA features
   python main.py --model A --mode train --feature_mode pca --out_dir results/increased_capacity_A
   
   # Model B: Larger base_channels
   # (Edit config to set base_channels=32)
   python main.py --model B --mode train --config model_B/config.yaml --out_dir results/increased_capacity_B
   ```

3. **With Augmentation**:
   ```bash
   # Model B: Enable augmentation
   # (Edit config to enable augmentation)
   python main.py --model B --mode train --config model_B/config.yaml --out_dir results/with_augmentation_B
   ```

4. **Larger Training Budget**:
   ```bash
   # Use full training set
   python main.py --model A --mode train --train_fraction 1.0 --out_dir results/full_training_A
   python main.py --model B --mode train --train_fraction 1.0 --out_dir results/full_training_B
   ```

## License

This project is created for academic purposes as part of the ELEC0134 AMLS assignment.

