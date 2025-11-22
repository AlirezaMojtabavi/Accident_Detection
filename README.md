# Video-Based Accident Prediction with Deep Learning (Nexar)

This project predicts whether a traffic collision will occur in short dashcam video clips from the **Nexar Collision Prediction** dataset. The model combines frame-level visual features from a CNN with temporal modeling and contextual metadata (e.g., lighting, weather, scene) to estimate collision risk.

---

## Project Overview

**Goal:**  
Given a few seconds of dashcam footage + basic scene metadata, output the probability that a collision occurs in that clip.

**Approach (high level):**

- Sample frames from each video clip and preprocess them (resize, normalize).
- Extract metadata such as `light_condition`, `weather`, and `scene` and encode them as numeric features.
- Build a hybrid model:
  - **Visual branch:** `TimeDistributed(MobileNetV2)` over frames to get per-frame embeddings.
  - **Temporal branch:** bi-directional GRU over the sequence of frame embeddings.
  - **Metadata branch:** small dense network over encoded metadata.
  - Concatenate temporal + metadata features and feed into fully-connected layers for binary classification.
- Train in **multiple stages**:
  1. **Stage 1:** Train only the top classifier layers with the CNN backbone frozen.
  2. **Stage 2:** Partially unfreeze the last layers of MobileNetV2 (fine-tuning).
  3. (Optional) **Stage 3:** Further fine-tuning with more layers unfrozen and a smaller learning rate.
- Use different **data augmentation regimes** and compare them (light vs. heavy augmentation).

**Best model summary:**

- Architecture: TimeDistributed MobileNetV2 + bi-GRU + metadata fusion.
- Training: multi-stage fine-tuning with **light augmentation** and partial CNN unfreezing.
- Metrics (held-out test set, balanced classes):
  - **Test AUC:** ~0.8  
  - **Macro F1:** ~0.75  
  - **Accuracy:** ~0.75  

---

## Repository Structure

The main workflow is organized into Jupyter notebooks:

- `Video_frame_generator.ipynb`  
  - Loads the Nexar dataset and accesses the original video clips.
  - Extracts a sequence of frames per clip using OpenCV.
  - Resizes frames to the input size expected by MobileNetV2.
  - Prepares and saves frame tensors so they can be reused by the model notebook.

- `Data_augmentation.ipynb`  
  - Experiments with different `ImageDataGenerator` settings (light / medium / heavy augmentation).
  - Includes transformations such as shifts, zoom, brightness changes, rotation, shear, etc.
  - Compares augmentation strategies by training a frozen CNN classifier (Stage 1) and tracking validation loss / AUC.
  - Shows that overly heavy augmentation can hurt validation performance compared to a lighter regime.

- `Metadata.ipynb`  
  - Loads Nexar clip-level metadata.
  - Cleans and filters columns (e.g., `light_condition`, `weather`, `scene`).
  - Maps raw string categories to a smaller, consistent set of classes (e.g., merging rare categories).
  - Encodes metadata as numeric features and aligns them with the frame tensors.
  - Saves processed metadata (and often train/val/test splits) as pickled files for the model notebook.

- `Model.ipynb`  
  - Loads the preprocessed frames (`X_train`, `X_val`, `X_test`) and metadata (`meta_train`, `meta_val`, `meta_test`).
  - Defines the hybrid model:
    - `TimeDistributed(MobileNetV2(include_top=False))` as the frame encoder.
    - bi-GRU (or GRU) to capture temporal dynamics.
    - Dense layers for metadata.
    - Fusion of video + metadata features followed by a binary classifier head.
  - Implements the **multi-stage training** strategy:
    - **Stage 1:** Train top layers with CNN frozen.
    - **Stage 2:** Unfreeze the last *N* CNN layers (excluding BatchNorm) and fine-tune with a low learning rate.
    - **Stage 3 (optional):** Unfreeze more layers with an even lower learning rate.
  - Uses callbacks such as `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` with **validation AUC** as the main monitor.
  - Evaluates the best model on the held-out test set and computes:
    - AUC, accuracy, macro F1, confusion matrix, and classification report.

---

## Dataset

- **Name:** Nexar Collision Prediction dataset  
- **Content:** short dashcam video clips with binary labels (`collision` / `no collision`) and clip-level metadata.  
- **Source:** typically accessed via Hugging Face or directly from the original Nexar release.

The notebooks assume you have:

- Downloaded the dataset (videos + metadata).
- Updated any local paths in the notebooks to point to your data location.

> ⚠️ Some paths inside the notebooks are user-specific (e.g., local directories or Hugging Face cache paths).  
> Make sure to adapt them to your environment before running.

---

## Model Architecture

- **Visual branch:**
  - Input: sequence of frames (e.g., `T` frames of size 224×224×3).
  - Preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input`.
  - Backbone: `TimeDistributed(MobileNetV2(include_top=False, weights="imagenet"))`.
  - Global average pooling per frame to get one feature vector per timestep.

- **Temporal branch:**
  - Bi-directional GRU (or GRU) over the sequence of frame features.
  - Outputs a clip-level embedding that captures motion and temporal patterns (e.g., approaching objects, sudden deceleration).

- **Metadata branch:**
  - Input: encoded vector of metadata features (light condition, scene type, weather, etc.).
  - One or more dense layers with non-linear activations.

- **Fusion + classifier:**
  - Concatenate temporal embedding and metadata embedding.
  - Dense layers with dropout.
  - Final sigmoid output for collision probability.

---

## Training Strategy

1. **Data splits**
   - Dataset split into **train / validation / test** with a ~50/50 collision / no-collision balance in each split.

2. **Stage 1 – Frozen CNN**
   - Freeze all MobileNetV2 layers.
   - Train only the new top layers (temporal + metadata + classifier).
   - Try different augmentation settings and pick the one with the best **validation loss / AUC**.

3. **Stage 2 – Partial Fine-Tuning**
   - Unfreeze the last *N* convolutional layers (skipping BatchNorm layers) of MobileNetV2.
   - Train with a **small learning rate** (e.g., `5e-5`) and monitor `val_auc`.
   - Use `EarlyStopping` and `ReduceLROnPlateau` to avoid overfitting.

4. **Stage 3 (Optional) – Deeper Fine-Tuning**
   - Optionally unfreeze more layers (e.g., last 14) and continue fine-tuning with an even smaller LR (e.g., `1e-5` or `3e-6`).
   - In practice, the biggest gains came from Stage 2; Stage 3 gave only marginal or no improvements and sometimes slightly hurt validation AUC.

5. **Evaluation**
   - Load the best checkpoint from training (by `val_auc`).
   - Compute:
     - AUC on validation and test sets.
     - Accuracy, precision, recall, macro F1.
     - Confusion matrix.
   - Tune the decision threshold (e.g., 0.4 instead of 0.5) to balance precision and recall across classes.

---

## Results

For the best-performing configuration (light augmentation + partial CNN unfreezing):

- **Test AUC:** ~0.8  
- **Macro F1:** ~0.75  
- **Accuracy:** ~0.75  
- **Class performance (approx.):**
  - Both collision and non-collision classes have similar precision and recall (~0.73–0.75), which is desirable for a balanced dataset.

Exact numbers and plots (loss, accuracy, AUC curves) are available inside `Model.ipynb`.

---

## Requirements

This project uses:

- Python 3.x
- TensorFlow / Keras (2.x)
- NumPy
- pandas
- scikit-learn
- OpenCV (`opencv-python`)
- Matplotlib / Seaborn
- tqdm (optional, for progress bars)
- Jupyter Notebook

You can install the core dependencies via:

```bash
pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib tqdm

