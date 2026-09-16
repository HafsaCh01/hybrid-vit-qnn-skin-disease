#  Hybrid ViT + Quantum Neural Network for Skin Disease Classification
A from-scratch implementation of a **Hybrid Vision Transformer + Quantum Neural Network** for classifying 6 classes of viral skin lesions, built without any pretrained weights. Combines classical deep learning (ViT with multi-head self-attention) with a PennyLane quantum circuit (StronglyEntanglingLayers) as the final decision layer.

---

## Task

**Dataset:** Multi-Class Viral Skin Lesion Dataset (MCVSLD) — 6 classes  
**Goal:** Multi-class image classification using a custom hybrid quantum-classical model  
**Constraint:** No pretrained models — architecture built entirely from scratch

---

##  Repository Structure

```
hybrid-vit-qnn-skin-disease/
│
├── Codefile.ipynb     # Full pipeline notebook (8 sections)
└── README.md
```

> Dataset is loaded from Google Drive — see Section 2 of the notebook for setup instructions.

---

##  Model Architecture

```
Input Image (224×224×3)
       ↓
 PatchEmbedding          16×16 patches → Linear projection → 128-dim
       ↓
 ViT Encoder             2× TransformerEncoderBlock
                         (LayerNorm → MultiHeadSelfAttention(4 heads) → MLP → Dropout)
       ↓
 CLS Token               128-dim representation
       ↓
 Linear Reduce           128 → 6 (one value per qubit)
 BatchNorm               stabilize quantum inputs
       ↓
 Quantum Circuit          6 qubits | RY angle encoding
  (PennyLane)            StronglyEntanglingLayers (4 layers)
                         Pauli-Z expectation measurement
       ↓
 Dropout (0.3)
       ↓
 Linear Classifier        6 → num_classes
       ↓
 Output (6 classes)
```

---

##  Pipeline — 8 Sections

| Section | Content |
|---|---|
| **0** | Environment setup — PyTorch, CUDA check, PennyLane install |
| **1** | Imports & global config (batch size, epochs, qubits, LR, seed) |
| **2** | Google Drive mount, dataset path validation |
| **3** | Dataset pipeline — image loading, augmentation, `WeightedRandomSampler`, class weight tensor |
| **4** | Model architecture — `PatchEmbedding`, `MultiHeadSelfAttention`, `TransformerEncoderBlock`, `ViT`, quantum circuit, `HybridModel` |
| **5** | Training pipeline — loss, optimizer, scheduler, early stopping |
| **6** | Training curves — loss & accuracy over epochs |
| **7** | Test set evaluation — accuracy, classification report, confusion matrix |
| **8** | XAI — prediction visualization, CLS token activation analysis, per-class accuracy bar chart |

---

##  Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch Size | 16 |
| Learning Rate | 3e-4 (AdamW) |
| Weight Decay | 1e-4 |
| Label Smoothing | 0.1 |
| Qubits | 6 |
| Quantum Layers | 4 (StronglyEntanglingLayers) |
| Scheduler | CosineAnnealingLR |
| Early Stopping Patience | 8 epochs |
| Max Samples/Class | 825 |

---

##  Anti-Overfitting Strategy

| Technique | Detail |
|---|---|
| Label Smoothing | `CrossEntropyLoss(label_smoothing=0.1)` |
| Weighted Loss | Inverse class frequency weights |
| Dropout | 0.15 in attention blocks, 0.3 before classifier |
| Weight Decay | AdamW L2 regularization |
| LR Scheduling | CosineAnnealingLR |
| Early Stopping | Best weights saved, training halts on no improvement |
| Data Augmentation | Flip, rotation ±20°, color jitter, random erasing |
| Balanced Sampling | `WeightedRandomSampler` per class |

---

##  Improvements Over Baseline

| Problem | Fix Applied |
|---|---|
| Only 4 classes loaded | Updated loader → all 6 classes |
| 10 epochs, ~61% train acc | 30 epochs + CosineAnnealingLR |
| Overfitting (large train-val gap) | Dropout + label smoothing + weight decay |
| Class imbalance ignored | WeightedRandomSampler + weighted loss |
| Weak augmentation | Flip, rotation, color jitter, random erasing |
| No quantum input stabilization | BatchNorm before quantum layer |
| CUDA/CPU device mismatch | All tensors kept on CPU (PennyLane requirement) |

---

##  XAI (Explainability)

Three visualization techniques included:
- **Prediction Grid** — test images with true vs predicted labels (green = correct, red = wrong)
- **CLS Token Activation Analysis** — L2 norm of ViT's CLS token per sample, colored by true class
- **Per-Class Accuracy Bar Chart** — green ≥70%, orange ≥50%, red <50%

---

## Getting Started

### 1. Install dependencies (Colab)
```python
!pip install pennylane -q
```
Standard libraries (PyTorch, torchvision, sklearn, matplotlib, seaborn) are pre-installed on Colab.

### 2. Dataset setup
Upload your MCVSLD dataset to Google Drive at:
```
MyDrive/Colab Notebooks/Skin Lesion Dataset/
├── train/
│   ├── class_1/
│   └── ...
├── val/
└── test/
```

### 3. Run
Open `Codefile.ipynb` in Google Colab and run all cells top to bottom.

---

##  Stack

`Python` · `PyTorch` · `PennyLane` · `torchvision` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Google Colab`

---

**Institute of Space Technology (IST), Islamabad — ML Lab Final**
