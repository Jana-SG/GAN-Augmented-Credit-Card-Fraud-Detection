# GAN-Augmented Credit Card Fraud Detection

**Improving Minority-Class Detection Using Generative Adversarial Networks**

---

## 📌 Overview

Credit card fraud detection is an extremely imbalanced classification problem where fraudulent transactions represent only **0.17% of all samples (578:1 ratio)**. Traditional classifiers often achieve high overall accuracy but fail in practical deployment due to excessive false positives.

This project explores **GAN-based data augmentation** to synthetically generate fraudulent transactions and improve downstream classifier performance.

Three GAN variants are implemented and compared:

- **Vanilla GAN**
- **WGAN-GP (Wasserstein GAN with Gradient Penalty)**
- **Conditional GAN (CGAN)**

The impact of synthetic balancing is evaluated using a Logistic Regression classifier.

---

## 📊 Dataset

- **Source:** Kaggle Credit Card Fraud Detection dataset (ULB)
- **Total samples:** 284,807 transactions
- **Fraud cases:** 492 (0.172%)
- **Features:** 30 numerical features (PCA components V1–V28 + Time + Amount)

### Data Split (Stratified)

- Train: 64%
- Validation: 16%
- Test: 20%

GAN models were trained exclusively on training data.  
The test set remained untouched for unbiased evaluation.

---

## 🧠 GAN Architectures

All models use fully connected neural networks for tabular data generation.

### 1️⃣ Vanilla GAN
- Latent dimension: 64  
- Loss: `BCEWithLogitsLoss`  
- Optimizer: Adam (lr = 2e-4, β = (0.5, 0.999))  
- 300 epochs  
- Trained only on fraudulent samples  

---

### 2️⃣ WGAN-GP
- Wasserstein loss with Gradient Penalty (λ = 10)  
- 5 critic updates per generator update  
- Adam (lr = 1e-4, β = (0.0, 0.9))  
- 300 epochs  
- Improved training stability and reduced mode collapse  

---

### 3️⃣ Conditional GAN (CGAN)
- Label-conditioned generation  
- 8-dimensional label embedding  
- Latent + label concatenation  
- Trained on full imbalanced dataset  

---

## ⚖️ Training Scenarios

Four classifier configurations were evaluated:

1. **Original Imbalanced Dataset (Baseline)**
2. **Vanilla GAN Balanced**
3. **WGAN-GP Balanced**
4. **CGAN Balanced**

Synthetic fraud samples were generated to fully balance the training set.

### Classifier

- Logistic Regression (`scikit-learn`)
- `class_weight = 'balanced'`
- `max_iter = 2000`
- Fixed threshold = 0.5
- Same configuration across all experiments

---

## 📈 Results

| Model | PR-AUC | Fraud Recall | Fraud Precision | Fraud F1 |
|-------|--------|--------------|-----------------|----------|
| Baseline (Imbalanced) | 0.668 | 0.916 | 0.065 | 0.12 |
| Vanilla GAN | 0.676 | 0.758 | 0.783 | 0.77 |
| **WGAN-GP** | **0.707** | **0.80** | **~0.76** | **0.776** |
| CGAN | ~0.69 | 0.684 | 0.774 | 0.726 |

---

## 🔥 Key Improvements

- Fraud precision improved from **6.5% → over 75%**
- False positives reduced by **98%**
- WGAN-GP achieved best overall precision-recall tradeoff
- Significant improvement in PR-AUC (critical metric for imbalanced datasets)
- GAN-balanced models showed smoother validation loss and better generalization

WGAN-GP demonstrated the strongest stability and minority-class modeling capability.

---
## Technologies Used

- Python
- PyTorch (GAN implementation)
- scikit-learn
- NumPy
- pandas
- Matplotlib / Seaborn

---

## 📁 Repository Structure
```
├── fraud_detection_with_gans.ipynb 
├── Report.pdf 
└── README.md
```
---

## 📚 Report

A complete technical report including:

- Architecture details  
- Hyperparameter configurations  
- Loss curves  
- Confusion matrices  
- Detailed comparative analysis  

is available in:

**`Report.pdf`**

---
