# GAN-Augmented Credit Card Fraud Detection  
*Exploring GAN-Based Data Augmentation for Class-Imbalanced Credit Card Fraud Detection*

---

##  Repository Structure
├── fraud_detection_with_gans.ipynb 
├── Report.pdf 
└── README.md


---

##  Notebook Overview
The notebook is organized into the following logical sections:

###  Data Loading & Preprocessing
###  GAN Training
Three GAN variants are trained:
- **Vanilla GAN**  
- **WGAN-GP (Wasserstein GAN with Gradient Penalty)**  
- **Conditional GAN (CGAN)**  
Each model learns to generate synthetic fraudulent transactions.

###  Dataset Balancing
### Classifier Training
- A **Logistic Regression** classifier is trained 
###  Evaluation
Models are evaluated using:
- ROC-AUC  
- PR-AUC  
- Precision, Recall, F1 (fraud class)  
- Confusion matrices  
- Training and validation loss curves  

---

##  Report

The full methodology, architecture details, results, and analysis are available in:

**`Report.pdf`**

---


