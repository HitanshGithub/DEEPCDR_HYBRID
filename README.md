#  DeepCDR-Hybrid  
### Transformer-Enhanced Multi-Modal Deep Learning for Cancer Drug Response Prediction

DeepCDR-Hybrid is a **state-of-the-art deep learning framework** for **predicting cancer drug response (IC50)** by integrating **multi-omics genomic data** with **hybrid drug representations**.  
The model leverages **transformer architectures**, **CNN-based molecular encoders**, **ChemBERTa embeddings**, and **explainable AI (XAI)** techniques to achieve **high predictive accuracy and interpretability**.

This repository contains the **Google Colab–ready implementation** used in the accompanying research work.

---

##  Problem Statement

Predicting how a cancer cell line responds to a drug is a **central challenge in precision oncology**.  
Traditional machine learning models struggle to capture:

- High-dimensional **multi-omics data**
- Complex **drug–genome interactions**
- Non-linear and context-dependent relationships

DeepCDR-Hybrid addresses these challenges using **transformer-based multi-omics encoding** and **hybrid drug representations**, achieving **significant performance improvements** over existing methods.

---

##  Research Overview

- **Task:** Regression (IC50 drug response prediction)
- **Prediction Target:** Log-transformed IC50
- **Learning Paradigm:** Supervised deep learning
- **Evaluation:** Held-out test set (no data leakage)

###  Key Contributions

1. **Transformer-Based Multi-Omics Encoder**
   - Integrates gene expression, DNA methylation, and mutation data
   - Uses multi-head self-attention and squeeze–excitation blocks

2. **Hybrid Drug Representation**
   - Structural features via **Morgan fingerprints (RDKit)**
   - Semantic features via **ChemBERTa language model embeddings**

3. **Bidirectional Cross-Attention Fusion**
   - Dynamically integrates CNN and LLM drug features

4. **Explainable AI (XAI)**
   - Permutation importance
   - Integrated gradients
   - Attention visualization
   - Sensitivity and interaction analysis

---

##  Datasets Used

###  Genomic Data
- **Gene Expression:** RNA-seq (697 genes)
- **DNA Methylation:** 808 CpG sites
- **Mutations:** 34,673 binary mutation features

###  Drug Data
- **SMILES strings** for 223 anti-cancer drugs
- Featurized using:
  - Morgan fingerprints (2048 bits)
  - ChemBERTa embeddings (768-dim)

###  Sources
- **Cancer Cell Line Encyclopedia (CCLE)**
- **Genomics of Drug Sensitivity in Cancer (GDSC)**

###  Data Split
| Split | Percentage |
|-----|-----------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

##  Model Architecture

### 1️ Multi-Omics Transformer Encoder
- PCA-reduced omics inputs (150 PCs each)
- Stacked as a sequence
- 2 Transformer layers, 4 attention heads
- Global max + average pooling
- Squeeze–Excitation for feature recalibration

### 2️ Hybrid Drug Encoder
**CNN Branch (Structural):**
- Multi-scale 1D convolutions on Morgan fingerprints
- Spatial attention + pooling

**LLM Branch (Semantic):**
- ChemBERTa embeddings
- Residual dense blocks
- Self-attention layer

### 3️ Cross-Modal Fusion
- Bidirectional cross-attention (CNN ↔ LLM)
- Gated fusion mechanism

### 4️ Prediction Head
- Fully connected layers with residual connections
- Output: predicted IC50 value

---

##  Results

###  Test Set Performance

| Metric | Value |
|-----|------|
| RMSE | **0.4289** |
| MAE | **0.3065** |
| R² | **0.9721** |
| Pearson Correlation | **0.9869** |
| Spearman Correlation | **0.9836** |

###  Comparison with Baselines

| Model | RMSE |
|-----|-----|
| Ridge Regression | 2.368 |
| Random Forest | 2.270 |
| CDRscan | 1.982 |
| tCNNs | 1.782 |
| DeepCDR (Base) | 1.058 |
| **DeepCDR-Hybrid (Ours)** | **0.4289** |

 **53.7% RMSE reduction** over DeepCDR.

---

##  Explainable AI (XAI) Analysis

###  Permutation Importance
- Drug Morgan fingerprints: **~77.5% contribution**
- Drug LLM embeddings: **~13.8%**
- Genomic features act as contextual modifiers

###  Sensitivity Analysis
- Drug structure highly sensitive
- Genomic features robust to noise

###  Attention Analysis
- ChemBERTa captures key pharmacophore patterns
- Cross-attention validates complementary drug representations

---

##  Tech Stack & Tools

###  Programming
- Python 3.9+

###  Deep Learning
- TensorFlow / Keras
- Mixed Precision Training (FP16)

###  Cheminformatics
- RDKit
- DeepChem

###  Data Science
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

###  NLP / Transformers
- HuggingFace Transformers
- ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`)
- Accelerate
- Tokenizers

###  Platform
- Google Colab
- Google Drive (for results & checkpoints)

---

##  Repository Structure

