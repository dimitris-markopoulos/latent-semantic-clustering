run the following code for package dependencies

pip install -r requirements.txt

# 📚 Unsupervised Author Text Analysis

This project applies a variety of unsupervised machine learning techniques to analyze and cluster book chapters written by four English-language authors. The goal is to uncover structure in the data and identify meaningful patterns in word usage without using author labels during training.

---

## 🧠 Problem Overview

The dataset consists of word count vectors from book chapters by four authors. Common stop words are treated as features. The problem is tackled through several stages:

1. **Visual Representation of Data**
2. **Clustering and Comparison of Algorithms**
3. **Pattern Recognition**
4. **Model Validation and Stability Analysis**

---

## 📦 Dataset

- 📁 Source: [clustRviz GitHub Repo](https://github.com/DataSlingers/clustRviz/tree/master/data)
- 📄 File: `authors.rda`
- 🎯 Description: Word count data for stop words in chapters, with ground-truth author labels (used only for validation).

---

## 🔧 Techniques Used

### 🔹 Dimensionality Reduction

- Principal Component Analysis (PCA)
- Non-negative Matrix Factorization (NMF)
- Multi-Dimensional Scaling (MDS)
- Spectral Embedding
- Uniform Manifold Approximation and Projection (UMAP)
- Biclustering

### 🔹 Clustering Algorithms

- K-Means Clustering
- Hierarchical Clustering (various linkages and distances)
- Gaussian Mixture Models (GMM)
- Spectral Clustering

### 🔹 Pattern Recognition & Validation

- Identification of discriminative word features
- Evaluation of clustering stability and generalizability
- Tuning of cluster count \( K \) using unsupervised criteria
- Accuracy assessment with respect to true author labels (post-hoc)

---

## 📊 Deliverables

- Visual summaries of both observations and features
- Cluster assignment plots with true label comparisons
- Evaluation of which clustering + dimensionality reduction combinations are most interpretable and effective
- Justification and reflection on method selection and performance

---

## 🧪 Requirements

- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyreadr`, `umap-learn`
