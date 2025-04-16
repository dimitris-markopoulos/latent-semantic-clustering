# 📁 Media Folder

This folder contains supporting visual materials and reference notebooks used throughout the project. It serves as a centralized archive for graphs, figures, and lightweight illustrative implementations that are not part of the main pipeline.

---

## 📊 Stored Graphs

All exported figures across notebooks (e.g., silhouette score plots, cluster visualizations, PCA projections, stability analyses) are stored here for:

- Easy access during reporting
- Keeping notebooks clean and modular
- Reuse across slides, reports, or publications

---

## 🧠 Reference Example: EM on Univariate Gaussian Mixture

📄 `em_univariate_demo.ipynb`

A self-contained notebook implementing the EM algorithm for a **two-component univariate Gaussian mixture model**, adapted from *The Elements of Statistical Learning (ESL)*.

**Highlights:**
- Implements E-step and M-step manually
- Demonstrates convergence of parameters (π, μ₁, μ₂, σ₁², σ₂²)
- Visualizes learning dynamics in a toy dataset
- Serves as a sandbox to debug or explain EM step-by-step

---

## 🧼 Folder Philosophy

- This folder **is not meant for code modules or final outputs**.
- It’s an organized space for **supporting artifacts**: demos, visuals, and one-off reference material.

---

Feel free to ignore this folder unless you're reviewing visual outputs or internal derivations.
