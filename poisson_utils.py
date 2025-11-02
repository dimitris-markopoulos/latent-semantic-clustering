import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

def poisson_mixture_em(X, K, max_iter=100, tol=1e-4, eps=1e-10, random_state=42):
    np.random.seed(random_state)
    X = np.asarray(X)
    N, d = X.shape

    # K-means initialization
    kmeans = KMeans(n_clusters=K, random_state=random_state).fit(X)
    w = np.bincount(kmeans.labels_) / N
    lam = np.vstack([X[kmeans.labels_ == k].mean(axis=0) for k in range(K)]) + eps
    
    for _ in range(max_iter):
        # E-step (log-space)
        log_pi = np.zeros((N, K))
        for k in range(K):
            log_pi[:, k] = (
                np.log(w[k] + eps) +
                (X * np.log(lam[k] + eps)).sum(axis=1) -
                lam[k].sum()
            )

        # log-sum-exp for numeric stability
        m = log_pi.max(axis=1, keepdims=True)
        pi = np.exp(log_pi - m)
        pi /= pi.sum(axis=1, keepdims=True)

        # M-step
        Nk = pi.sum(axis=0) + eps
        w_new = Nk / N
        
        lam_new = (pi.T @ X) / Nk[:, None]

        # convergence check
        if np.max(np.abs(lam_new - lam)) < tol:
            break

        lam = lam_new
        w = w_new

    return w, lam, pi

def plot_word_heatmap(lam, feature_names, top_n):
    K, d = lam.shape

    # Collect union of top words
    top_words = set()
    for k in range(K):
        idx = np.argsort(lam[k])[::-1][:top_n]
        top_words.update(feature_names[i] for i in idx)

    top_words = list(top_words)

    # Map words → column indices
    word_idx = [np.where(feature_names == w)[0][0] for w in top_words]

    # Build matrix (clusters × selected words)
    lam_subset = lam[:, word_idx]

    # Put into dataframe
    df = pd.DataFrame(
        lam_subset,
        index=[f"Cluster {k}" for k in range(K)],
        columns=top_words
    )

    df_norm = df.div(df.max(axis=1), axis=0)

    plt.figure(figsize=(16,5))
    sns.heatmap(df_norm, cmap="viridis", annot=False)
    if top_n >= 69:
        plt.title(f"All Poisson λ Words per Cluster (normalized)")
    else:
        plt.title(f"Top {top_n} Poisson λ Words per Cluster (normalized)")
    plt.show()
