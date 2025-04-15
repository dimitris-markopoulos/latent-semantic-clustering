import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

class ClusterEvaluator:
    def __init__(self, pred_clusters, true_labels):
        self.df = pd.DataFrame({'Cluster': pred_clusters, 'Label': true_labels})
        self.mapping = self.df.groupby('Cluster')['Label'].agg(lambda x: x.mode()[0]).to_dict()
        
    @property
    def accuracy(self):
        predicted = self.df['Cluster'].map(self.mapping)
        return (predicted == self.df['Label']).mean(), self.mapping

class SilhouetteEvaluator:
    def __init__(self, X, model_class, k_range=range(2, 11), **model_kwargs):
        self.X = X
        self.model_class = model_class
        self.k_range = k_range
        self.model_kwargs = model_kwargs
        self.ave_silhouette = {}
        self.best_k = None

    def evaluate(self):
        for k in self.k_range:
            model = self.model_class(n_clusters=k, **self.model_kwargs)
            labels = model.fit_predict(self.X)
            score = silhouette_score(self.X, labels)
            self.ave_silhouette[k] = score
        self.best_k = max(self.ave_silhouette, key=self.ave_silhouette.get)
        return self.ave_silhouette, self.best_k

    def plot(self, title, ax=None, color='r'):
        if not self.ave_silhouette:
            raise RuntimeError("Call `.evaluate()` before plotting.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.ave_silhouette.keys(), self.ave_silhouette.values(), marker='o')
        ax.axvline(self.best_k, linestyle='--', color=color, label=f'Best K = {self.best_k}')
        ax.set_title(f"{title}: Silhouette Score vs. K")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.legend()

def make_hierarchical(linkage, metric):
    return lambda n_clusters: AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)

def make_gmm(**kwargs):
    return lambda n_clusters: GaussianMixture(n_components=n_clusters, **kwargs)

def make_spectral(**kwargs):
    return lambda n_clusters: SpectralClustering(n_clusters=n_clusters, **kwargs)

class GaussianMixtureEM:
    def __init__(self, K, num_iterations, allow_singular=True):
        self.K = K
        self.num_iterations = num_iterations
        self.allow_singular = allow_singular

    def fit(self, X):
        epsilon = 1e-6  # regularization constant
        X_array = X.to_numpy()
        n_rows, n_cols = X.shape

        # Initialization
        means = X.sample(n=self.K).to_numpy()
        shared_cov = np.cov(X_array, rowvar=False, ddof=1)
        cov = [shared_cov.copy() for _ in range(self.K)]
        pis = [1 / self.K] * self.K
        gamma = np.zeros((n_rows, self.K))

        pis_dict = {'Initial': [pis.copy()]}
        pis_dict.update({f'Iteration_{i}': [] for i in range(self.num_iterations)})

        for iter in range(self.num_iterations):
            # E-step
            for i in range(n_rows):
                xi = X.iloc[i].values
                denom = 0
                for k in range(self.K):
                    numerator = pis[k] * multivariate_normal.pdf(
                        xi, mean=means[k], cov=cov[k], allow_singular=self.allow_singular
                    )
                    gamma[i, k] = numerator
                    denom += numerator
                gamma[i, :] /= denom

            # M-step
            Nk = [np.sum(gamma[:, j]) for j in range(self.K)]

            means = [
                np.sum([gamma[i, k] * X_array[i, :] for i in range(n_rows)], axis=0) / Nk[k]
                for k in range(self.K)
            ]

            cov = [
                np.sum(
                    [gamma[i, k] * np.outer(X_array[i, :] - means[k], X_array[i, :] - means[k])
                     for i in range(n_rows)],
                    axis=0
                ) / Nk[k] + epsilon * np.eye(n_cols)
                for k in range(self.K)
            ]

            pis = np.array(Nk) / n_rows
            pis_dict[f'Iteration_{iter}'].append(pis.copy())

        return {'pis_dict': pis_dict, 'Nk': Nk, 'means': means, 'cov': cov}