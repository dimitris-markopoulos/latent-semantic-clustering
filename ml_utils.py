import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

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